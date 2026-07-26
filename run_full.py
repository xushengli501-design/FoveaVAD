#!/usr/bin/env python3
"""Full UCF-Crime Kappa-VAD evaluation. Run on one GPU split."""
import torch, numpy as np, sys, os, time, argparse, pickle
from sklearn.metrics import roc_auc_score

_PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJ)
sys.path.insert(0, '/sdb/data_public/llms/llm/InternVL2-8B')
sys.path.insert(0, os.path.join(os.path.dirname(_PROJ), 'CoReVAD', 'utils'))

from PIL import Image; import torchvision.transforms as T
from data_pipeline import load_video_segment, find_video_path, preprocess_for_vit
from pipeline import SAGEPipeline, SAGEConfig
from scpage.phase_encode import PhaseEncoder
from scpage.instability import compute_phase_instability
from scpage.crop_utils import search_max_energy_box, compute_padding_mask, patch_box_to_pixel, expand_and_square_box
from scpage import _resize_for_internvl
from htag.nodes import build_evidence_nodes
from htag.graph import build_compact_graph, build_stage_mask, forward_one_stage, forward_batched
from decord import VideoReader, cpu as dcpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--split', type=str, required=True, help='i,N format, e.g. 0,4')
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--max_per_video', type=int, default=30)
    parser.add_argument('--out', type=str, default='/tmp/kappa_full')
    args = parser.parse_args()

    i_split, n_split = [int(x) for x in args.split.split(',')]
    device = torch.device(f'cuda:{args.gpu}')
    print(f'GPU {args.gpu}, split {i_split}/{n_split}')

    # Init models
    cfg = SAGEConfig(); pipe = SAGEPipeline(cfg); pipe._ensure_models()
    tt = T.ToTensor(); w = pipe.internvl; dev = w.device; dt = w.dtype
    enc = PhaseEncoder(device='cuda'); BOX = 5
    grp = {'P': [0,1], 'T': [2,3], 'F1': [4,5], 'F2': [6,7]}

    def sc_box(frame):
        ow, oh = Image.fromarray(frame).size; vi = preprocess_for_vit(frame)
        e = enc.encode_phases(vi); H, _, jv = compute_phase_instability(e['features'], e['valid_mask'])
        pm = compute_padding_mask(ow, oh).to(dev); sm = jv & pm
        br, _ = search_max_energy_box(H, box_size=BOX, valid_mask=sm)
        if br is None: return (0, 0, ow, oh)
        xm, ym, xM, yM = patch_box_to_pixel(br, box_size=BOX, phase_offset=(0,0))
        xe, ye, xMe, yMe = expand_and_square_box(xm, ym, xM, yM, expand_ratio=1.10, image_w=224, image_h=224)
        s = min(224/ow, 224/oh); nw, nh = int(ow*s), int(oh*s)
        px, py = (224-nw)//2, (224-nh)//2
        xo = max(0, min(int((xe-px)/s), ow-1)); yo = max(0, min(int((ye-py)/s), oh-1))
        xoM = max(xo+1, min(int((xMe-px)/s), ow)); yoM = max(yo+1, min(int((yMe-py)/s), oh))
        return (xo, yo, xoM-xo, yoM-yo)

    def signals(wf):
        sp = [wf[i] for i in [0,4,8,12,17,21,25,29]]
        st = [tt(Image.fromarray(f).convert('RGB')) for f in sp]
        gis = [_resize_for_internvl(t, 448) for t in st]
        gv = w.encode_images(torch.cat(gis, 0).to(dev)); gt = list(gv[:8])
        box = sc_box(wf[11])
        tf = [sp[2], sp[3]]
        tc = [T.functional.crop(torch.from_numpy(f).permute(2,0,1).float()/255.0, box[1], box[0], box[3], box[2]) for f in tf]
        tc = [_resize_for_internvl(c, 448) for c in tc]
        lf = [torch.zeros(1,3,448,448) for _ in range(8)]; lf[2] = tc[0]; lf[3] = tc[1]
        lv = w.encode_images(torch.cat(gis+lf, 0).to(dev)); lt = list(lv[8:])

        # Single G+L graph; modulate modality visibility via inactive_modalities
        nodes, metas = build_evidence_nodes(w, gt, lt, grp, active_modalities={'G','L'}, token_mode='wmean')
        graph, seg = build_compact_graph(w, nodes, metas)
        dec = seg['decision_pos']; pos_ids = seg['position_ids']

        # 4 masks for 4 evidence conditions, batched into 1 forward
        mask_vB    = build_stage_mask(seg, ['P','T'],        dev, dt, inactive_modalities={'L'})
        mask_vBL   = build_stage_mask(seg, ['P','T'],        dev, dt)
        mask_vBF1  = build_stage_mask(seg, ['P','T','F1'],   dev, dt, inactive_modalities={'L'})
        mask_vBLF1 = build_stage_mask(seg, ['P','T','F1'],   dev, dt)

        z_vals = forward_batched(w, graph, [mask_vB, mask_vBL, mask_vBF1, mask_vBLF1], dec, pos_ids)
        vB, vBL, vBF1, vBLF1 = z_vals
        return vB, vBLF1-vBL-vBF1+vB

    # Load video list + GT
    cache = np.load('/tmp/ucf_cache.npy', allow_pickle=True).item()
    TEST_LIST = '/sdb/data_public/llms/videos/UCFcrime/Anomaly_Detection_splits/Anomaly_Test.txt'
    with open(TEST_LIST) as f:
        order_all = [l.strip().replace('.mp4','').split('/')[-1] for l in f if l.strip()]
    order_all = [v for v in order_all if v in cache]

    # Split videos
    n_total = len(order_all)
    chunk = (n_total + n_split - 1) // n_split
    my_videos = order_all[i_split*chunk : min((i_split+1)*chunk, n_total)]
    print(f'My videos: {len(my_videos)} (total={n_total})')

    gt_full = np.load(os.path.join(os.path.dirname(_PROJ), 'CoReVAD/src/ucf/gt_ucf.npy')).astype(np.int8)
    gt_offsets = {}; off = 0
    for vname in order_all:
        if vname in cache: gt_offsets[vname] = off; off += cache[vname]['L']

    all_vb = []; all_k1 = []; all_labels = []
    t0 = time.time(); n_done = 0

    for vname in my_videos:
        if vname not in cache: continue
        vpath = find_video_path(vname)
        if vpath is None: continue
        try: vr = VideoReader(vpath, ctx=dcpu(0), num_threads=1); nf = len(vr)
        except: continue
        gt_off = gt_offsets.get(vname, 0)

        vbw = []; k1w = []; labs = []
        for s in range(0, nf-30, args.stride):
            ts, te = s+8, s+14
            if te >= nf: continue
            center = (ts+te)//2
            ia = int(gt_full[min(gt_off+center, len(gt_full)-1)] > 0)
            try:
                wf = load_video_segment(vpath, s, 30)
                vb, k1 = signals(wf)
                vbw.append(vb); k1w.append(k1); labs.append(ia)
            except: pass
            if args.max_per_video > 0 and len(labs) >= args.max_per_video: break

        if len(labs) > 0:
            all_vb.extend(vbw); all_k1.extend(k1w); all_labels.extend(labs)

        n_done += 1
        if n_done % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / n_done * len(my_videos) - elapsed
            print(f'[{n_done}/{len(my_videos)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, {len(all_labels)} windows', flush=True)

    # Save
    out_file = f'{args.out}_gpu{args.gpu}_split{i_split}.pkl'
    data = {'vb': np.array(all_vb), 'k1': np.array(all_k1), 'labels': np.array(all_labels)}
    with open(out_file, 'wb') as f: pickle.dump(data, f)
    print(f'Saved {len(all_labels)} windows to {out_file} ({time.time()-t0:.0f}s)')

if __name__ == '__main__':
    main()
