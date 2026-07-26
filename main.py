#!/usr/bin/env python3
"""
Kappa-VAD: Non-Additive Local-Future Evidence Interaction for Training-Free VAD.

Entry point. Computes kappa = v(B+L+F) - v(B+L) - v(B+F) + v(B) on video windows.

Usage:
    python main.py --video Abuse028_x264 --gpu 0 --stride 4
    python main.py --video Abuse028_x264 --gpu 1 --stride 12 --max_windows 100

Dependencies: InternVL2-8B at /sdb/data_public/llms/llm/InternVL2-8B
              ViT-B/16 at /sdb/data_public/llms/vit-base-patch16-224-in21k
              UCF-Crime videos at /sdb/data_public/llms/videos/UCFcrime/videos
"""
import torch, numpy as np, sys, os, time, argparse
from collections import defaultdict
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
from htag.graph import build_compact_graph, build_stage_mask, forward_one_stage
from decord import VideoReader, cpu as dcpu


class KappaVAD:
    """Complete kappa-based VAD pipeline."""

    def __init__(self, gpu=0):
        self.device = torch.device(f'cuda:{gpu}')
        cfg = SAGEConfig(); pipe = SAGEPipeline(cfg); pipe._ensure_models()
        self.w = pipe.internvl
        self.tt = T.ToTensor()
        self.enc = PhaseEncoder(device='cuda')
        self.BOX = 5
        self.grp = {'P': [0,1], 'T': [2,3], 'F1': [4,5], 'F2': [6,7]}

    def _sc_box(self, frame):
        """Target-SC: compute 5x5 patch box from phase instability."""
        ow, oh = Image.fromarray(frame).size
        vi = preprocess_for_vit(frame)
        e = self.enc.encode_phases(vi)
        H, _, jv = compute_phase_instability(e['features'], e['valid_mask'])
        pm = compute_padding_mask(ow, oh).to(self.w.device)
        sm = jv & pm
        br, _ = search_max_energy_box(H, box_size=self.BOX, valid_mask=sm)
        if br is None:
            return (0, 0, ow, oh)
        xm, ym, xM, yM = patch_box_to_pixel(br, box_size=self.BOX, phase_offset=(0,0))
        xe, ye, xMe, yMe = expand_and_square_box(xm, ym, xM, yM, expand_ratio=1.10, image_w=224, image_h=224)
        s = min(224/ow, 224/oh); nw, nh = int(ow*s), int(oh*s)
        px, py = (224-nw)//2, (224-nh)//2
        xo = max(0, min(int((xe-px)/s), ow-1))
        yo = max(0, min(int((ye-py)/s), oh-1))
        xoM = max(xo+1, min(int((xMe-px)/s), ow))
        yoM = max(yo+1, min(int((yMe-py)/s), oh))
        return (xo, yo, xoM-xo, yoM-yo)

    def _htag_forward(self, gt_tok, lt_tok, visible_groups):
        """Single HTAG forward with given evidence visibility."""
        w, grp, dev, dt = self.w, self.grp, self.w.device, self.w.dtype
        if lt_tok is not None:
            nodes, metas = build_evidence_nodes(w, gt_tok, lt_tok, grp, active_modalities={'G','L'}, token_mode='wmean')
        else:
            nodes, metas = build_evidence_nodes(w, gt_tok, None, grp, active_modalities={'G'}, token_mode='wmean')
        graph, seg = build_compact_graph(w, nodes, metas)
        mask = build_stage_mask(seg, visible_groups, dev, dt)
        z, _, _, _ = forward_one_stage(w, graph, mask, seg['decision_pos'], seg['position_ids'])
        return z, seg

    def process_window(self, wstart, video_path, N):
        """Process one 30-frame window. Returns (vB, vBL, vBF1, vBLF1, label)."""
        wf = load_video_segment(video_path, wstart, 30)
        sp = [wf[i] for i in [0,4,8,12,17,21,25,29]]
        st = [self.tt(Image.fromarray(f).convert('RGB')) for f in sp]
        gis = [_resize_for_internvl(t, 448) for t in st]
        gv = self.w.encode_images(torch.cat(gis, 0).to(self.w.device))
        gt = list(gv[:8])

        # Target-SC: local crops for T block only
        box = self._sc_box(wf[11])
        tf, tt2 = [sp[2], sp[3]], [st[2], st[3]]
        tc = []
        for fr in tf:
            bx, by, bw, bh = box
            c = T.functional.crop(torch.from_numpy(fr).permute(2,0,1).float()/255.0, by, bx, bh, bw)
            tc.append(_resize_for_internvl(c, 448))
        lf = [torch.zeros(1,3,448,448) for _ in range(8)]; lf[2] = tc[0]; lf[3] = tc[1]
        lv = self.w.encode_images(torch.cat(gis+lf, 0).to(self.w.device))
        lt = list(lv[8:])

        vB,    _ = self._htag_forward(gt, None, ['P','T'])
        vBL,   _ = self._htag_forward(gt, lt,   ['P','T'])
        vBF1,  _ = self._htag_forward(gt, None, ['P','T','F1'])
        vBLF1, _ = self._htag_forward(gt, lt,   ['P','T','F1'])

        return vB, vBL, vBF1, vBLF1

    def score_video(self, video_name, stride=4, max_windows=0):
        """Score all windows in a video. Returns dict with vB, k1, combined, labels."""
        vpath = find_video_path(video_name)
        gt_full = np.load(os.path.join(os.path.dirname(_PROJ), 'CoReVAD/src/ucf/gt_ucf.npy')).astype(np.int8)
        vr = VideoReader(vpath, ctx=dcpu(0), num_threads=1)
        N = len(vr)
        gt_video = gt_full[:N]
        anom_idx = np.where(gt_video > 0)[0]
        if len(anom_idx) == 0:
            anom_range = (N//2, N//2)
        else:
            anom_range = (anom_idx[0], anom_idx[-1])
        print(f'Video: {video_name}, frames={N}, GT anomaly: {anom_range}')

        data = defaultdict(list)
        t0 = time.time()
        for s in range(0, N-30, stride):
            ts, te = s+8, s+14
            if te >= N: continue
            center = (ts+te)//2
            ia = int(anom_range[0] <= center <= anom_range[1])
            ho = anom_range[0] < te and ts < anom_range[1]
            if ho and not ia: continue

            vB, vBL, vBF1, vBLF1 = self.process_window(s, vpath, N)
            k1 = vBLF1 - vBL - vBF1 + vB
            data['vB'].append(vB); data['k1'].append(k1)
            data['label'].append(ia); data['start'].append(s)

            nN = sum(1 for l in data['label'] if l == 0)
            nA = sum(data['label'])
            if (nN+nA) % 10 == 0:
                print(f'  [{nN}N+{nA}A] {time.time()-t0:.0f}s', flush=True)
            if max_windows > 0 and nN+nA >= max_windows: break

        vB = np.array(data['vB']); k1 = np.array(data['k1'])
        y = np.array(data['label'])

        def zscore(x): return (x - x.mean()) / (x.std() + 1e-8)
        combined = zscore(vB) + 0.5 * zscore(k1)

        results = {}
        for name, vals in [('vB', vB), ('k1', k1), ('combined', combined)]:
            if len(np.unique(y)) > 1:
                results[name] = roc_auc_score(y, vals)
        results['n_windows'] = len(y)
        results['n_anomaly'] = int(y.sum())
        results['time'] = time.time() - t0
        return results


def main():
    parser = argparse.ArgumentParser(description='Kappa-VAD')
    parser.add_argument('--video', default='Abuse028_x264')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--max_windows', type=int, default=0)
    args = parser.parse_args()

    kv = KappaVAD(gpu=args.gpu)
    results = kv.score_video(args.video, stride=args.stride, max_windows=args.max_windows)

    print(f'\n=== Results ({results["n_windows"]} windows, {results["n_anomaly"]} anomaly) ===')
    for k in ['vB', 'k1', 'combined']:
        if k in results:
            print(f'{k:<10} AUC={results[k]:.4f}')
    print(f'Time: {results["time"]:.0f}s')


if __name__ == '__main__':
    main()
