#!/usr/bin/env python3
"""
PAVAD + ALL Kappa-VAD signals: vB, dL, Δ¹, κ¹ — full multi-channel integration.
"""
import torch, numpy as np, sys, os, time
from sklearn.metrics import roc_auc_score

_PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.join(os.path.dirname(_PROJ), 'lace_vad_github_inspired'))
sys.path.insert(0, '/sdb/data_public/llms/llm/InternVL2-8B')
sys.path.insert(0, os.path.join(os.path.dirname(_PROJ), 'CoReVAD', 'utils'))

from PIL import Image; import torchvision.transforms as T
from lace_vad.sota_bridge import compute_sota_channels, compute_sota_stats
from lace_vad.fusion import postproc
from data_pipeline import load_video_segment, find_video_path, preprocess_for_vit
from pipeline import SAGEPipeline, SAGEConfig
from scpage.phase_encode import PhaseEncoder
from scpage.instability import compute_phase_instability
from scpage.crop_utils import search_max_energy_box, compute_padding_mask, patch_box_to_pixel, expand_and_square_box
from scpage import _resize_for_internvl
from htag.nodes import build_evidence_nodes
from htag.graph import build_compact_graph, build_stage_mask, forward_one_stage
from decord import VideoReader, cpu as dcpu

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

def compute_all_signals(wf):
    """Return vB, dL, dF1, k1 — all Kappa-VAD signals from one window."""
    sp = [wf[i] for i in [0,4,8,12,17,21,25,29]]
    st = [tt(Image.fromarray(f).convert('RGB')) for f in sp]
    gis = [_resize_for_internvl(t, 448) for t in st]
    gv = w.encode_images(torch.cat(gis, 0).to(dev)); gt = list(gv[:8])
    box = sc_box(wf[11])
    tf, tt2 = [sp[2], sp[3]], [st[2], st[3]]
    tc = [T.functional.crop(torch.from_numpy(f).permute(2,0,1).float()/255.0, box[1], box[0], box[3], box[2]) for f in tf]
    tc = [_resize_for_internvl(c, 448) for c in tc]
    lf = [torch.zeros(1,3,448,448) for _ in range(8)]; lf[2] = tc[0]; lf[3] = tc[1]
    lv = w.encode_images(torch.cat(gis+lf, 0).to(dev)); lt = list(lv[8:])
    def v(gt_tok, lt_tok, vis):
        if lt_tok is not None:
            nodes, metas = build_evidence_nodes(w, gt_tok, lt_tok, grp, active_modalities={'G','L'}, token_mode='wmean')
        else:
            nodes, metas = build_evidence_nodes(w, gt_tok, None, grp, active_modalities={'G'}, token_mode='wmean')
        graph, seg = build_compact_graph(w, nodes, metas)
        mask = build_stage_mask(seg, vis, dev, dt)
        z, _, _, _ = forward_one_stage(w, graph, mask, seg['decision_pos'], seg['position_ids'])
        return z
    vB    = v(gt, None, ['P','T'])
    vBL   = v(gt, lt,   ['P','T'])
    vBF1  = v(gt, None, ['P','T','F1'])
    vBLF1 = v(gt, lt,   ['P','T','F1'])
    return vB, vBL-vB, vBF1-vB, vBLF1-vBL-vBF1+vB

# PAVAD
cfg_sota = {'FL': 128, 'SEV_T': 0.12, 'TAU': 0.02, 'GATE': 0.30, 'a': 4.0, 'p': 1.2, 'b': 8.0, 'q': 1.5, 'fg': 1.0, 'bp': 3.0, 'bpcg': 3.0, 'bvos': 0.3, 'clip_t': 0.8, 'clip_l': 0.7, 'clip_p': 0.8, 'clip_pcg': 0.8, 'clip_v': 1.0, 'g_size': 3, 'g_sigma': 2.0, 'pos': 0.5, 'ctr': 0.5, 'rgp_ref': 'intra'}
cache = np.load('/tmp/ucf_cache.npy', allow_pickle=True).item()
IS_DIR = os.path.join(os.path.dirname(_PROJ), 'CoReProbe', 'internal_feats')
IS = {}
for i in range(3):
    p = os.path.join(IS_DIR, f'internal_signals_split{i}.npy')
    if os.path.exists(p):
        for k, v in np.load(p, allow_pickle=True).item().items(): IS[k.replace('#', '')] = v
TEST_LIST = '/sdb/data_public/llms/videos/UCFcrime/Anomaly_Detection_splits/Anomaly_Test.txt'
with open(TEST_LIST) as f:
    order_all = [l.strip().replace('.mp4','').split('/')[-1] for l in f if l.strip()]
order_all = [v for v in order_all if v in cache]
st = compute_sota_stats(cache, IS, order_all, cfg_sota)
gt_full = np.load(os.path.join(os.path.dirname(_PROJ), 'CoReVAD/src/ucf/gt_ucf.npy')).astype(np.int8)
gt_offsets = {}; off = 0
for vname in order_all:
    if vname in cache: gt_offsets[vname] = off; off += cache[vname]['L']

categories = ['Abuse','Fighting','Assault','Robbery','Burglary','Shooting','Arson',
              'RoadAccidents','Stealing','Shoplifting','Vandalism','Explosion']
selected = []
for cat in categories:
    matches = [v for v in order_all if v in cache and cat in v]
    selected.extend(matches[:2])
normals = [v for v in order_all if v in cache and 'Normal' in v]
selected.extend(normals[:2])
print(f'Selected {len(selected)} videos')

STRIDE = 16; MAX_WIN = 15
all_pavad = []; all_vb = []; all_dL = []; all_dF = []; all_k1 = []; all_labels = []
t0 = time.time()
results = []

for vname in selected:
    if vname not in cache: continue
    cv = cache[vname]; M, L, FL = cv['M'], cv['L'], cfg_sota['FL']
    is_segs = IS.get(vname)
    if is_segs is None: continue
    sem_M, rep_M, _ = compute_sota_channels(cv, is_segs, st, cfg_sota)
    sem_f = postproc(sem_M, L, FL, cfg_sota['g_size'], cfg_sota['g_sigma'], cfg_sota['pos'], cfg_sota['ctr'])
    vpath = find_video_path(vname)
    if vpath is None: continue
    try: vr = VideoReader(vpath, ctx=dcpu(0), num_threads=1); nf = len(vr)
    except: continue
    gt_off = gt_offsets.get(vname, 0)

    pw = []; vbw = []; dLw = []; dFw = []; k1w = []; labs = []
    for s in range(0, nf-30, STRIDE):
        ts, te = s+8, s+14
        if te >= nf: continue
        center = (ts+te)//2
        ia = int(gt_full[min(gt_off+center, len(gt_full)-1)] > 0)
        try:
            wf = load_video_segment(vpath, s, 30)
            vb, dl, df, k1 = compute_all_signals(wf)
            pw.append(sem_f[min(s+15, L-1)]); vbw.append(vb)
            dLw.append(dl); dFw.append(df); k1w.append(k1); labs.append(ia)
        except: pass
        if len(labs) >= MAX_WIN: break
    if len(labs) < 3: continue
    ya = np.array(labs)
    if ya.sum() == 0 or ya.sum() == len(ya): continue

    pa = np.array(pw)
    vb = np.array(vbw); dl = np.array(dLw); df = np.array(dFw); k1 = np.array(k1w)

    # Normalize all signals
    def mn(x): return (x-x.min())/(x.ptp()+1e-9)
    pa_n = mn(pa); vb_n = mn(vb); dl_n = mn(np.abs(dl))  # magnitude of local contribution
    df_n = mn(np.abs(df))  # magnitude of future influence
    k1_n = mn(np.maximum(k1, 0))  # positive interaction only

    # Multi-channel fusion: PAVAD + Kappa signals as corrections
    # Use PAVAD's gate for EACH signal independently
    channels = [pa_n]
    for sig, name in [(vb_n, 'vB'), (dl_n, 'dL'), (df_n, 'dF'), (k1_n, 'k1')]:
        tau = np.quantile(sig, 0.80)
        gate = np.maximum(0, sig - tau)
        if gate.max() > 1e-9: gate = gate / gate.max()
        channels.append(0.10 * gate * (1.0 - pa_n))

    fused = sum(channels) / len(channels)
    auc_p = roc_auc_score(ya, pa_n)
    auc_f = roc_auc_score(ya, fused)
    print(f'{vname:<35} {len(ya):>3}w PAVAD={auc_p:.3f} Fused={auc_f:.3f} Δ={auc_f-auc_p:+.3f}', flush=True)
    results.append((vname, auc_p, auc_f, len(ya)))

    all_pavad.extend(pa_n); all_vb.extend(vb_n); all_dL.extend(dl_n)
    all_dF.extend(df_n); all_k1.extend(k1_n); all_labels.extend(ya)

# Overall
if len(all_labels) > 1:
    pa = np.array(all_pavad); vb = np.array(all_vb); dl = np.array(all_dL)
    df = np.array(all_dF); k1 = np.array(all_k1); ya = np.array(all_labels)

    tau_vb = np.quantile(vb, 0.80); g_vb = np.maximum(0, vb-tau_vb)
    if g_vb.max()>1e-9: g_vb /= g_vb.max()
    tau_dl = np.quantile(dl, 0.80); g_dl = np.maximum(0, dl-tau_dl)
    if g_dl.max()>1e-9: g_dl /= g_dl.max()
    tau_df = np.quantile(df, 0.80); g_df = np.maximum(0, df-tau_df)
    if g_df.max()>1e-9: g_df /= g_df.max()
    tau_k1 = np.quantile(k1, 0.80); g_k1 = np.maximum(0, k1-tau_k1)
    if g_k1.max()>1e-9: g_k1 /= g_k1.max()

    fused_a = (pa + 0.10*g_vb*(1-pa) + 0.10*g_dl*(1-pa) + 0.10*g_df*(1-pa) + 0.10*g_k1*(1-pa)) / 5

    print(f'\n=== OVERALL ({len(ya)} windows, {len(results)} videos) ===')
    print(f'PAVAD:           AUC={roc_auc_score(ya, pa):.4f}')
    print(f'PAVAD + 4 Kappa: AUC={roc_auc_score(ya, fused_a):.4f}')
    print(f'Gate improves: {sum(1 for _,p,f,_ in results if f>p)}/{len(results)}')
print(f'Time: {time.time()-t0:.0f}s')
