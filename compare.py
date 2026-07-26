#!/usr/bin/env python3
"""Compare PAVAD baseline vs Kappa-VAD on the SAME windows."""
import torch, numpy as np, sys, os, time
from sklearn.metrics import roc_auc_score

_PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.join(os.path.dirname(_PROJ), 'lace_vad_github_inspired'))
sys.path.insert(0, '/sdb/data_public/llms/llm/InternVL2-8B')
sys.path.insert(0, os.path.join(os.path.dirname(_PROJ), 'CoReVAD', 'utils'))

from lace_vad.sota_bridge import compute_sota_channels, compute_sota_stats
from lace_vad.fusion import postproc
from decord import VideoReader, cpu as dcpu

# ---- 1. PAVAD baseline on Abuse028 ----
print("=== PAVAD Baseline ===")
cfg_sota = {
    'FL': 128, 'SEV_T': 0.12, 'TAU': 0.02, 'GATE': 0.30,
    'a': 4.0, 'p': 1.2, 'b': 8.0, 'q': 1.5, 'fg': 1.0,
    'bp': 3.0, 'bpcg': 3.0, 'bvos': 0.3,
    'clip_t': 0.8, 'clip_l': 0.7, 'clip_p': 0.8, 'clip_pcg': 0.8, 'clip_v': 1.0,
    'g_size': 3, 'g_sigma': 2.0, 'pos': 0.5, 'ctr': 0.5,
    'rgp_ref': 'intra',
}

cache = np.load('/tmp/ucf_cache.npy', allow_pickle=True).item()
IS_DIR = os.path.join(os.path.dirname(_PROJ), 'CoReProbe', 'internal_feats')

# Load IS for Abuse028
IS = {}
for i in range(3):
    p = os.path.join(IS_DIR, f'internal_signals_split{i}.npy')
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True).item()
        for k, v in d.items():
            IS[k.replace('#', '')] = v

# Compute PAVAD stats on all test videos
TEST_LIST = '/sdb/data_public/llms/videos/UCFcrime/Anomaly_Detection_splits/Anomaly_Test.txt'
with open(TEST_LIST) as f:
    order_all = [l.strip().replace('.mp4', '').split('/')[-1] for l in f if l.strip()]
order_all = [v for v in order_all if v in cache]
st = compute_sota_stats(cache, IS, order_all, cfg_sota)

# PAVAD on Abuse028
vname = 'Abuse028_x264'
cv = cache[vname]; M, L, FL = cv['M'], cv['L'], cfg_sota['FL']
is_segs = IS.get(vname)
sem_M, rep_M, _ = compute_sota_channels(cv, is_segs, st, cfg_sota)

# Post-process to frame level
sem_f = postproc(sem_M, L, FL, cfg_sota['g_size'], cfg_sota['g_sigma'], cfg_sota['pos'], cfg_sota['ctr'])
# Normalize
sem_f = (sem_f - sem_f.min()) / (sem_f.ptp() + 1e-9)

# Load GT
gt_full = np.load(os.path.join(os.path.dirname(_PROJ), 'CoReVAD/src/ucf/gt_ucf.npy')).astype(np.int8)
gt_video = gt_full[:L]

# ---- 2. Kappa-VAD results (load from saved file) ----
print("\n=== Kappa-VAD ===")
data = np.load('/tmp/kappa_Abuse028_x264.npz')
vB = data['vB']; k1 = data['k1']; y_kappa = data['label']; starts = data['start']

def zscore(x): return (x - x.mean()) / (x.std() + 1e-8)
combined_kappa = zscore(vB) + 0.5 * zscore(k1)

# ---- 3. Compare on same windows: window-level aggregation ----
print("\n=== Comparison (same windows) ===")
# For each Kappa window [s, s+30], take PAVAD score mean over that interval
pavad_window = []
for s in starts:
    win_scores = sem_f[s:s+30] if s+30 <= len(sem_f) else sem_f[s:]
    pavad_window.append(win_scores.mean())
pavad_window = np.array(pavad_window)
# Normalize within these windows (fair comparison)
pavad_window = (pavad_window - pavad_window.min()) / (pavad_window.ptp() + 1e-9)

print(f"Windows: {len(y_kappa)} ({int(y_kappa.sum())} anomaly)")
print(f"{'Method':<25} {'AUC':>8}")
print(f"{'PAVAD (window-level)':<25} {roc_auc_score(y_kappa, pavad_window):8.4f}")
print(f"{'Kappa-VAD vB (G-only)':<25} {roc_auc_score(y_kappa, vB):8.4f}")
print(f"{'Kappa-VAD k1 (interact)':<25} {roc_auc_score(y_kappa, k1):8.4f}")
print(f"{'Kappa-VAD combined':<25} {roc_auc_score(y_kappa, combined_kappa):8.4f}")
