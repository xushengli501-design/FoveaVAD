#!/usr/bin/env python3
"""
Three-condition validation experiments for FoveaVAD.

Usage: python run_experiments.py --gpu 0 --stride 30 --max_per_video 5
"""
import torch, numpy as np, sys, os, time, json, argparse, random
from collections import defaultdict
from sklearn.metrics import roc_auc_score

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, '/sdb/data_public/llms/llm/InternVL2-8B')
sys.path.insert(0, os.path.join(os.path.dirname(_PROJ), 'CoReVAD', 'utils'))

from PIL import Image
import torchvision.transforms as T
from data_pipeline import load_video_segment, find_video_path, preprocess_for_vit
from pipeline import SAGEPipeline, SAGEConfig
from scpage.phase_encode import PhaseEncoder
from scpage.instability import compute_phase_instability
from scpage.crop_utils import (
    search_max_energy_box, compute_padding_mask,
    patch_box_to_pixel, expand_and_square_box
)
from scpage import _resize_for_internvl
from htag.nodes import build_evidence_nodes
from htag.graph import (build_compact_graph, build_stage_mask,
                         forward_one_stage, forward_batched)
from decord import VideoReader, cpu as dcpu

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_RESULTS_DIR, exist_ok=True)

SMALL_TARGET_CATS = {'Shoplifting', 'Stealing', 'Burglary'}
LARGE_TARGET_CATS = {'Explosion', 'Arson', 'Fighting', 'RoadAccidents',
                     'Shooting', 'Abuse', 'Assault', 'Robbery', 'Vandalism', 'Arrest'}

def get_category(vname):
    for cat in (SMALL_TARGET_CATS | LARGE_TARGET_CATS):
        if vname.startswith(cat): return cat
    return 'Normal' if 'Normal' in vname else 'Unknown'


# ═══ Box helpers ═══
def _patch_to_pixel(br, BOX, ow, oh):
    xm,ym,xM,yM = patch_box_to_pixel(br, box_size=BOX, phase_offset=(0,0))
    xe,ye,xMe,yMe = expand_and_square_box(xm,ym,xM,yM, expand_ratio=1.10,
                                          image_w=224, image_h=224)
    s=min(224/ow,224/oh); nw,nh=int(ow*s),int(oh*s)
    px,py=(224-nw)//2,(224-nh)//2
    return (max(0,min(int((xe-px)/s),ow-1)),
            max(0,min(int((ye-py)/s),oh-1)),
            max(1,min(int((xMe-px)/s),ow))-max(0,min(int((xe-px)/s),ow-1)),
            max(1,min(int((yMe-py)/s),oh))-max(0,min(int((ye-py)/s),oh-1)))



class ExperimentRunner:
    def __init__(self, gpu=0):
        self.device = torch.device(f'cuda:{gpu}')
        cfg = SAGEConfig(); pipe = SAGEPipeline(cfg); pipe._ensure_models()
        self.w = pipe.internvl; self.tt = T.ToTensor()
        self.enc = PhaseEncoder(device='cuda'); self.BOX = 5
        self.grp = {'P':[0,1],'T':[2,3],'F1':[4,5],'F2':[6,7]}

        self.gt_full = np.load(
            os.path.join(os.path.dirname(_PROJ),'CoReVAD/src/ucf/gt_ucf.npy')
        ).astype(np.int8)
        self.cache = np.load('/tmp/ucf_cache.npy', allow_pickle=True).item()
        TEST_LIST = '/sdb/data_public/llms/videos/UCFcrime/Anomaly_Detection_splits/Anomaly_Test.txt'
        with open(TEST_LIST) as f:
            self.order_all = [l.strip().replace('.mp4','').split('/')[-1]
                              for l in f if l.strip()]
        self.order_all = [v for v in self.order_all if v in self.cache]
        self.gt_offsets = {}; off = 0
        for vname in self.order_all:
            if vname in self.cache: self.gt_offsets[vname] = off; off += self.cache[vname]['L']

    def select_videos(self, n_per_cat=2, n_normal=2):
        cats = defaultdict(list)
        for vname in self.order_all:
            if vname not in self.cache: continue
            c = get_category(vname)
            if c in ('Normal','Unknown'): continue
            if len(cats[c]) < n_per_cat: cats[c].append(vname)
        sel = [v for cv in cats.values() for v in cv]
        nml = [v for v in self.order_all if 'Normal' in v and v in self.cache]
        sel.extend(nml[:n_normal])
        return sel

    # ── Box methods ──
    def _box_phase(self, frame):
        ow,oh = Image.fromarray(frame).size
        vi=preprocess_for_vit(frame)
        e=self.enc.encode_phases(vi)
        H,_,jv=compute_phase_instability(e['features'],e['valid_mask'])
        pm=compute_padding_mask(ow,oh).to(self.device); sm=jv&pm
        br,_=search_max_energy_box(H,box_size=self.BOX,valid_mask=sm)
        if br is None: return (0,0,ow,oh)
        return _patch_to_pixel(br,self.BOX,ow,oh)

    def _box_center(self, frame):
        ow,oh=Image.fromarray(frame).size; half=self.BOX//2
        pr,pc=7-half,7-half
        xm=pc*16+4;ym=pr*16+4;xM=xm+self.BOX*16;yM=ym+self.BOX*16
        xe,ye,xMe,yMe=expand_and_square_box(xm,ym,xM,yM,expand_ratio=1.10,image_w=224,image_h=224)
        s=min(224/ow,224/oh);nw,nh=int(ow*s),int(oh*s);px,py=(224-nw)//2,(224-nh)//2
        return (max(0,min(int((xe-px)/s),ow-1)),max(0,min(int((ye-py)/s),oh-1)),
                max(1,min(int((xMe-px)/s),ow))-max(0,min(int((xe-px)/s),ow-1)),
                max(1,min(int((yMe-py)/s),oh))-max(0,min(int((ye-py)/s),oh-1)))

    def _box_random(self, frame, seed=0):
        ow,oh=Image.fromarray(frame).size; ms=14-self.BOX
        if ms<2: return (0,0,ow,oh)
        random.seed(hash(frame.tobytes()[:100])+seed)
        pr,pc=random.randint(1,ms-1),random.randint(1,ms-1)
        xm=pc*16+4;ym=pr*16+4;xM=xm+self.BOX*16;yM=ym+self.BOX*16
        xe,ye,xMe,yMe=expand_and_square_box(xm,ym,xM,yM,expand_ratio=1.10,image_w=224,image_h=224)
        s=min(224/ow,224/oh);nw,nh=int(ow*s),int(oh*s);px,py=(224-nw)//2,(224-nh)//2
        return (max(0,min(int((xe-px)/s),ow-1)),max(0,min(int((ye-py)/s),oh-1)),
                max(1,min(int((xMe-px)/s),ow))-max(0,min(int((xe-px)/s),ow-1)),
                max(1,min(int((yMe-py)/s),oh))-max(0,min(int((ye-py)/s),oh-1)))

    def _box_full(self, frame):
        ow,oh=Image.fromarray(frame).size; return (0,0,ow,oh)

    def _box_tok_norm(self, frame):
        ow,oh=Image.fromarray(frame).size
        vi=preprocess_for_vit(frame); e=self.enc.encode_phases(vi)
        H=e['features'][0].norm(dim=-1).float()
        pm=compute_padding_mask(ow,oh).to(self.device); jv=e['valid_mask'][0]&pm
        br,_=search_max_energy_box(H,box_size=self.BOX,valid_mask=jv)
        if br is None: return (0,0,ow,oh)
        return _patch_to_pixel(br,self.BOX,ow,oh)

    # ── Encoding (optimized) ──
    def encode_global(self, wf):
        sp=[wf[i] for i in [0,4,8,12,17,21,25,29]]
        st=[self.tt(Image.fromarray(f).convert('RGB')) for f in sp]
        gis=[_resize_for_internvl(t,448) for t in st]
        return list(self.w.encode_images(torch.cat(gis,0).to(self.device)))

    def encode_local_for_box(self, wf, box):
        """Encode ONLY the 2 local crop images (efficient)."""
        sp=[wf[i] for i in [0,4,8,12,17,21,25,29]]
        tf=[sp[2],sp[3]]
        bx,by,bw,bh=box
        tc=[]
        for fr in tf:
            c=T.functional.crop(torch.from_numpy(fr).permute(2,0,1).float()/255.0,by,bx,bh,bw)
            tc.append(_resize_for_internvl(c,448))
        lv=self.w.encode_images(torch.cat(tc,0).to(self.device))
        # Build full 8-frame list: only T positions filled
        lt=[torch.zeros(256,4096, device=self.device, dtype=self.w.dtype) for _ in range(8)]
        lt[2]=lv[0]; lt[3]=lv[1]
        return lt

    def forward_4_batched(self, gt, lt):
        w=self.w; grp=self.grp; dev=self.device; dt=w.dtype
        nodes,metas=build_evidence_nodes(w,gt,lt,grp,active_modalities={'G','L'},token_mode='wmean')
        graph,seg=build_compact_graph(w,nodes,metas)
        dec=seg['decision_pos']; pos_ids=seg['position_ids']
        masks=[
            build_stage_mask(seg,['P','T'],dev,dt,inactive_modalities={'L'}),
            build_stage_mask(seg,['P','T'],dev,dt),
            build_stage_mask(seg,['P','T','F1'],dev,dt,inactive_modalities={'L'}),
            build_stage_mask(seg,['P','T','F1'],dev,dt),
        ]
        z=forward_batched(w,graph,masks,dec,pos_ids)
        return {'vB':z[0],'vBL':z[1],'vBF1':z[2],'vBLF1':z[3],
                'kappa':z[3]-z[1]-z[2]+z[0]}

    # ═══════════════════════════════════════════
    # EXP 1: Box methods
    # ═══════════════════════════════════════════
    def run_exp1(self, videos, stride, max_per):
        print("\n"+"="*60)
        print("EXP 1: Box Methods (phase/center/random/full/tok_norm)")
        print("="*60)
        methods=['phase','center','random','full','tok_norm']
        box_fns={'phase':self._box_phase,'center':self._box_center,
                 'random':self._box_random,'full':self._box_full,'tok_norm':self._box_tok_norm}
        data={m:{'vB':[],'kappa':[],'labels':[]} for m in methods}
        t0=time.time()

        for vname in videos:
            if vname not in self.cache: continue
            vpath=find_video_path(vname)
            if vpath is None: continue
            try: vr=VideoReader(vpath,ctx=dcpu(0),num_threads=1); nf=len(vr)
            except: continue
            gt_off=self.gt_offsets.get(vname,0); n_win=0

            for s in range(0,nf-30,stride):
                ts,te=s+8,s+14
                if te>=nf: continue
                ia=int(self.gt_full[min(gt_off+(ts+te)//2,len(self.gt_full)-1)]>0)
                try:
                    wf=load_video_segment(vpath,s,30)
                    gt=self.encode_global(wf)  # shared
                    for m in methods:
                        box=box_fns[m](wf[11])
                        lt=self.encode_local_for_box(wf,box)
                        r=self.forward_4_batched(gt,lt)
                        data[m]['vB'].append(r['vB'])
                        data[m]['kappa'].append(r['kappa'])
                        data[m]['labels'].append(ia)
                    n_win+=1
                except Exception as e: pass
                if n_win>=max_per: break
            print(f"  [{vname}] {n_win}w ({time.time()-t0:.0f}s)",flush=True)

        print("\n--- Results ---")
        res={}
        for m in methods:
            r=data[m]
            if len(r['labels'])<2 or len(np.unique(r['labels']))<2: continue
            def zs(x): return (x-x.mean())/(x.std()+1e-8)
            vb=np.array(r['vB']); k=np.array(r['kappa']); y=np.array(r['labels'])
            comb=zs(vb)+0.5*zs(k)
            try: auc_vb=roc_auc_score(y,vb); auc_c=roc_auc_score(y,comb)
            except: continue
            res[m]={'auc_vB':float(auc_vb),'auc_combined':float(auc_c),
                    'delta':float(auc_c-auc_vb),'n':int(len(y))}
            print(f"  {m:<10} vB={auc_vb:.4f} Comb={auc_c:.4f} Δ={auc_c-auc_vb:+.4f}")

        print("\n--- Ranking ---")
        for i,(m,r) in enumerate(sorted(res.items(),key=lambda x:x[1]['auc_combined'],reverse=True)):
            print(f"  {i+1}. {m:<10} AUC={r['auc_combined']:.4f}")
        return res

    # ═══════════════════════════════════════════
    # EXP 2: Small vs Large targets
    # ═══════════════════════════════════════════
    def run_exp2(self, videos, stride, max_per):
        print("\n"+"="*60)
        print("EXP 2: Target Size Analysis")
        print("="*60)
        cat_data=defaultdict(lambda:{'vB':[],'kappa':[],'labels':[]})
        t0=time.time()

        for vname in videos:
            if vname not in self.cache: continue
            cat=get_category(vname)
            if cat in ('Normal','Unknown'): continue
            vpath=find_video_path(vname)
            if vpath is None: continue
            try: vr=VideoReader(vpath,ctx=dcpu(0),num_threads=1); nf=len(vr)
            except: continue
            gt_off=self.gt_offsets.get(vname,0); n_win=0

            for s in range(0,nf-30,stride):
                ts,te=s+8,s+14
                if te>=nf: continue
                ia=int(self.gt_full[min(gt_off+(ts+te)//2,len(self.gt_full)-1)]>0)
                try:
                    wf=load_video_segment(vpath,s,30)
                    gt=self.encode_global(wf)
                    box=self._box_phase(wf[11])
                    lt=self.encode_local_for_box(wf,box)
                    r=self.forward_4_batched(gt,lt)
                    cat_data[cat]['vB'].append(r['vB'])
                    cat_data[cat]['kappa'].append(r['kappa'])
                    cat_data[cat]['labels'].append(ia)
                    n_win+=1
                except Exception as e:
                    if n_win == 0:
                        import traceback
                        print(f"  ERROR in {vname}: {e}", flush=True)
                        traceback.print_exc()
                if n_win>=max_per: break

        print("\n--- Per-Category ---")
        res={'categories':{}}
        small_vb,small_k,small_y=[],[],[]
        large_vb,large_k,large_y=[],[],[]

        for cat in sorted(cat_data.keys()):
            r=cat_data[cat]
            if len(r['labels'])<2 or len(np.unique(r['labels']))<2: continue
            def zs(x): return (x-x.mean())/(x.std()+1e-8)
            vb=np.array(r['vB']); k=np.array(r['kappa']); y=np.array(r['labels'])
            comb=zs(vb)+0.5*zs(k)
            try: auc_vb=roc_auc_score(y,vb); auc_c=roc_auc_score(y,comb)
            except: continue
            sz='small' if cat in SMALL_TARGET_CATS else 'large'
            tag='🔍' if sz=='small' else '📐'
            res['categories'][cat]={'size':sz,'auc_vB':float(auc_vb),
                'auc_combined':float(auc_c),'delta':float(auc_c-auc_vb),'n':int(len(y))}
            print(f"  {tag} {cat:<18} vB={auc_vb:.4f} Comb={auc_c:.4f} Δ={auc_c-auc_vb:+.4f}")
            if sz=='small': small_vb.extend(r['vB']);small_k.extend(r['kappa']);small_y.extend(r['labels'])
            else: large_vb.extend(r['vB']);large_k.extend(r['kappa']);large_y.extend(r['labels'])

        for label,vb_a,k_a,y_a in [('small',small_vb,small_k,small_y),('large',large_vb,large_k,large_y)]:
            if len(set(y_a))<2: continue
            def zs(x): return (x-x.mean())/(x.std()+1e-8)
            vb=np.array(vb_a); comb=zs(vb)+0.5*zs(np.array(k_a)); y=np.array(y_a)
            auc_vb=roc_auc_score(y,vb); auc_c=roc_auc_score(y,comb)
            res[label]= {'auc_vB':float(auc_vb),'auc_combined':float(auc_c),
                         'delta':float(auc_c-auc_vb),'n':int(len(y))}
            print(f"  {label}: vB={auc_vb:.4f} Comb={auc_c:.4f} Δ={auc_c-auc_vb:+.4f}")

        if 'small' in res and 'large' in res:
            sd,ld=res['small']['delta'],res['large']['delta']
            print(f"\n  *** Small Δ={sd:+.4f} vs Large Δ={ld:+.4f} ***")
            if sd>ld+0.01: print("  >>> Gains CONCENTRATE on small targets ✓")
            else: print("  >>> No clear concentration ✗")
        return res

    # ═══════════════════════════════════════════
    # EXP 3: Component ablation
    # ═══════════════════════════════════════════
    def run_exp3(self, videos, stride, max_per):
        print("\n"+"="*60)
        print("EXP 3: Component Ablation")
        print("="*60)
        data=defaultdict(list); t0=time.time()

        for vname in videos:
            if vname not in self.cache: continue
            vpath=find_video_path(vname)
            if vpath is None: continue
            try: vr=VideoReader(vpath,ctx=dcpu(0),num_threads=1); nf=len(vr)
            except: continue
            gt_off=self.gt_offsets.get(vname,0); n_win=0

            for s in range(0,nf-30,stride):
                ts,te=s+8,s+14
                if te>=nf: continue
                ia=int(self.gt_full[min(gt_off+(ts+te)//2,len(self.gt_full)-1)]>0)
                try:
                    wf=load_video_segment(vpath,s,30)
                    gt=self.encode_global(wf)
                    box=self._box_phase(wf[11])
                    lt=self.encode_local_for_box(wf,box)
                    w=self.w; grp=self.grp; dev=self.device; dt=w.dtype

                    # (a) Global only (G-only graph, M0)
                    nodes_g,metas_g=build_evidence_nodes(w,gt,None,grp,active_modalities={'G'},token_mode='wmean')
                    graph_g,seg_g=build_compact_graph(w,nodes_g,metas_g)
                    mask_g=build_stage_mask(seg_g,['P','T'],dev,dt)
                    zg,_,_,_=forward_one_stage(w,graph_g,mask_g,seg_g['decision_pos'],seg_g['position_ids'])
                    data['a_global_only'].append(float(zg))

                    # (b) Naive concat
                    zc=self._forward_concat(w,gt,lt,grp)
                    data['b_concat'].append(float(zc))

                    # (c) Full kappa
                    r=self.forward_4_batched(gt,lt)
                    data['c_vB_HTAG'].append(float(r['vB']))
                    data['c_vBL_HTAG'].append(float(r['vBL']))
                    data['c_kappa'].append(float(r['kappa']))
                    data['labels'].append(ia)
                    n_win+=1
                except Exception as e:
                    if n_win == 0:
                        import traceback
                        print(f"  ERROR in {vname}: {e}", flush=True)
                        traceback.print_exc()
                if n_win>=max_per: break
            print(f"  [{vname}] {n_win}w ({time.time()-t0:.0f}s)",flush=True)

        if len(data['labels'])<2: print("Not enough data!"); return {}
        y=np.array(data['labels'])
        def zs(x): return (x-x.mean())/(x.std()+1e-8)

        a=np.array(data['a_global_only']); b=np.array(data['b_concat'])
        vb=np.array(data['c_vB_HTAG']); vbl=np.array(data['c_vBL_HTAG'])
        k=np.array(data['c_kappa']); comb=zs(vb)+0.5*zs(k)

        auc_a=roc_auc_score(y,a); auc_b=roc_auc_score(y,b)
        auc_vb=roc_auc_score(y,vb); auc_vbl=roc_auc_score(y,vbl)
        auc_comb=roc_auc_score(y,comb); auc_k=roc_auc_score(y,k)

        print("\n--- Results ---")
        print(f"  (a) Global only (G, M0):            AUC={auc_a:.4f}")
        print(f"  (b) G+L concat (no HTAG):           AUC={auc_b:.4f}  Δ={auc_b-auc_a:+.4f}")
        print(f"  (c) HTAG vB (L hidden):             AUC={auc_vb:.4f}")
        print(f"  (d) HTAG vBL (L visible, M0):       AUC={auc_vbl:.4f}  Δ={auc_vbl-auc_vb:+.4f}")
        print(f"  (e) Full: z(vB)+0.5*z(κ):           AUC={auc_comb:.4f}  Δ={auc_comb-auc_vb:+.4f}")
        print(f"  (f) Kappa only:                      AUC={auc_k:.4f}")

        print(f"\n--- Key ---")
        print(f"  HTAG vs Concat: {auc_vbl-auc_b:+.4f}")
        print(f"  Kappa vs HTAG M0: {auc_comb-auc_vbl:+.4f}")
        print(f"  Total gain: {auc_comb-auc_a:+.4f}")

        res={'a_global':float(auc_a),'b_concat':float(auc_b),
             'c_HTAG_vB':float(auc_vb),'c_HTAG_vBL':float(auc_vbl),
             'd_full_kappa':float(auc_comb),'d_kappa_only':float(auc_k),
             'htag_vs_concat':float(auc_vbl-auc_b),
             'kappa_vs_HTAG_M0':float(auc_comb-auc_vbl),
             'total':float(auc_comb-auc_a)}
        if auc_vbl>auc_b+0.005: print("  >>> HTAG > concat ✓")
        if auc_comb>auc_vbl+0.005: print("  >>> Kappa adds value ✓")
        return res

    def _forward_concat(self, w, gt, lt, grp):
        dev=w.device; dt=w.dtype; tok=w.tokenizer
        from htag.token_select import compute_simple_sst_weights, sst_weighted_mean
        txt="Judge only the TARGET block. PAST, FUTURE-1 and FUTURE-2 are context. Answer only Yes or No."
        pieces=[w.embed_tokens(torch.tensor(tok.encode(txt,add_special_tokens=False),device=dev)).to(dtype=dt)]
        fi_map={'P':[0,1],'T':[2,3],'F1':[4,5],'F2':[6,7]}
        for g in ['P','T','F1','F2']:
            fi=fi_map[g]
            ws=compute_simple_sst_weights(gt[fi[0]],gt[fi[1]])
            pieces.append(sst_weighted_mean(gt[fi[0]],gt[fi[1]],ws).unsqueeze(0).to(dev=dev,dtype=dt))
            if lt is not None:
                ws_l=compute_simple_sst_weights(lt[fi[0]],lt[fi[1]])
                pieces.append(sst_weighted_mean(lt[fi[0]],lt[fi[1]],ws_l).unsqueeze(0).to(dev=dev,dtype=dt))
        q="\nIs the TARGET block anomalous?"
        pieces.append(w.embed_tokens(torch.tensor(tok.encode(q,add_special_tokens=False),device=dev)).to(dtype=dt))
        a="\nAnswer:"
        pieces.append(w.embed_tokens(torch.tensor(tok.encode(a,add_special_tokens=False),device=dev)).to(dtype=dt))
        graph=torch.cat(pieces,0).unsqueeze(0); L=graph.shape[1]; dec=L-1
        mask=torch.zeros(1,1,L,L,device=dev,dtype=dt)
        for qq in range(L):
            for kk in range(qq+1): mask[0,0,qq,kk]=0.0
        z,_,_,_=forward_one_stage(w,graph,mask,dec)
        return float(z)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--stride',type=int,default=30)
    parser.add_argument('--max_per_video',type=int,default=5)
    parser.add_argument('--experiments',type=str,default='1,2,3')
    parser.add_argument('--videos_per_cat',type=int,default=2)
    parser.add_argument('--all_videos',action='store_true')
    args=parser.parse_args()

    runner=ExperimentRunner(gpu=args.gpu)
    videos=runner.order_all if args.all_videos else runner.select_videos(args.videos_per_cat)

    print(f"Videos: {len(videos)}, stride={args.stride}, max_per_video={args.max_per_video}")
    for v in videos: print(f"  {v}")

    all_results={'config':{'stride':args.stride,'max_per_video':args.max_per_video,
                            'n_videos':len(videos),'videos':videos}}
    exps=[int(x.strip()) for x in args.experiments.split(',')]

    if 1 in exps: all_results['exp1_box_methods']=runner.run_exp1(videos,args.stride,args.max_per_video)
    if 2 in exps: all_results['exp2_target_size']=runner.run_exp2(videos,args.stride,args.max_per_video)
    if 3 in exps: all_results['exp3_ablation']=runner.run_exp3(videos,args.stride,args.max_per_video)

    ts=time.strftime('%Y%m%d_%H%M%S')
    out=os.path.join(_RESULTS_DIR,f'results_{ts}.json')
    with open(out,'w') as f: json.dump(all_results,f,indent=2,default=str)
    print(f"\nSaved: {out}")

if __name__=='__main__':
    main()
