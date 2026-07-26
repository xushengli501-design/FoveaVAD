"""
Single-pass HTAG: all tokens from embedding layer through all 32 layers.

Sequence: [STATIC_RULES | NODE_0 | NODE_1 | ... | C_T | Answer:]

Mask rules:
  - STATIC_RULES: causal self-attention
  - Each NODE: block-diagonal — can only see itself (no cross-node, no static access)
    Within node: type_label → vision_tokens → anchors (causal)
  - C_T: sees visible anchors (controlled by TPVA stage)
  - Answer: sees C_T + itself (causal) — NOT raw anchors, NOT static

Three TPVA stages differ ONLY in which anchors C_T can see.
"""

import torch
import torch.nn.functional as F
import sys, os
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════════
# Build compact graph
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_compact_graph(
    internvl_ll,
    node_embeds_list: List[torch.Tensor],
    node_meta_list: List[Dict],
    canonical_positions: Optional[Dict] = None,  # pre-computed G-only positions for position reuse
) -> Tuple[torch.Tensor, Dict]:
    """
    Build compact graph sequence (no mask yet — mask varies per TPVA stage).

    If canonical_positions is provided, L/A/R nodes in the same temporal block
    reuse the G node's position range, keeping C_T and Answer positions fixed.
    This ensures adding optional modalities does not shift RoPE positions.

    Returns:
        graph_embeds: [1, L, 4096]
        seg: dict with segment boundaries and position_ids
    """
    device = internvl_ll.device
    dtype = internvl_ll.dtype
    tokenizer = internvl_ll.tokenizer
    emb = internvl_ll.embed_tokens

    # ── Static rules ──
    static_text = (
        "Judge only the TARGET block. "
        "PAST, FUTURE-1 and FUTURE-2 are context and must not be scored. "
        "Determine whether TARGET contains any anomalous or policy-violating content. "
        "Different violation types need not be mutually exclusive. "
        "Answer only Yes or No."
    )
    static_ids = tokenizer.encode(static_text, add_special_tokens=False)
    static_embeds = emb(torch.tensor(static_ids, device=device))
    if static_embeds.dim() == 2:
        static_embeds = static_embeds

    pieces = [static_embeds.to(dtype=dtype)]
    static_end = len(static_ids)
    offset = static_end

    # ── Evidence nodes ──
    node_boundaries = []
    for node_embeds, meta in zip(node_embeds_list, node_meta_list):
        L_n = node_embeds.shape[0]
        pieces.append(node_embeds.to(device=device, dtype=dtype))

        anchor_pos = meta['anchor_positions']
        node_boundaries.append({
            'start': offset,
            'end': offset + L_n,
            'vision_end': offset + anchor_pos[0],  # everything before first anchor = vision+label
            'anchor_start': offset + anchor_pos[0],
            'anchor_end': offset + anchor_pos[-1] + 1,
            'temporal': meta['temporal'],
            'modality': meta['modality'],
        })
        offset += L_n

    # ── C_T ──
    ct_embed = internvl_ll.get_token_embedding(internvl_ll.anchor_token_id)
    pieces.append(ct_embed.unsqueeze(0).to(device=device, dtype=dtype))
    ct_pos = offset
    offset += 1

    # ── Answer suffix ──
    answer_text = "\nAnswer:"
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    answer_embeds = emb(torch.tensor(answer_ids, device=device))
    if answer_embeds.dim() == 2:
        answer_embeds = answer_embeds
    pieces.append(answer_embeds.to(dtype=dtype))
    answer_start = offset
    answer_end = offset + len(answer_ids)
    decision_pos = answer_end - 1

    graph_embeds = torch.cat(pieces, dim=0).unsqueeze(0)
    L = graph_embeds.shape[1]

    # ── Graph-relative position IDs with temporal reuse ──
    # Strategy: nodes in the same temporal block SHARE position ranges.
    # G and L nodes are isolated by block-diagonal mask, so reusing positions
    # is safe (they cannot attend to each other). C_T and Answer have fixed positions.
    # max(position_ids) < L ensures RoPE cache compatibility.

    pos_ids = torch.zeros(L, dtype=torch.long, device=device)

    # Static rules: positions 0..static_end-1
    pos_ids[:static_end] = torch.arange(static_end, device=device)

    # FIXED positions for C_T and Answer (from G-only canonical layout)
    if canonical_positions is not None:
        ct_position = canonical_positions['ct_position']
        ans_positions = canonical_positions['answer_positions']
        slot_positions = canonical_positions['slots']  # {temporal: {modality: {vis_start, anc_start}}}
    else:
        # Compute canonical positions from this (G-only) layout
        ct_position = None  # will be set below
        ans_positions = None
        slot_positions = {}

    # Assign positions: G nodes get canonical, L nodes reuse G's range
    current_pos = static_end
    temporal_order = ['P', 'T', 'F1', 'F2']

    for g_name in temporal_order:
        # Find G node in this temporal block (reference for positions)
        g_nb = None
        for nb in node_boundaries:
            if nb['temporal'] == g_name and nb['modality'] == 'G':
                g_nb = nb
                break

        if g_nb is not None and canonical_positions is None:
            # Record canonical positions for this G node
            n_vis = g_nb['vision_end'] - g_nb['start']
            n_anc = g_nb['anchor_end'] - g_nb['anchor_start']
            slot_positions[g_name] = {
                'vis_start': current_pos,
                'anc_start': current_pos + n_vis,
                'anc_end': current_pos + n_vis + n_anc,
            }
            pos_ids[g_nb['start']:g_nb['vision_end']] = current_pos + torch.arange(n_vis, device=device)
            pos_ids[g_nb['anchor_start']:g_nb['anchor_end']] = current_pos + n_vis + torch.arange(n_anc, device=device)
            current_pos += n_vis + n_anc

        elif g_nb is not None and canonical_positions is not None:
            # Use canonical positions for G node
            ref = canonical_positions['slots'][g_name]
            n_vis = g_nb['vision_end'] - g_nb['start']
            n_anc = g_nb['anchor_end'] - g_nb['anchor_start']
            # Map vis tokens: distribute across canonical vis range
            vis_ref_start = ref['vis_start']
            vis_ref_len = ref['anc_start'] - ref['vis_start']
            anc_ref_start = ref['anc_start']
            anc_ref_len = ref['anc_end'] - ref['anc_start']

            # Vision tokens: map linearly to canonical vis positions
            vis_indices = vis_ref_start + torch.linspace(0, vis_ref_len - 1, n_vis, device=device).round().long()
            pos_ids[g_nb['start']:g_nb['vision_end']] = vis_indices
            # Anchors: map to canonical anchor positions (most important for C_T access)
            anc_indices = anc_ref_start + torch.linspace(0, anc_ref_len - 1, n_anc, device=device).round().long()
            pos_ids[g_nb['anchor_start']:g_nb['anchor_end']] = anc_indices

        # Handle non-G nodes (L, A, R): reuse G's position range
        for nb in node_boundaries:
            if nb['temporal'] != g_name or nb['modality'] == 'G':
                continue
            ref = canonical_positions['slots'][g_name] if canonical_positions is not None else slot_positions[g_name]
            n_vis = nb['vision_end'] - nb['start']
            n_anc = nb['anchor_end'] - nb['anchor_start']
            vis_ref_start = ref['vis_start']
            vis_ref_len = ref['anc_start'] - ref['vis_start']
            anc_ref_start = ref['anc_start']
            anc_ref_len = ref['anc_end'] - ref['anc_start']

            vis_indices = vis_ref_start + torch.linspace(0, vis_ref_len - 1, n_vis, device=device).round().long()
            pos_ids[nb['start']:nb['vision_end']] = vis_indices
            anc_indices = anc_ref_start + torch.linspace(0, anc_ref_len - 1, n_anc, device=device).round().long()
            pos_ids[nb['anchor_start']:nb['anchor_end']] = anc_indices

    if canonical_positions is not None:
        ct_position = canonical_positions['ct_position']
        ans_positions = canonical_positions['answer_positions']
    else:
        ct_position = current_pos
        n_ans = answer_end - answer_start
        ans_positions = current_pos + 1 + torch.arange(n_ans, device=device)

    pos_ids[ct_pos] = ct_position
    n_ans = answer_end - answer_start
    pos_ids[answer_start:answer_end] = ans_positions[:n_ans]

    seg = {
        'static_end': static_end,
        'nodes': node_boundaries,
        'ct_pos': ct_pos,
        'answer_start': answer_start,
        'answer_end': answer_end,
        'decision_pos': decision_pos,
        'L': L,
        'position_ids': pos_ids,
        'ct_position': ct_position,
    }
    return graph_embeds, seg


# ══════════════════════════════════════════════════════════════════════
# Build 4D mask for a given TPVA stage
# ══════════════════════════════════════════════════════════════════════

def build_stage_mask(
    seg: Dict,
    visible_temporal: List[str],
    device: torch.device,
    dtype: torch.dtype,
    inactive_modalities: set = None,
) -> torch.Tensor:
    """
    Build 4D attention mask for one TPVA stage.

    Rules:
      - STATIC: causal self-attention only
      - Each NODE: block-diagonal — no cross-node, no static access
        vision tokens see earlier vision+label; anchors see all vision+label+earlier anchors
      - C_T: sees visible anchors EXCEPT those from inactive_modalities
      - Answer: sees C_T + itself (causal) — NO static, NO anchors

    inactive_modalities: set of modality chars whose anchors C_T should NOT see
                         (e.g., {'L'} to mask out local nodes)
    """
    L = seg['L']
    mask = torch.full((1, 1, L, L), float('-inf'), device=device, dtype=dtype)
    s_end = seg['static_end']
    a_start = seg['answer_start']
    dec = seg['decision_pos']
    ct = seg['ct_pos']

    # ── Static: self-causal ──
    for q in range(s_end):
        for k in range(q + 1):
            mask[0, 0, q, k] = 0.0

    # ── Nodes: block-diagonal, no static access ──
    for nb in seg['nodes']:
        n_start, v_end = nb['start'], nb['vision_end']
        a_s, a_e = nb['anchor_start'], nb['anchor_end']

        # Vision tokens (including type label): within-node causal only
        for q in range(n_start, v_end):
            for k in range(n_start, q + 1):
                mask[0, 0, q, k] = 0.0

        # Anchor tokens: see all vision+label in this node + earlier anchors
        for q in range(a_s, a_e):
            for k in range(n_start, v_end):
                mask[0, 0, q, k] = 0.0
            for k in range(a_s, q + 1):
                mask[0, 0, q, k] = 0.0

    # ── C_T: sees visible anchors (skip inactive modalities) ──
    if inactive_modalities is None:
        inactive_modalities = set()
    for nb in seg['nodes']:
        if nb['temporal'] in visible_temporal and nb['modality'] not in inactive_modalities:
            for k in range(nb['anchor_start'], nb['anchor_end']):
                mask[0, 0, ct, k] = 0.0
    mask[0, 0, ct, ct] = 0.0

    # ── Answer suffix: sees C_T + visible anchors + itself (causal) ──
    # FullAccess mode: Answer directly reads relevant anchors, not just via C_T
    for q in range(a_start, seg['answer_end']):
        mask[0, 0, q, ct] = 0.0
        # Answer also directly reads anchors from visible temporal groups
        for nb in seg['nodes']:
            if nb['temporal'] in visible_temporal and nb['modality'] not in inactive_modalities:
                for k in range(nb['anchor_start'], nb['anchor_end']):
                    mask[0, 0, q, k] = 0.0
        for k in range(a_start, q + 1):
            mask[0, 0, q, k] = 0.0

    return mask


# ══════════════════════════════════════════════════════════════════════
# Single-pass forward
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def forward_one_stage(
    internvl_ll,
    graph_embeds: torch.Tensor,
    mask_4d: torch.Tensor,
    decision_pos: int,
    position_ids: torch.Tensor = None,
) -> Tuple[float, float, float, float]:
    """Forward through ALL 32 layers → Yes/No at decision position."""
    out = internvl_ll.stage2_graph_forward(
        graph_embeds, mask_4d,
        decision_positions=[decision_pos],
        output_hidden_states=True,
        position_ids=position_ids,
    )
    yes, no = internvl_ll.yes_no_logits(out['logits'])
    z = internvl_ll.log_odds(yes, no).item()
    p = internvl_ll.anomaly_probability(torch.tensor(z)).item()
    return z, p, yes.item(), no.item()


@torch.no_grad()
def forward_batched(
    internvl_ll,
    graph_embeds: torch.Tensor,       # [1, L, D] — single graph
    masks: List[torch.Tensor],        # list of [1, 1, L, L] masks
    decision_pos: int,
    position_ids: torch.Tensor = None,  # [1, L]
) -> List[float]:
    """Forward same graph with multiple masks in one batch → list of z values.

    All masks must have the same shape (same graph, different visibility rules).
    Vision encoding is shared; only the attention mask differs.
    """
    B = len(masks)
    L = graph_embeds.shape[1]
    device = graph_embeds.device
    dtype = graph_embeds.dtype

    # Stack masks: [B, 1, L, L]
    batch_mask = torch.cat([m.to(device) for m in masks], dim=0)

    # Expand graph: [B, L, D]
    batch_graph = graph_embeds.expand(B, -1, -1).to(device=device, dtype=dtype)

    # Position IDs: [B, L]
    if position_ids is not None:
        batch_pos = position_ids.expand(B, -1).to(device)
    else:
        batch_pos = None

    # Decision positions: one per batch item
    batch_dec = [decision_pos] * B

    out = internvl_ll.stage2_graph_forward(
        batch_graph, batch_mask,
        decision_positions=batch_dec,
        output_hidden_states=True,
        position_ids=batch_pos,
    )
    yes, no = internvl_ll.yes_no_logits(out['logits'])  # [B]
    z_vals = (yes - no).cpu().tolist()  # list of B floats
    return z_vals


# ══════════════════════════════════════════════════════════════════════
# Grouped perturbation test
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def perturbed_forward(
    internvl_ll,
    graph_embeds: torch.Tensor,
    seg: Dict,
    perturb_group: str,  # 'P','T','F1','F2' or None
    visible_temporal: List[str],
) -> float:
    """
    Forward with one temporal group's vision tokens randomized.
    Returns z value.
    """
    device = internvl_ll.device
    dtype = internvl_ll.dtype
    perturbed = graph_embeds.clone()

    if perturb_group is not None:
        for nb in seg['nodes']:
            if nb['temporal'] == perturb_group:
                # Randomize all vision+label tokens in this node
                n_s, v_e = nb['start'], nb['vision_end']
                n_tokens = v_e - n_s
                perturbed[0, n_s:v_e, :] = torch.randn(
                    n_tokens, graph_embeds.shape[-1], device=device, dtype=dtype,
                )

    mask = build_stage_mask(seg, visible_temporal, device, dtype)
    z, _, _, _ = forward_one_stage(internvl_ll, perturbed, mask, seg['decision_pos'], seg['position_ids'])
    return z


# ══════════════════════════════════════════════════════════════════════
# Full HTAG + TPVA forward
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def htag_tpva_forward(
    internvl_ll,
    global_tokens: List[torch.Tensor],
    local_tokens: Optional[List[torch.Tensor]],
    groups: Dict[str, List[int]],
    audio_texts=None, prior_texts=None,
    active_modalities=None, token_mode=None,
) -> Dict:
    """
    Single-pass HTAG + TPVA. Returns dict with z0,z1,z2 etc.
    """
    from htag.nodes import build_evidence_nodes

    if active_modalities is None:
        active_modalities = {"G", "L"} if local_tokens is not None else {"G"}

    # ── Position invariance: compute canonical positions from G-only layout ──
    canonical = None
    if 'L' in active_modalities or 'A' in active_modalities or 'R' in active_modalities:
        nodes_g, metas_g = build_evidence_nodes(
            internvl_ll, global_tokens, None, groups,
            active_modalities={'G'}, token_mode=token_mode,
        )
        _, seg_g = build_compact_graph(internvl_ll, nodes_g, metas_g)
        # Extract canonical positions
        canonical = {
            'ct_position': seg_g['position_ids'][seg_g['ct_pos']].item(),
            'answer_positions': seg_g['position_ids'][seg_g['answer_start']:seg_g['answer_end']].clone(),
            'slots': {},
        }
        for g_name in ['P', 'T', 'F1', 'F2']:
            for nb in seg_g['nodes']:
                if nb['temporal'] == g_name and nb['modality'] == 'G':
                    g_vis_start = seg_g['position_ids'][nb['start']].item()
                    g_anc_start = seg_g['position_ids'][nb['anchor_start']].item()
                    g_anc_end = seg_g['position_ids'][nb['anchor_end'] - 1].item() + 1
                    canonical['slots'][g_name] = {
                        'vis_start': g_vis_start,
                        'anc_start': g_anc_start,
                        'anc_end': g_anc_end,
                    }
                    break

    node_embeds_list, node_meta_list = build_evidence_nodes(
        internvl_ll, global_tokens, local_tokens, groups,
        audio_texts=audio_texts, prior_texts=prior_texts,
        active_modalities=active_modalities, token_mode=token_mode,
    )
    node_names = [m['node_name'] for m in node_meta_list]
    total_anchors = sum(m['num_anchors'] for m in node_meta_list)

    graph_embeds, seg = build_compact_graph(
        internvl_ll, node_embeds_list, node_meta_list,
        canonical_positions=canonical,
    )
    device = internvl_ll.device
    dtype = internvl_ll.dtype
    dec = seg['decision_pos']

    stage_configs = [
        ('M0', ['P', 'T']),
        ('M1', ['P', 'T', 'F1']),
        ('M2', ['P', 'T', 'F1', 'F2']),
    ]

    z_vals, p_vals, yes_vals, no_vals = [], [], [], []
    results = {}

    for stage_name, vis in stage_configs:
        mask = build_stage_mask(seg, vis, device, dtype)
        z, p, yes_l, no_l = forward_one_stage(internvl_ll, graph_embeds, mask, dec, seg['position_ids'])
        z_vals.append(z); p_vals.append(p)
        yes_vals.append(yes_l); no_vals.append(no_l)
        results[stage_name] = {'z': z, 'p': p}

    z0, z1, z2 = z_vals
    delta1 = z1 - z0
    delta2 = z2 - z1
    t = _classify(z0, z1, z2)

    return {
        'z0': z0, 'z1': z1, 'z2': z2,
        'p0': p_vals[0], 'p1': p_vals[1], 'p2': p_vals[2],
        'delta1': delta1, 'delta2': delta2,
        'trajectory': t[0], 'trajectory_compact': t[1],
        'final_score': p_vals[2],
        'stage_details': results,
        'yes_logits': yes_vals, 'no_logits': no_vals,
        'node_manifest': node_names,
        'num_nodes': len(node_names), 'num_anchors': total_anchors,
    }


def _classify(z0, z1, z2):
    s = ','.join('+' if x > 0 else '-' for x in [z0, z1, z2])
    m = {'+,+,+': 'stable_anomaly', '-,+,+': 'delayed', '-,-,+': 'delayed',
         '+,-,-': 'corrected', '+,+,-': 'corrected',
         '+,-,+': 'oscillation', '-,+,-': 'oscillation', '-,-,-': 'stable_normal'}
    return s, m.get(s, 'unknown')


# ══════════════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=== Single-pass HTAG test ===")
    import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from sage_pavad.internvl_ll import InternVLLowLevel
    from sage_pavad.htag.nodes import build_evidence_nodes

    wrapper = InternVLLowLevel()
    device = wrapper.device
    dtype = wrapper.dtype
    torch.manual_seed(42)
    dummy_global = [torch.randn(256, 4096) for _ in range(8)]
    groups = {"P": [0, 1], "T": [2, 3], "F1": [4, 5], "F2": [6, 7]}

    for mode in ['mean', 'topk']:
        nodes, metas = build_evidence_nodes(wrapper, dummy_global, None, groups, active_modalities={"G"}, token_mode=mode)
        graph, seg = build_compact_graph(wrapper, nodes, metas)

        # Verify block-diagonal: nodes cannot see each other
        node_starts = [n['start'] for n in seg['nodes']]
        for m0_idx in range(3):  # M0: only P, T visible
            mask = build_stage_mask(seg, ['P', 'T'], device, dtype)
            # Check: node 0 (P) cannot see node 1 (T) vision range
            for nb_i, nb in enumerate(seg['nodes']):
                for nb_j, nb2 in enumerate(seg['nodes']):
                    if nb_i != nb_j:
                        # nb_i anchor cannot see nb_j vision
                        for q in range(nb['anchor_start'], nb['anchor_end']):
                            for k in range(nb2['start'], nb2['vision_end']):
                                assert mask[0,0,q,k].item() < -1e4, \
                                    f"Node {nb_i} anchor sees node {nb_j} vision!"

        z0, _, _, _ = forward_one_stage(wrapper, graph,
            build_stage_mask(seg, ['P', 'T'], device, dtype), seg['decision_pos'])
        z2, _, _, _ = forward_one_stage(wrapper, graph,
            build_stage_mask(seg, ['P', 'T', 'F1', 'F2'], device, dtype), seg['decision_pos'])
        print(f"  {mode}: L={seg['L']} dec={seg['decision_pos']} z0={z0:.4f} z2={z2:.4f} ✓ block-diag pass")

    # Grouped perturbation test
    print("\n=== Grouped perturbation test ===")
    nodes, metas = build_evidence_nodes(wrapper, dummy_global, None, groups, active_modalities={"G"}, token_mode='mean')
    graph, seg = build_compact_graph(wrapper, nodes, metas)

    base = {}
    for vis in [['P','T'], ['P','T','F1'], ['P','T','F1','F2']]:
        stage = f'M{len(vis)-2}' if len(vis) <= 2 else f'M{len(vis)-2}'
        if stage == 'M0': stage = 'M0'
        elif stage == 'M1': stage = 'M1'
        else: stage = 'M2'
        if len(vis) == 2: stage = 'M0'
        elif len(vis) == 3: stage = 'M1'
        else: stage = 'M2'
        mask = build_stage_mask(seg, vis, device, dtype)
        base[stage], _, _, _ = forward_one_stage(wrapper, graph, mask, seg['decision_pos'])

    tol = 1e-4
    results = {}
    for g in ['P', 'T', 'F1', 'F2']:
        results[g] = {}
        for vis, stage in [(['P','T'],'M0'), (['P','T','F1'],'M1'), (['P','T','F1','F2'],'M2')]:
            mask = build_stage_mask(seg, vis, device, dtype)
            zp = perturbed_forward(wrapper, graph, seg, g, vis)
            delta = abs(zp - base[stage])
            results[g][stage] = delta
            print(f"  {g} @ {stage}: base={base[stage]:.4f} pert={zp:.4f} δ={delta:.2e}")

    # Verify expected visibility pattern
    checks = [
        ('F1', 'M0', False),  # F1 should NOT affect M0
        ('F2', 'M0', False),  # F2 should NOT affect M0
        ('F2', 'M1', False),  # F2 should NOT affect M1
        ('T', 'M0', True),   # T SHOULD affect M0
        ('F1', 'M1', True),  # F1 SHOULD affect M1
        ('F2', 'M2', True),  # F2 SHOULD affect M2
    ]
    for g, stage, should_respond in checks:
        d = results[g][stage]
        if should_respond:
            ok = d > tol
        else:
            ok = d < tol
        status = '✓' if ok else '✗ FAIL'
        print(f"  {g}@{stage} {'should' if should_respond else 'should NOT'} respond: δ={d:.2e} {status}")
    print("=== Done ===")
