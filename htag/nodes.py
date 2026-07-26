"""
HTAG: Hierarchical Target-Locked Typed Evidence Access Graph.

Evidence node construction controlled by active_modalities set.
Each node: [vision/audio/prior tokens | type_label | anchor_1..anchor_K]

UCF (no audio, no PAVAD prior):
  D2  (global-only):  active_modalities={"G"}          → 4 nodes, 16 anchors
  D3  (global+local): active_modalities={"G","L"}      → 8 nodes, 32 anchors

Rules:
  - "L" in active_modalities → local_tokens must be non-None
  - "A" in active_modalities → audio_texts must be non-None
  - "R" in active_modalities → prior_texts must be non-None
  - NEVER create empty/zero-filled nodes — skip entirely
"""

import torch
import sys, os
from typing import Dict, List, Tuple, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_evidence_nodes(
    internvl_ll,              # InternVLLowLevel instance
    global_tokens: List[torch.Tensor],    # [8] each [256, 4096]
    local_tokens: Optional[List[torch.Tensor]],  # [8] or None
    groups: Dict[str, List[int]],         # {"P": [0,1], "T": [2,3], "F1": [4,5], "F2": [6,7]}
    audio_texts: Optional[Dict[str, Optional[str]]] = None,
    prior_texts: Optional[Dict[str, Optional[str]]] = None,
    num_anchors: int = 4,
    active_modalities: Optional[Set[str]] = None,
    token_mode: Optional[str] = None,  # 'mean'|'wmean'|'topk'|'topk_cov'
) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Build evidence node embedding sequences.

    Args:
        active_modalities: set of modality chars, e.g. {"G"}, {"G","L"}.
        token_mode:
            'mean' (P0)  — double mean-pool: 256→1, 2 frames→1
            'wmean' (P1) — SST-weighted mean: 256→1, 2 frames→1
            'topk' (P2)  — SST Top-16 per frame, kept separate
            'topk_cov' (P3) — SST Top-8 + uniform 8 per frame, kept separate

    Returns:
        node_embeds_list, node_meta_list
        Each meta dict includes node_name (e.g. "P_G") for manifest tracking.
    """
    temporal_groups = ['P', 'T', 'F1', 'F2']

    if active_modalities is None:
        active_modalities = {"G"}

    # ── Validate modality requirements ──
    if "L" in active_modalities:
        if local_tokens is None:
            raise ValueError("'L' in active_modalities but local_tokens is None")
        if len(local_tokens) != 8:
            raise ValueError(f"local_tokens must have length 8, got {len(local_tokens)}")

    if "A" in active_modalities:
        if audio_texts is None:
            raise ValueError("'A' in active_modalities but audio_texts is None")

    if "R" in active_modalities:
        if prior_texts is None:
            raise ValueError("'R' in active_modalities but prior_texts is None")

    # ── Determine token mode ──
    if token_mode is None:
        token_mode = 'topk_cov'  # default: P3

    # ── Build nodes ──
    node_embeds_list = []
    node_meta_list = []

    for g in temporal_groups:
        frame_indices = groups[g]  # [0,1] or [2,3] etc.

        for m in sorted(active_modalities):  # deterministic order: G, L, A, R
            vision_tokens, audio_text, prior_text = None, None, None
            early_tokens, late_tokens = None, None

            if m == 'G':
                f1_tok = global_tokens[frame_indices[0]]  # [256, D]
                f2_tok = global_tokens[frame_indices[1]]  # [256, D]

                if token_mode == 'mean':
                    # P0: double mean-pool (original broken behavior)
                    vision_tokens = 0.5 * (f1_tok.mean(dim=0) + f2_tok.mean(dim=0))

                elif token_mode == 'wmean':
                    # P1: SST-weighted single vector
                    from htag.token_select import compute_simple_sst_weights, sst_weighted_mean
                    w_sst = compute_simple_sst_weights(f1_tok, f2_tok)
                    vision_tokens = sst_weighted_mean(f1_tok, f2_tok, w_sst)

                elif token_mode == 'topk':
                    # P2: Top-16 SST per frame (kept separate)
                    from htag.token_select import compute_simple_sst_weights, select_global_tokens
                    w_sst = compute_simple_sst_weights(f1_tok, f2_tok)
                    early_tokens, late_tokens, _ = select_global_tokens(
                        f1_tok, f2_tok, w_sst, k_salient=16, k_coverage=0,
                    )

                elif token_mode == 'topk_cov':
                    # P3: SST Top-8 + uniform 8 (kept separate)
                    from htag.token_select import compute_simple_sst_weights, select_global_tokens
                    w_sst = compute_simple_sst_weights(f1_tok, f2_tok)
                    early_tokens, late_tokens, _ = select_global_tokens(
                        f1_tok, f2_tok, w_sst, k_salient=8, k_coverage=8,
                    )

            elif m == 'L':
                f1_tok = local_tokens[frame_indices[0]]
                f2_tok = local_tokens[frame_indices[1]]

                if token_mode in ('mean', 'wmean'):
                    # Old mode: double mean-pool
                    vision_tokens = 0.5 * (f1_tok.mean(dim=0) + f2_tok.mean(dim=0))
                else:
                    # P2/P3: 4×4 spatial pool (kept separate)
                    from htag.token_select import select_local_tokens
                    early_tokens, late_tokens = select_local_tokens(f1_tok, f2_tok)

            elif m == 'A':
                audio_text = audio_texts.get(g) if audio_texts else None
                if audio_text is None:
                    continue

            elif m == 'R':
                prior_text = prior_texts.get(g) if prior_texts else None
                if prior_text is None:
                    continue

            type_label = f"{g} {m}"
            node_name = f"{g}_{m}"

            # Build node: use v2 if early/late separate, v1 if single vision token
            if early_tokens is not None and late_tokens is not None:
                node_embeds, anchor_pos = internvl_ll.build_node_sequence_v2(
                    early_tokens=early_tokens,
                    late_tokens=late_tokens,
                    type_label=type_label,
                    num_anchors=num_anchors,
                    audio_text=audio_text,
                    prior_text=prior_text,
                )
            else:
                node_embeds, anchor_pos = internvl_ll.build_node_sequence(
                    vision_tokens=vision_tokens,
                    type_label=type_label,
                    num_anchors=num_anchors,
                    audio_text=audio_text,
                    prior_text=prior_text,
                )

            node_embeds_list.append(node_embeds)
            node_meta_list.append({
                'node_name': node_name,
                'temporal': g,
                'modality': m,
                'anchor_positions': anchor_pos,
                'num_anchors': num_anchors,
                'has_content': True,
                'seq_len': node_embeds.shape[0],
            })

    # ── Post-validate ──
    node_names = [m['node_name'] for m in node_meta_list]
    total_anchors = sum(m['num_anchors'] for m in node_meta_list)

    if "L" not in active_modalities:
        expected_names = [f"{g}_G" for g in temporal_groups]
        assert node_names == expected_names, \
            f"Expected {expected_names}, got {node_names}"
        assert total_anchors == 16, \
            f"Expected 16 anchors for G-only, got {total_anchors}"

    if active_modalities == {"G", "L"}:
        expected_names = [f"{g}_{m}" for g in temporal_groups for m in ['G', 'L']]
        assert node_names == expected_names, \
            f"Expected {expected_names}, got {node_names}"
        assert total_anchors == 32, \
            f"Expected 32 anchors for G+L, got {total_anchors}"

    return node_embeds_list, node_meta_list


def make_type_label_map() -> Dict[str, str]:
    """Human-readable type labels for debugging."""
    return {'G': 'GLOBAL', 'L': 'LOCAL', 'A': 'AUDIO', 'R': 'PRIOR'}


# ══════════════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=== HTAG nodes test ===")
    import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from internvl_ll import InternVLLowLevel
    wrapper = InternVLLowLevel()
    dummy_global = [torch.randn(256, 4096) for _ in range(8)]
    dummy_local = [torch.randn(256, 4096) for _ in range(8)]
    groups = {"P": [0, 1], "T": [2, 3], "F1": [4, 5], "F2": [6, 7]}

    # D2: G-only
    embeds, metas = build_evidence_nodes(wrapper, dummy_global, None, groups, active_modalities={"G"})
    print(f"D2 (G-only): {len(embeds)} nodes")
    for m in metas: print(f"  {m['node_name']}")
    assert len(embeds) == 4
    assert [m['node_name'] for m in metas] == ['P_G', 'T_G', 'F1_G', 'F2_G']

    # D3: G+L
    embeds, metas = build_evidence_nodes(wrapper, dummy_global, dummy_local, groups, active_modalities={"G","L"})
    print(f"D3 (G+L): {len(embeds)} nodes")
    for m in metas: print(f"  {m['node_name']}")
    assert len(embeds) == 8

    # G+L+A+R (full)
    audio = {"T": "[SOUND] screaming"}
    prior = {"T": "SRP=0.82"}
    embeds, metas = build_evidence_nodes(wrapper, dummy_global, dummy_local, groups,
                                          audio_texts=audio, prior_texts=prior,
                                          active_modalities={"G","L","A","R"})
    print(f"Full (G+L+A+R): {len(embeds)} nodes")
    for m in metas: print(f"  {m['node_name']}")

    # Error cases
    try:
        build_evidence_nodes(wrapper, dummy_global, None, groups, active_modalities={"G","L"})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Error (L without local_tokens): {e}")

    try:
        build_evidence_nodes(wrapper, dummy_global, dummy_local, groups, active_modalities={"G","A"})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Error (A without audio_texts): {e}")

    print("=== OK! ===")
