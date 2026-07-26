"""
InternVL2-8B low-level access wrapper for SAGE-PAVAD.

Provides direct control over vision encoding, vision projection, and
LLM forward passes — bypassing the high-level model.chat() API.

Architecture summary (verified 2026-07-20):
  - LLM: InternLM2ForCausalLM, 32 transformer layers
  - ViT: InternVisionEncoder, 24 layers, 448×448 input, 14×14 patch = 1024 patches
  - Vision projection (mlp1): LayerNorm→Linear(4096,4096)→GELU→Linear(4096,4096)
  - pixel_shuffle downsamples 32×32→16×16 = 256 image tokens per frame
  - IMG_CONTEXT token id = 92546
  - No FlashAttention2 installed → eager attention, 4D mask compatible
  - language_model.forward() accepts inputs_embeds ✓
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class InternVLLowLevel:
    """Low-level InternVL2-8B access for HTAG + TPVA."""

    def __init__(
        self,
        model_path: str = '/sdb/data_public/llms/llm/InternVL2-8B',
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype or (torch.bfloat16 if self.device.type == 'cuda' else torch.float32)

        print(f'[InternVLLowLevel] Loading InternVL2-8B on {self.device} ({self.dtype})...', flush=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval().to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False,
        )

        # ── Architecture references ──
        self.vision_model = self.model.vision_model
        self.mlp1 = self.model.mlp1  # vision projection
        self.language_model = self.model.language_model
        self.lm_head = self.model.lm_head  # 92553 × 4096
        self.llm_layers = self.language_model.model.layers  # ModuleList, 32 layers
        self.embed_tokens = self.language_model.model.tok_embeddings  # token embeddings

        self.num_llm_layers = len(self.llm_layers)
        self.num_image_token = self.model.num_image_token  # 256
        self.hidden_dim = 4096

        # ── Key token IDs ──
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.img_start_token = '<img>'
        self.img_end_token = '</img>'
        self.img_context_token = '<IMG_CONTEXT>'

        # For HTAG anchors: use single-token words
        self.yes_token_id = self.tokenizer.encode(' Yes', add_special_tokens=False)[0]  # 7560
        self.no_token_id = self.tokenizer.encode(' No', add_special_tokens=False)[0]    # 2458
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')           # 92542
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids('<|im_start|>')         # 92543

        # Single-token anchor candidate: " mark" (id=2017, single token, semantically neutral)
        # NOT IMG_CONTEXT (92546) — that is reserved for vision feature replacement
        self.anchor_token_id = self.tokenizer.encode(' mark', add_special_tokens=False)[0]

        # ── HTAG split point ──
        # Stage 1: LLM layers 0..stage1_end (inclusive), outputs hidden states for extraction
        # Stage 2: LLM layers stage1_start..31, receives compact evidence graph
        self.stage1_end = 23        # last layer of stage 1 (0-indexed)
        self.stage2_start = 24      # first layer of stage 2 (0-indexed)
        self.num_stage1_layers = self.stage1_end + 1   # 24
        self.num_stage2_layers = self.num_llm_layers - self.stage2_start  # 8

        print(f'[InternVLLowLevel] LLM layers={self.num_llm_layers}, '
              f'Stage1=0..{self.stage1_end} ({self.num_stage1_layers}L), '
              f'Stage2={self.stage2_start}..{self.num_llm_layers-1} ({self.num_stage2_layers}L)',
              flush=True)

    # ══════════════════════════════════════════════════════════════════════
    # Vision encoding: frames → vision tokens
    # ══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def encode_images(
        self,
        pixel_values: torch.Tensor,  # [B, 3, 448, 448]
        output_vision_layer: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode images → projected vision tokens [B, 256, 4096].

        Args:
            pixel_values: preprocessed images at 448×448
            output_vision_layer: if set, extract patch tokens from specific ViT layer
                                 before pixel_shuffle+mlp1 (for diagnostic purposes)
        Returns:
            vision_tokens: [B, num_image_token, hidden_dim]
        """
        if output_vision_layer is not None:
            # Extract raw patch tokens from specific ViT layer
            vit_out = self.vision_model(
                pixel_values=pixel_values.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[output_vision_layer]
            vit_embeds = vit_out[:, 1:, :]  # remove CLS
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values.to(device=self.device, dtype=self.dtype),
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
            vit_embeds = vit_embeds[:, 1:, :]  # remove CLS, [B, 1024, 4096]

        # pixel_shuffle: 32×32 → 16×16 (downsample_ratio=0.5)
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.model.pixel_shuffle(
            vit_embeds, scale_factor=self.model.downsample_ratio,
        )
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

        # mlp1 projection
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds  # [B, 256, 4096]

    # ══════════════════════════════════════════════════════════════════════
    # Token embedding
    # ══════════════════════════════════════════════════════════════════════

    def embed_tokens_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs → embeddings. [B, L] → [B, L, 4096] or [L] → [L, 4096]."""
        emb = self.embed_tokens(token_ids.to(device=self.device))
        # nn.Embedding returns [N, D] for 1D input — add batch dim for uniformity
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        return emb

    def get_token_embedding(self, token_id: int) -> torch.Tensor:
        """Get single token embedding. [4096]."""
        tid = torch.tensor([token_id], device=self.device)
        emb = self.embed_tokens(tid)
        # nn.Embedding returns [N, D] for 1D input, [B, L, D] for 2D input
        if emb.dim() == 2:
            return emb[0, :]
        return emb[0, 0, :]

    # ══════════════════════════════════════════════════════════════════════
    # Stage 1: evidence-node independent encoding (full LLM forward)
    # ══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def stage1_encode_nodes(
        self,
        inputs_embeds: torch.Tensor,         # [B, L_max, 4096] padded batch of node sequences
        attention_mask: torch.Tensor,         # [B, L_max] 1=valid, 0=pad
        node_anchor_positions: List[List[int]],  # per-sample: list of anchor token positions
        output_layer: int = 23,               # extract hidden states at this layer (0-indexed)
    ) -> List[torch.Tensor]:
        """
        Stage 1: encode each evidence node independently.

        The batch dimension contains different evidence nodes (e.g., 16 nodes).
        Each node's sequence: [V_gm_tokens | type_label_tokens | anchor_tokens].

        Returns hidden states at the anchor positions from the specified layer.

        Args:
            inputs_embeds: [B, L_max, 4096] padded batch
            attention_mask: [B, L_max]
            node_anchor_positions: per-sample list of anchor token positions
            output_layer: which LLM layer's output to extract (default 23 = layer 23)

        Returns:
            list of [K, 4096] tensors, one per batch sample,
            where K = len(anchor_positions) per sample
        """
        B = inputs_embeds.shape[0]
        embeds = inputs_embeds.to(device=self.device, dtype=self.dtype)
        mask = attention_mask.to(device=self.device)

        outputs = self.language_model(
            inputs_embeds=embeds,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        # hidden_states[0] = embeddings, hidden_states[1] = layer 0 output, ...
        # hidden_states[output_layer + 1] = output of target layer
        target_hidden = outputs.hidden_states[output_layer + 1]  # [B, L_max, 4096]

        # Extract anchor positions per sample
        results = []
        for b in range(B):
            anchors = target_hidden[b, node_anchor_positions[b], :]  # [K, 4096]
            results.append(anchors.float())

        return results

    # ══════════════════════════════════════════════════════════════════════
    # Stage 2: compact evidence graph with custom 4D attention mask
    # ══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def stage2_graph_forward(
        self,
        inputs_embeds: torch.Tensor,          # [B, L, 4096] compact graph
        custom_4d_mask: torch.Tensor,          # [B, 1, L, L] 0=allowed, -inf=masked
        output_hidden_states: bool = True,
        decision_positions: Optional[List[int]] = None,
        position_ids: Optional[torch.Tensor] = None,  # [B, L] graph-relative positions
    ) -> Dict:
        """
        Stage 2: forward compact evidence graph through LLM with custom 4D mask.

        Bypasses InternLM2's _prepare_decoder_attention_mask (which expects 2D)
        by monkey-patching it to pass through 4D masks as-is.

        Returns:
            dict with logits, last_hidden, hidden_states at decision positions.
        """
        B, L, D = inputs_embeds.shape
        embeds = inputs_embeds.to(device=self.device, dtype=self.dtype)
        mask_4d = custom_4d_mask.to(device=self.device, dtype=self.dtype)

        # Monkey-patch: allow 4D attention masks through _prepare_decoder_attention_mask
        inner_model = self.language_model.model
        original_prepare = inner_model._prepare_decoder_attention_mask

        def patched_prepare(attention_mask, input_shape, inputs_embeds, past_key_values_length):
            if attention_mask is not None and attention_mask.dim() == 4:
                # Already 4D: create causal mask and combine
                bsz, tgt_len = input_shape
                dt = inputs_embeds.dtype
                dev = inputs_embeds.device
                causal = torch.full((tgt_len, tgt_len), torch.finfo(dt).min, device=dev)
                causal.masked_fill_(
                    torch.arange(tgt_len, device=dev) < (torch.arange(tgt_len, device=dev) + 1).view(tgt_len, 1),
                    0,
                )
                causal = causal.to(dt)[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)
                if past_key_values_length > 0:
                    zeros_pad = torch.zeros(bsz, 1, tgt_len, past_key_values_length, dtype=dt, device=dev)
                    causal = torch.cat([zeros_pad, causal], dim=-1)
                return attention_mask + causal
            return original_prepare(attention_mask, input_shape, inputs_embeds, past_key_values_length)

        inner_model._prepare_decoder_attention_mask = patched_prepare

        try:
            lm_kwargs = dict(
                inputs_embeds=embeds,
                attention_mask=mask_4d,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                use_cache=False,
            )
            if position_ids is not None:
                # Ensure [B, L] shape
                if position_ids.dim() == 1:
                    position_ids = position_ids.unsqueeze(0)
                lm_kwargs['position_ids'] = position_ids.to(device=self.device)
            outputs = self.language_model(**lm_kwargs)
        finally:
            inner_model._prepare_decoder_attention_mask = original_prepare

        result = {}

        if decision_positions is not None:
            all_logits = outputs.logits  # [B, L, vocab_size]
            result['logits'] = torch.stack([
                all_logits[b, decision_positions[b], :] for b in range(B)
            ], dim=0).float()

            last_hidden = outputs.hidden_states[-1]  # [B, L, 4096]
            result['last_hidden'] = torch.stack([
                last_hidden[b, decision_positions[b], :] for b in range(B)
            ], dim=0).float()

        if output_hidden_states:
            result['hidden_states'] = [h.float() for h in outputs.hidden_states]

        return result

    # ══════════════════════════════════════════════════════════════════════
    # Full forward (non-HTAG fallback: all visual + text → logits)
    # ══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def forward_full(
        self,
        pixel_values: torch.Tensor,           # [B_img, 3, 448, 448]
        text_input_ids: torch.Tensor,          # [B_txt, L_txt]
        image_positions_in_text: List[int],    # where to insert image tokens
    ) -> Dict:
        """
        Standard InternVL forward: images + text → logits.

        This is a fallback for non-HTAG comparisons (e.g., 'simple prompt'
        baseline in ablation studies).
        """
        # Encode images
        vit_embeds = self.encode_images(pixel_values)  # [B_img, 256, 4096]

        # Get text embeddings
        text_embeds = self.embed_tokens(text_input_ids.to(self.device))  # [B_txt, L_txt, 4096]

        # Insert vision tokens at image positions
        B = text_embeds.shape[0]
        L = text_embeds.shape[1]
        flat_embeds = text_embeds.reshape(B * L, -1)
        flat_ids = text_input_ids.reshape(B * L).to(self.device)

        # Use IMG_CONTEXT token positions
        img_mask = (flat_ids == self.img_context_token_id)
        n_img = img_mask.sum().item()

        if n_img > 0:
            vit_flat = vit_embeds.reshape(-1, self.hidden_dim)
            if vit_flat.shape[0] >= n_img:
                flat_embeds[img_mask] = vit_flat[:n_img].to(dtype=flat_embeds.dtype)
            else:
                flat_embeds[img_mask[:vit_flat.shape[0]]] = vit_flat.to(dtype=flat_embeds.dtype)

        inputs_embeds_final = flat_embeds.reshape(B, L, -1)

        # Forward through LLM
        outputs = self.language_model(
            inputs_embeds=inputs_embeds_final.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        return {
            'logits': outputs.logits.float(),
            'hidden_states': [h.float() for h in outputs.hidden_states],
        }

    # ══════════════════════════════════════════════════════════════════════
    # Yes/No logit helpers
    # ══════════════════════════════════════════════════════════════════════

    def yes_no_logits(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract Yes/No logits from vocabulary logits.

        Args:
            logits: [..., vocab_size]

        Returns:
            yes_logits: [...] , no_logits: [...]
        """
        yes = logits[..., self.yes_token_id]
        no = logits[..., self.no_token_id]
        return yes, no

    def log_odds(self, yes_logits: torch.Tensor, no_logits: torch.Tensor) -> torch.Tensor:
        """z = logit_Yes - logit_No."""
        return yes_logits - no_logits

    def anomaly_probability(self, z: torch.Tensor) -> torch.Tensor:
        """p = sigmoid(z)."""
        return torch.sigmoid(z)

    # ══════════════════════════════════════════════════════════════════════
    # Utility: build node sequence for one evidence node
    # ══════════════════════════════════════════════════════════════════════

    def build_node_sequence(
        self,
        vision_tokens: torch.Tensor,         # [N_v, 4096] or None
        type_label: str,                      # e.g., "PAST GLOBAL"
        num_anchors: int = 4,
        audio_text: Optional[str] = None,     # for audio nodes
        prior_text: Optional[str] = None,     # for prior nodes
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Build embedding sequence for one evidence node.

        Sequence: [vision_tokens | audio_text | prior_text | type_label | anchor_1..anchor_K]

        Returns:
            node_embeds: [L, 4096]
            anchor_positions: list of anchor token positions in the sequence
        """
        pieces = []

        # 1. Vision tokens (if any)
        if vision_tokens is not None:
            vt = vision_tokens.to(device=self.device, dtype=self.dtype)
            if vt.dim() == 1:
                vt = vt.unsqueeze(0)  # [D] → [1, D]
            pieces.append(vt)

        # 2. Audio text (if any)
        if audio_text is not None:
            audio_ids = self.tokenizer.encode(audio_text, add_special_tokens=False)
            audio_embeds = self.embed_tokens(
                torch.tensor(audio_ids, device=self.device)
            )
            if audio_embeds.dim() == 2:
                audio_embeds = audio_embeds  # [N, D] — keep as-is for cat
            pieces.append(audio_embeds.to(dtype=self.dtype))

        # 3. Prior text (if any)
        if prior_text is not None:
            prior_ids = self.tokenizer.encode(prior_text, add_special_tokens=False)
            prior_embeds = self.embed_tokens(
                torch.tensor(prior_ids, device=self.device)
            )
            pieces.append(prior_embeds.to(dtype=self.dtype))

        # 4. Type label text
        label_ids = self.tokenizer.encode(type_label, add_special_tokens=False)
        label_embeds = self.embed_tokens(
            torch.tensor(label_ids, device=self.device)
        )
        pieces.append(label_embeds.to(dtype=self.dtype))

        # 5. Anchor tokens (K copies of anchor_token_id)
        anchor_ids = torch.full((num_anchors,), self.anchor_token_id, device=self.device)
        anchor_embeds = self.embed_tokens(anchor_ids)
        pieces.append(anchor_embeds.to(dtype=self.dtype))

        # Concatenate
        node_embeds = torch.cat(pieces, dim=0)  # [L, 4096]

        # Anchor positions are the last K positions
        L = node_embeds.shape[0]
        anchor_positions = list(range(L - num_anchors, L))

        return node_embeds, anchor_positions

    def build_node_sequence_v2(
        self,
        early_tokens: torch.Tensor,           # [K, 4096] or None — early frame selected tokens
        late_tokens: torch.Tensor,            # [K, 4096] or None — late frame selected tokens
        type_label: str,                      # e.g., "PAST GLOBAL"
        num_anchors: int = 4,
        audio_text: Optional[str] = None,
        prior_text: Optional[str] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Build node sequence with SEPARATE early/late frame tokens.

        Sequence: [EARLY_FRAME_label | early_tokens | LATE_FRAME_label | late_tokens | type_label | anchors]

        This preserves per-frame spatial structure and temporal order.
        No mean-pool across frames or patches.
        """
        pieces = []

        # 1. Early frame
        if early_tokens is not None and early_tokens.shape[0] > 0:
            early_label_ids = self.tokenizer.encode(
                ' EARLY_FRAME', add_special_tokens=False,
            )
            early_label_emb = self.embed_tokens(
                torch.tensor(early_label_ids, device=self.device)
            )
            if early_label_emb.dim() == 2:
                early_label_emb = early_label_emb  # [N, D]
            pieces.append(early_label_emb.to(dtype=self.dtype))
            pieces.append(early_tokens.to(device=self.device, dtype=self.dtype))

        # 2. Late frame
        if late_tokens is not None and late_tokens.shape[0] > 0:
            late_label_ids = self.tokenizer.encode(
                ' LATE_FRAME', add_special_tokens=False,
            )
            late_label_emb = self.embed_tokens(
                torch.tensor(late_label_ids, device=self.device)
            )
            if late_label_emb.dim() == 2:
                late_label_emb = late_label_emb
            pieces.append(late_label_emb.to(dtype=self.dtype))
            pieces.append(late_tokens.to(device=self.device, dtype=self.dtype))

        # 3. Audio (same as v1)
        if audio_text is not None:
            audio_ids = self.tokenizer.encode(audio_text, add_special_tokens=False)
            audio_embeds = self.embed_tokens(torch.tensor(audio_ids, device=self.device))
            if audio_embeds.dim() == 2:
                audio_embeds = audio_embeds
            pieces.append(audio_embeds.to(dtype=self.dtype))

        # 4. Prior (same as v1)
        if prior_text is not None:
            prior_ids = self.tokenizer.encode(prior_text, add_special_tokens=False)
            prior_embeds = self.embed_tokens(torch.tensor(prior_ids, device=self.device))
            if prior_embeds.dim() == 2:
                prior_embeds = prior_embeds
            pieces.append(prior_embeds.to(dtype=self.dtype))

        # 5. Type label
        label_ids = self.tokenizer.encode(type_label, add_special_tokens=False)
        label_embeds = self.embed_tokens(torch.tensor(label_ids, device=self.device))
        if label_embeds.dim() == 2:
            label_embeds = label_embeds
        pieces.append(label_embeds.to(dtype=self.dtype))

        # 6. Anchors
        anchor_ids = torch.full((num_anchors,), self.anchor_token_id, device=self.device)
        anchor_embeds = self.embed_tokens(anchor_ids)
        if anchor_embeds.dim() == 2:
            anchor_embeds = anchor_embeds
        pieces.append(anchor_embeds.to(dtype=self.dtype))

        node_embeds = torch.cat(pieces, dim=0)
        L = node_embeds.shape[0]
        anchor_positions = list(range(L - num_anchors, L))

        return node_embeds, anchor_positions

    # ══════════════════════════════════════════════════════════════════════
    # Utility: build compact graph with C_T + prompt
    # ══════════════════════════════════════════════════════════════════════

    def build_compact_graph(
        self,
        anchor_tokens: torch.Tensor,          # [N_anchors, 4096]
        ct_token_embed: Optional[torch.Tensor] = None,  # [4096] C_T token embedding
    ) -> Tuple[torch.Tensor, int]:
        """
        Build compact evidence graph: [anchors | C_T | R_static | prompt | Y].

        Returns:
            graph_embeds: [L, 4096]
            decision_position: index of Y token
        """
        pieces = []

        # 1. All anchor tokens from stage 1
        pieces.append(anchor_tokens.to(device=self.device, dtype=self.dtype))
        offset = anchor_tokens.shape[0]

        # 2. C_T target audit token
        if ct_token_embed is None:
            ct_token_embed = self.get_token_embedding(self.anchor_token_id)
        pieces.append(ct_token_embed.unsqueeze(0).to(device=self.device, dtype=self.dtype))
        ct_pos = offset
        offset += 1

        # 3. Static instructions
        static_text = (
            "Judge only the TARGET block. "
            "PAST, FUTURE-1 and FUTURE-2 are context and must not be scored. "
            "Determine whether TARGET contains any anomalous or policy-violating content. "
            "Different violation types need not be mutually exclusive. "
            "Answer only Yes or No.\n"
            "Answer:"
        )
        static_ids = self.tokenizer.encode(static_text, add_special_tokens=False)
        static_embeds = self.embed_tokens(
            torch.tensor(static_ids, device=self.device)
        )
        pieces.append(static_embeds.to(dtype=self.dtype))
        offset += len(static_ids)

        # 4. Decision position (last token of "Answer:")
        decision_position = offset - 1

        graph_embeds = torch.cat(pieces, dim=0)

        return graph_embeds, decision_position, ct_pos


# ══════════════════════════════════════════════════════════════════════════
# Quick smoke test
# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=== InternVLLowLevel smoke test ===")
    wrapper = InternVLLowLevel()

    # Test 1: encode a dummy image
    dummy_img = torch.randn(1, 3, 448, 448)
    print("\nTest 1: encode_images...")
    vit = wrapper.encode_images(dummy_img)
    print(f"  vision tokens shape: {vit.shape}  (expected: [1, 256, 4096])")
    assert vit.shape == (1, 256, 4096), f"Unexpected shape: {vit.shape}"

    # Test 2: build a sample node
    print("\nTest 2: build_node_sequence...")
    node_embeds, anchor_pos = wrapper.build_node_sequence(
        vision_tokens=vit[0],  # [256, 4096]
        type_label="TARGET GLOBAL",
        num_anchors=4,
    )
    print(f"  node_embeds shape: {node_embeds.shape}")
    print(f"  anchor_positions: {anchor_pos}")
    assert len(anchor_pos) == 4
    assert node_embeds.shape[0] == 256 + len(wrapper.tokenizer.encode("TARGET GLOBAL", add_special_tokens=False)) + 4

    # Test 3: stage 1 encode (single node)
    print("\nTest 3: stage1_encode_nodes (single node)...")
    L = node_embeds.shape[0]
    batch_embeds = node_embeds.unsqueeze(0)  # [1, L, 4096]
    batch_mask = torch.ones(1, L)
    results = wrapper.stage1_encode_nodes(
        batch_embeds, batch_mask,
        node_anchor_positions=[anchor_pos],
        output_layer=23,
    )
    print(f"  anchor hidden states: {results[0].shape}  (expected: [4, 4096])")
    assert results[0].shape == (4, 4096)

    # Test 4: stage 2 compact graph
    print("\nTest 4: stage2_graph_forward...")
    all_anchors = results[0]  # [4, 4096] from a single node (simulated)
    graph_embeds, dec_pos, ct_pos = wrapper.build_compact_graph(all_anchors)
    L2 = graph_embeds.shape[0]
    print(f"  graph sequence length: {L2}, decision_pos={dec_pos}, ct_pos={ct_pos}")

    # Build a simple 4D mask: causal + C_T sees all anchors + Y sees C_T
    mask_4d = torch.full((1, 1, L2, L2), float('-inf'))
    # causal
    for i in range(L2):
        for j in range(i + 1):
            mask_4d[0, 0, i, j] = 0.0
    # C_T also sees all anchors
    for j in range(all_anchors.shape[0]):
        mask_4d[0, 0, ct_pos, j] = 0.0
    # Y sees C_T
    mask_4d[0, 0, dec_pos, ct_pos] = 0.0

    out = wrapper.stage2_graph_forward(
        graph_embeds.unsqueeze(0),
        mask_4d,
        decision_positions=[dec_pos],
    )
    yes, no = wrapper.yes_no_logits(out['logits'])
    z = wrapper.log_odds(yes, no)
    p = wrapper.anomaly_probability(z)
    print(f"  Yes logit={yes.item():.2f}, No logit={no.item():.2f}")
    print(f"  log-odds z={z.item():.4f}, p_anomaly={p.item():.4f}")

    print("\n=== All tests passed! ===")
