#!/usr/bin/env python3
"""Enable hybrid KV cache manager (HMA) on MoRIIO for Flash SWA+MLA (217665).

5a4c turns HMA off because MoRIIOConnector does not subclass SupportsHMA.
vLLM then unifies SWA into full attention: "models with sliding window
attention will run with reduced performance." Flash decode POSTs ~4s/token
(391s at max_tokens=100) while prefill stays ~4s.

Upstream:
  - Nixl #32204: FA+SWA HMA (per-group block ids). Not in MoRIIO.
  - MoRIIO #51052: Kimi mamba/KDA SupportsHMA. Unmerged. Wrong hybrid.
  - vLLM main MoRIIO still: class MoRIIOConnector(KVConnectorBase_V1).
  - --no-disable-hybrid-kv-cache-manager raises unless SupportsHMA is on.

This Flash-only patcher:
  1. Subclass SupportsHMA + request_finished_all_groups.
  2. Keep ALL KV-cache groups from get_block_ids() (not [0]).
  3. WRITE notify + save use per-layer group block ids (64-block .attn
     vs 256-block MLA/swa_cache).
  4. wait_for_save transfers registered .attn (indexer still skipped).
  5. 217685: pick the layer's group inside _compute_block_transfer_offsets.
     Engine remote ids are request_info.block_ids (all groups, nested
     lists). write-pick only flattened task.local_block_ids, so
     merge_contiguous_blocks hit TypeError: int() ... not 'list'.
  6. 217691: do not guess group 0/1/2. Build layer→group from each
     kv_cache_groups spec (unwrap UniformTypeKVCacheSpecs) + layer_names.
     Unmatched registered tensors skip transfer (empty ids). Clip SWA
     groups like Nixl (last cdiv(window,block)+1, drop leading null 0).
     Log group table + first write of each layer: group, n_local, n_remote,
     block_size.
  7. 217705: specmap paired (0 unmatched) but g0 MLA spec page is 256
     while MLA 3D geometry used tensor dim[1] (even .attn=64, odd=2).
     Fold kernel rows to spec.block_size like the 5D path. Skip a layer
     if fold cannot apply (geom_bs != spec_bs).
  8. 217718: layout fold patched but g0 WRITE stayed tensor page 64/2.
     UniformType inner specs for most .attn are 64/2; only layers 2/3
     carried the group MLA page 256 (geom-skip). Drive fold + skip from
     the HMA *group* page (max wrapper/inner block_size, g0=256) via
     spec_bs_override. Do not mutate layer_to_spec (attention reads it).
     9. 217732: groupspec reached layout (fold-miss spec=256 geom=64/2 on all
     41 g0 .attn) but 3D required num_blocks % kbpb == 0. Packed dim[0]
     is not that multiple. Floor-fold anyway when num_blocks >= kbpb;
     log shape/num_blocks; drop remainder kernel rows. If num_blocks <
     kbpb, still skip (cannot form one spec page).
    10. 218699: WRITE zip aborted at 8k (`local=17 remote=16` on group-3
     compressor, writes_done peaked 152/243). Prefill clips Nixl
     cdiv(window,page)+1; decode may hold one fewer dest (prefix null
     0 dropped, or one less allocated page). Drop leading 0s *before*
     clip so a null does not occupy a window slot. If local is still
     longer *and dests are present*, tail-align (`local[-len(remote):]`)
     — same as READ's shorter-local tail — and keep the newest SWA
     pages. Does not invent dests; 2k 17=17 is unchanged.
    11. 220216: empty `n_remote` at write-pick is the dest *hint*
     (`meta.remote_block_ids`); real dests arrive later from decode
     alloc notify (`offsets local=0 remote=1` on `.attn`). Returning
     `[], []` dropped the source, so decode sampled an empty full-attn
     cache (Chinese curl). Pass the hint through; tail-align only when
     both sides are non-empty.

Must run AFTER mixed-bs + transfer-gate. Do NOT run skip-swa in the same
cell — 217546 hung because group-0 (256) ids were applied to block-64
.attn; per-group ids are the replacement. GLM/DSV3/Hy3 never run this.
Idempotent. Missing SupportsHMA or a critical anchor is a hard error.
Re-apply on a 217705-era tree adds layout fold + geom-skip.

Usage: apply_moriio_dsv4_supports_hma_fix.py <vllm_install_dir>
       apply_moriio_dsv4_supports_hma_fix.py --selftest
"""
from __future__ import annotations

import os
import re
import sys

CONN_REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
LAYOUT_REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py"
MARKER = "DSV4-HMA"
LAYOUT_MARKER = "DSV4-HMA-GEOM"


HELPERS = '''
def _dsv4_all_group_block_ids(blocks) -> list:
    """HMA: keep every KV-cache group. Non-HMA is a 1-tuple."""
    ids = blocks.get_block_ids()
    if not ids:
        return []
    return [list(g) for g in ids]


def _dsv4_block_id_count(block_ids) -> int:
    if not block_ids:
        return 0
    if isinstance(block_ids[0], (list, tuple)):
        return sum(len(g) for g in block_ids)
    return len(block_ids)


def _dsv4_inner_spec(spec):
    """Unwrap UniformTypeKVCacheSpecs to a representative inner spec."""
    nested = getattr(spec, "kv_cache_specs", None)
    if isinstance(nested, dict) and nested:
        return next(iter(nested.values()))
    return spec


def _dsv4_group_page(spec) -> int:
    """HMA group page for RDMA. Max of wrapper + all UniformType inners.

    217718: build_layer_to_spec stores per-layer inners (even .attn=64,
    odd .attn=2) while group-0 block ids are page 256. First-inner was
    256, so the group log looked right and WRITEs still used 64/2.
    """
    pages = []
    seen = [spec]
    inner = _dsv4_inner_spec(spec)
    if inner is not None and inner is not spec:
        seen.append(inner)
    for obj in seen:
        if obj is None:
            continue
        bs = getattr(obj, "block_size", None)
        if bs:
            pages.append(int(bs))
        nested = getattr(obj, "kv_cache_specs", None)
        if isinstance(nested, dict):
            for v in nested.values():
                b = getattr(v, "block_size", None)
                if b:
                    pages.append(int(b))
    return max(pages) if pages else 0


def _dsv4_group_spec_page(layer_name: str, layer_to_group: dict, group_spec_bs) -> int:
    """Page for this layer's HMA group, or 0 if unmatched."""
    if not group_spec_bs:
        return 0
    gi = _dsv4_lookup_group(layer_name, layer_to_group or {}, len(group_spec_bs))
    if gi is None or gi >= len(group_spec_bs):
        return 0
    return int(group_spec_bs[gi] or 0)


def _dsv4_blocks_per_sw(spec) -> int:
    """Nixl get_sw_clipped_blocks: cdiv(window, block_size) + 1, else 0."""
    inner = _dsv4_inner_spec(spec)
    if inner is None:
        return 0
    sw = getattr(inner, "sliding_window", None) or 0
    bs = getattr(inner, "block_size", None) or 0
    if not sw or not bs:
        return 0
    return (int(sw) + int(bs) - 1) // int(bs) + 1


def _dsv4_build_layer_to_group(kv_cache_config):
    """layer_name -> group index from spec + layer_names. No FA=0/SWA=1 guess.

    217691 dumped unmatched .swa_cache onto group 0 and hardcoded
    .attn->1 / compressor->2. Flash has 5 groups; those indices are not
    stable. Also unwrap UniformTypeKVCacheSpecs.kv_cache_specs keys.
    """
    groups = list(getattr(kv_cache_config, "kv_cache_groups", None) or [])
    layer_to_group = {}
    group_clip = []
    group_spec_bs = []
    for gi, g in enumerate(groups):
        spec = getattr(g, "kv_cache_spec", None)
        names = list(getattr(g, "layer_names", None) or [])
        nested = getattr(spec, "kv_cache_specs", None) if spec is not None else None
        if isinstance(nested, dict):
            for n in nested:
                if n not in names:
                    names.append(n)
        for n in names:
            layer_to_group[n] = gi
        clip = _dsv4_blocks_per_sw(spec)
        group_clip.append(clip)
        page = _dsv4_group_page(spec)
        group_spec_bs.append(page)
        inner = _dsv4_inner_spec(spec)
        logger.info(
            "[dsv4-hma] group=%d spec=%s block_size=%s group_page=%s "
            "sliding_window=%s n_layers=%d clip_blocks=%d sample=%s",
            gi,
            type(inner).__name__ if inner is not None else None,
            getattr(inner, "block_size", None) if inner is not None else None,
            page,
            getattr(inner, "sliding_window", None) if inner is not None else None,
            len(names),
            clip,
            names[:3],
        )
    logger.info(
        "[dsv4-hma] worker kv_groups=%d layer_map=%d specmap=217692 "
        "groupspec=217719",
        len(groups),
        len(layer_to_group),
    )
    return layer_to_group, group_clip, group_spec_bs


def _dsv4_lookup_group(layer_name: str, layer_to_group: dict, n_groups: int):
    """Return group index or None. Never guess FA=0 / SWA=1 / compressor=2."""
    if n_groups <= 1:
        return 0
    if not layer_to_group:
        return None
    if layer_name in layer_to_group:
        gi = int(layer_to_group[layer_name])
        if 0 <= gi < n_groups:
            return gi
        return None
    # Same-tensor aliases only. Never map .swa_cache / state_cache onto
    # parent .attn (different spec, different group).
    best = None
    best_len = -1
    for name, idx in layer_to_group.items():
        if not layer_name.startswith(name + "."):
            continue
        rest = layer_name[len(name) + 1 :]
        if rest not in ("k_cache", "v_cache", "key_cache", "value_cache"):
            continue
        if len(name) > best_len:
            best = int(idx)
            best_len = len(name)
    if best is not None and 0 <= best < n_groups:
        return best
    return None


def _dsv4_pick_group_ids(
    layer_name: str, block_ids, layer_to_group: dict, group_clip=None
) -> list:
    """Select this layer's KV-group block ids (nested) or the flat list.

    217685: decode notify sends all groups as list[list[int]]. Engine
    _prepare_transfer_plan uses that as remote_block_ids. A list element
    in zip() becomes a list offset -> np.fromiter TypeError.

    217691: unmatched name -> [] (do not steal group 0). SWA groups clip
    to last Nixl window. Drop leading null block 0 *before* clip so a
    sentinel does not consume a window slot (218699) — only when that
    group is clipped. Group-0 ``clip_blocks=0``: block 0 is a real page.
    218768 curl skipped ``.attn`` ``local=1 remote=0`` because this drop
    emptied the only dest.
    """
    if not block_ids:
        return []
    if not isinstance(block_ids[0], (list, tuple)):
        return [int(x) for x in block_ids]
    gi = _dsv4_lookup_group(layer_name, layer_to_group or {}, len(block_ids))
    if gi is None:
        return []
    picked = list(block_ids[gi] or [])
    if picked and isinstance(picked[0], (list, tuple)):
        picked = list(picked[0] or [])
    picked = [int(x) for x in picked]
    clip = 0
    if group_clip is not None and gi < len(group_clip):
        clip = int(group_clip[gi] or 0)
    # Leading 0 is a Nixl SWA sentinel only on clipped groups. A lone [0]
    # (curl, group-0 page 256) is the first KV page — do not drop it.
    if clip > 0:
        while picked and picked[0] == 0 and len(picked) > 1:
            picked = picked[1:]
        if len(picked) > clip:
            picked = picked[-clip:]
    return picked


_DSV4_ALIGN_TAIL_LOGS = [0]


def _dsv4_align_write_ids(layer_name, local_ids, remote_ids):
    """WRITE zip: len(local) must not exceed len(remote) once dests exist.

    218699 8k: compressor clip 17 vs decode 16. Keep newest local pages
    (tail), matching READ's shorter-local tail. Empty remote at
    write-pick is the dest hint (218392 / 220216) — pass through.
    """
    local_ids = list(local_ids or [])
    remote_ids = list(remote_ids or [])
    if not local_ids:
        return local_ids, remote_ids
    if not remote_ids:
        n = int(_DSV4_ALIGN_TAIL_LOGS[0])
        if n < 64:
            _DSV4_ALIGN_TAIL_LOGS[0] = n + 1
            logger.info(
                "[dsv4-hma] clip-tail hint layer=%s local=%d remote=0",
                layer_name,
                len(local_ids),
            )
        return local_ids, remote_ids
    n_l, n_r = len(local_ids), len(remote_ids)
    if n_l <= n_r:
        return local_ids, remote_ids
    n = int(_DSV4_ALIGN_TAIL_LOGS[0])
    if n < 64:
        _DSV4_ALIGN_TAIL_LOGS[0] = n + 1
        logger.warning(
            "[dsv4-hma] clip-tail layer=%s local=%d remote=%d",
            layer_name,
            n_l,
            n_r,
        )
    return local_ids[-n_r:], remote_ids

'''


def _replace_once(src: str, old: str, new: str, name: str) -> str:
    n = src.count(old)
    if n == 0:
        raise ValueError(f"anchor missing: {name}")
    if n > 1:
        raise ValueError(f"anchor not unique ({n}x): {name}")
    return src.replace(old, new, 1)


def _add_supports_hma_import(src: str) -> str:
    if re.search(
        r"from vllm\.distributed\.kv_transfer\.kv_connector\.v1\.base import \([^)]*SupportsHMA",
        src,
        re.S,
    ):
        return src
    m = re.search(
        r"from vllm\.distributed\.kv_transfer\.kv_connector\.v1\.base import \((.*?)\)",
        src,
        re.S,
    )
    if not m:
        raise ValueError("anchor missing: import SupportsHMA")
    inner = m.group(1)
    indent_m = re.search(r"\n([ \t]+)\S", inner)
    indent = indent_m.group(1) if indent_m else "    "
    new_inner = inner.rstrip() + f"\n{indent}SupportsHMA,\n"
    return src[: m.start(1)] + new_inner + src[m.end(1) :]


def _patch_offsets_pick(src: str) -> str:
    """217685: flatten HMA groups before layout.compute_block_transfer_offsets."""
    if "DSV4-HMA: 217685" in src:
        return src
    m = re.search(
        r"^([ \t]+)return compute_block_transfer_offsets\(\n"
        r"[ \t]+layer_name=layer_name,\n"
        r"[ \t]+kv_cache=self\.kv_caches\[layer_name\],\n",
        src,
        re.M,
    )
    if not m:
        raise ValueError("anchor missing: return compute_block_transfer_offsets")
    ind = m.group(1)
    insert = (
        f"{ind}# DSV4-HMA: 217685. Engine remote ids are all HMA groups "
        f"(nested lists).\n"
        f"{ind}# merge_contiguous_blocks needs flat ints. Same pick as write-pick.\n"
        f"{ind}_l2g = getattr(self, \"_dsv4_layer_to_group\", {{}}) or {{}}\n"
        f"{ind}_clip = getattr(self, \"_dsv4_group_clip\", None)\n"
        f"{ind}_remote_nested = bool(remote_block_ids) and isinstance(\n"
        f"{ind}    remote_block_ids[0], (list, tuple)\n"
        f"{ind})\n"
        f"{ind}local_block_ids = _dsv4_pick_group_ids("
        f"layer_name, local_block_ids, _l2g, _clip)\n"
        f"{ind}remote_block_ids = _dsv4_pick_group_ids("
        f"layer_name, remote_block_ids, _l2g, _clip)\n"
        f"{ind}if _remote_nested and not getattr("
        f"self, \"_dsv4_logged_nested_remote\", False):\n"
        f"{ind}    self._dsv4_logged_nested_remote = True\n"
        f"{ind}    logger.info(\n"
        f"{ind}        \"[dsv4-hma] 217685 flattening nested remote block ids\",\n"
        f"{ind}    )\n"
        f"{ind}logger.debug(\n"
        f"{ind}    \"[dsv4-hma] offsets layer=%s local=%d remote=%d\",\n"
        f"{ind}    layer_name, len(local_block_ids), len(remote_block_ids),\n"
        f"{ind})\n"
        f"{ind}local_block_ids, remote_block_ids = _dsv4_align_write_ids(\n"
        f"{ind}    layer_name, local_block_ids, remote_block_ids\n"
        f"{ind})\n"
    )
    return src[: m.start()] + insert + src[m.start() :]


OLD_PICK_AFTER_INT = """    clip = 0
    if group_clip is not None and gi < len(group_clip):
        clip = int(group_clip[gi] or 0)
    if clip > 0 and len(picked) > clip:
        picked = picked[-clip:]
    while picked and picked[0] == 0:
        picked = picked[1:]
    return picked
"""

NEW_PICK_AFTER_INT = """    while picked and picked[0] == 0:
        picked = picked[1:]
    clip = 0
    if group_clip is not None and gi < len(group_clip):
        clip = int(group_clip[gi] or 0)
    if clip > 0 and len(picked) > clip:
        picked = picked[-clip:]
    return picked
"""

# 218768: do not drop block 0 on unclipped group-0 (curl dest was [0]).
NEWER_PICK_AFTER_INT = """    clip = 0
    if group_clip is not None and gi < len(group_clip):
        clip = int(group_clip[gi] or 0)
    if clip > 0:
        while picked and picked[0] == 0 and len(picked) > 1:
            picked = picked[1:]
        if len(picked) > clip:
            picked = picked[-clip:]
    return picked
"""

LAYOUT_LONGER_RAISE = """    if len(local_block_ids) > len(remote_block_ids):
        raise ValueError(
            "local_block_ids longer than remote_block_ids: "
            f"{len(local_block_ids)} > {len(remote_block_ids)}"
        )
"""

LAYOUT_LONGER_TAIL = """    # DSV4-HMA-ALIGN-TAIL: 218699. Prefill SWA Nixl clip is +1 vs decode.
    # WRITE zip used to abort (8k 17>16). Keep newest local pages.
    # Empty remote here is dests-missing at zip, not the WRITE hint
    # (hint is handled in _dsv4_align_write_ids at write-pick).
    if len(local_block_ids) > len(remote_block_ids):
        if remote_block_ids:
            local_block_ids = local_block_ids[-len(remote_block_ids) :]
        else:
            local_block_ids = []
"""

# edbd649 returned [], [] on empty remote. 220216: that is the WRITE hint.
ALIGN_EMPTY_SKIP = '''    if not remote_ids:
        n = int(_DSV4_ALIGN_TAIL_LOGS[0])
        if n < 64:
            _DSV4_ALIGN_TAIL_LOGS[0] = n + 1
            logger.warning(
                "[dsv4-hma] clip-tail skip layer=%s local=%d remote=0",
                layer_name,
                len(local_ids),
            )
        return [], []
'''

ALIGN_EMPTY_HINT = '''    if not remote_ids:
        n = int(_DSV4_ALIGN_TAIL_LOGS[0])
        if n < 64:
            _DSV4_ALIGN_TAIL_LOGS[0] = n + 1
            logger.info(
                "[dsv4-hma] clip-tail hint layer=%s local=%d remote=0",
                layer_name,
                len(local_ids),
            )
        return local_ids, remote_ids
'''


def _align_helper_block() -> str:
    start = HELPERS.find("_DSV4_ALIGN_TAIL_LOGS = [0]")
    if start < 0:
        raise ValueError("HELPERS missing _dsv4_align_write_ids")
    return HELPERS[start:]


def _insert_align_offsets(src: str) -> str:
    if "local_block_ids, remote_block_ids = _dsv4_align_write_ids(" in src:
        return src
    m = re.search(
        r'^([ \t]*)logger\.debug\(\n'
        r'[ \t]*"\[dsv4-hma\] offsets layer=%s local=%d remote=%d",\n'
        r'[ \t]*layer_name, len\(local_block_ids\), len\(remote_block_ids\),\n'
        r'[ \t]*\)\n',
        src,
        re.M,
    )
    if not m:
        raise ValueError("anchor missing: offsets debug log")
    ind = m.group(1)
    insert = (
        f"{m.group(0)}"
        f"{ind}local_block_ids, remote_block_ids = _dsv4_align_write_ids(\n"
        f"{ind}    layer_name, local_block_ids, remote_block_ids\n"
        f"{ind})\n"
    )
    return src[: m.start()] + insert + src[m.end() :]


def _insert_align_write_pick(src: str) -> str:
    if "_dsv4_align_write_ids(layer_name, _local, _remote)" in src:
        return src
    m = re.search(
        r'(logger\.error\(\n'
        r'[ \t]*"\[dsv4-hma\] unmatched layer=%s; skip transfer",\n'
        r'[ \t]*layer_name,\n'
        r'[ \t]*\)\n)'
        r'([ \t]*)self\.schedule_write_blocks\(',
        src,
    )
    if not m:
        raise ValueError("anchor missing: write-pick unmatched / schedule_write_blocks")
    ind = m.group(2)
    insert = (
        f"{m.group(1)}"
        f"{ind}_local, _remote = _dsv4_align_write_ids("
        f"layer_name, _local, _remote)\n"
    )
    return src[: m.start()] + insert + src[m.end(1) :]


def _upgrade_align_write(src: str) -> str:
    """218699 tail-align; 218768 keep block 0; 220216 empty-hint pass-through."""
    if OLD_PICK_AFTER_INT in src:
        src = src.replace(OLD_PICK_AFTER_INT, NEWER_PICK_AFTER_INT, 1)
    if NEW_PICK_AFTER_INT in src:
        src = src.replace(NEW_PICK_AFTER_INT, NEWER_PICK_AFTER_INT, 1)
    if "def _dsv4_align_write_ids(" not in src:
        if NEWER_PICK_AFTER_INT not in src:
            raise ValueError("anchor missing: pick clip/drop for align helper")
        src = src.replace(
            NEWER_PICK_AFTER_INT,
            NEWER_PICK_AFTER_INT + "\n\n" + _align_helper_block(),
            1,
        )
    if ALIGN_EMPTY_SKIP in src:
        src = src.replace(ALIGN_EMPTY_SKIP, ALIGN_EMPTY_HINT, 1)
    src = _insert_align_offsets(src)
    src = _insert_align_write_pick(src)
    return src


def _patch_layout_align(src: str) -> str:
    """Last-guard in layout.py so a missed connector align cannot kill the worker."""
    if "DSV4-HMA-ALIGN-TAIL" in src:
        return src
    if LAYOUT_LONGER_RAISE not in src:
        return src
    return src.replace(LAYOUT_LONGER_RAISE, LAYOUT_LONGER_TAIL, 1)


def _fold_mla_3d(num_blocks: int, tensor_bs: int, spec_bs: int) -> tuple[int, int, int]:
    """Map kernel rows onto one HMA spec page. kbpb=1 means no fold.

    217732: packed dim[0] may not divide kbpb. Floor when num_blocks >= kbpb.
    """
    if (
        spec_bs > 0
        and tensor_bs > 0
        and spec_bs != tensor_bs
        and spec_bs % tensor_bs == 0
    ):
        kbpb = spec_bs // tensor_bs
        if num_blocks >= kbpb:
            return num_blocks // kbpb, spec_bs, kbpb
    return num_blocks, tensor_bs, 1


LAYOUT_MLA_3D_OLD = """    if is_mla_cache and len(shape) == 3:
        num_blocks, block_size, latent_dim = shape
        slot_size_bytes = latent_dim * element_size
        block_len = block_size * slot_size_bytes
        return LayerTransferGeometry(
            num_blocks=num_blocks,
            block_size=block_size,
            block_len=block_len,
            slot_size_bytes=slot_size_bytes,
            block_stride=stride[0],
            local_kv_stride=None,
            remote_kv_stride=None,
            transfers_per_block=1,
            regions_per_block=1,
            split_kv_regions=False,
        )
"""

LAYOUT_MLA_3D_NEW = """    if is_mla_cache and len(shape) == 3:
        num_blocks, block_size, latent_dim = shape
        slot_size_bytes = latent_dim * element_size
        block_len = block_size * slot_size_bytes
        block_stride = stride[0]
        # DSV4-HMA-GEOM: 217705. MLA 3D used tensor dim[1] as the page
        # (even .attn=64, odd .attn=2) while the HMA group spec is 256.
        # Same fold as the 5D kernel-block path: one logical block_id
        # spans spec.block_size / kernel_bs rows.
        spec_bs = int(spec_bs_override or 0) or int(getattr(spec, "block_size", 0) or 0)
        if (
            spec_bs > 0
            and block_size > 0
            and spec_bs != block_size
            and spec_bs % block_size == 0
        ):
            kbpb = spec_bs // block_size
            # DSV4-HMA-GEOM-FLOOR: 217732. Packed dim[0] was not
            # divisible by kbpb (64->256: 4; 2->256: 128). Still map
            # each logical id to kbpb kernel rows; drop the remainder.
            if num_blocks >= kbpb:
                num_blocks = num_blocks // kbpb
                block_stride = stride[0] * kbpb
                block_len = block_len * kbpb
                block_size = spec_bs
        return LayerTransferGeometry(
            num_blocks=num_blocks,
            block_size=block_size,
            block_len=block_len,
            slot_size_bytes=slot_size_bytes,
            block_stride=block_stride,
            local_kv_stride=None,
            remote_kv_stride=None,
            transfers_per_block=1,
            regions_per_block=1,
            split_kv_regions=False,
        )
"""


def _patch_geom_skip(src: str) -> str:
    """Skip RDMA when tensor page still disagrees with the HMA spec page."""
    if "geom-skip layer=" in src:
        return src
    m = re.search(
        r"^([ \t]+)return compute_block_transfer_offsets\(\n"
        r"[ \t]+layer_name=layer_name,\n",
        src,
        re.M,
    )
    if not m:
        raise ValueError("anchor missing: geom-skip compute_block_transfer_offsets")
    ind = m.group(1)
    insert = (
        f"{ind}# DSV4-HMA-GEOM: 217705. After kernel-row fold, geom page\n"
        f"{ind}# must match the group spec. Otherwise skip this layer.\n"
        f"{ind}_spec_bs = _dsv4_group_spec_page(\n"
        f"{ind}    layer_name,\n"
        f"{ind}    getattr(self, \"_dsv4_layer_to_group\", {{}}) or {{}},\n"
        f"{ind}    getattr(self, \"_dsv4_group_spec_bs\", None) or [],\n"
        f"{ind})\n"
        f"{ind}if _spec_bs:\n"
        f"{ind}    _geom_bs = int(\n"
        f"{ind}        self._get_layer_transfer_geometry(layer_name).block_size\n"
        f"{ind}    )\n"
        f"{ind}    if _geom_bs != _spec_bs:\n"
        f"{ind}        if not getattr(self, \"_dsv4_logged_geom_skip\", None):\n"
        f"{ind}            self._dsv4_logged_geom_skip = set()\n"
        f"{ind}        if layer_name not in self._dsv4_logged_geom_skip:\n"
        f"{ind}            self._dsv4_logged_geom_skip.add(layer_name)\n"
        f"{ind}            logger.error(\n"
        f"{ind}                \"[dsv4-hma] geom-skip layer=%s spec_bs=%d geom_bs=%d\",\n"
        f"{ind}                layer_name, _spec_bs, _geom_bs,\n"
        f"{ind}            )\n"
        f"{ind}        return [], [], []\n"
    )
    return src[: m.start()] + insert + src[m.start() :]


def _upgrade_geom_skip_groupspec(src: str) -> str:
    """217718 geom-skip used layer_to_spec (64/2). Use HMA group page."""
    if "geom-skip layer=" not in src:
        return _patch_geom_skip(src)
    if "_spec_bs = _dsv4_group_spec_page(" in src:
        return src
    src2, n = re.subn(
        r"^([ \t]+)_spec = \(getattr\(self, \"layer_to_spec\", None\) or \{\}\)\.get\(\n"
        r"[ \t]+layer_name\n"
        r"[ \t]+\)\n"
        r"[ \t]+_spec_bs = int\(getattr\(_spec, \"block_size\", 0\) or 0\) "
        r"if _spec is not None else 0\n",
        (
            r"\1_spec_bs = _dsv4_group_spec_page(\n"
            r"\1    layer_name,\n"
            r'\1    getattr(self, "_dsv4_layer_to_group", {}) or {},\n'
            r'\1    getattr(self, "_dsv4_group_spec_bs", None) or [],\n'
            r"\1)\n"
        ),
        src,
        count=1,
        flags=re.M,
    )
    if n != 1:
        raise ValueError("anchor missing: geom-skip layer_to_spec -> group page")
    return src2


def _patch_geometry_override(src: str) -> str:
    """Pass HMA group page into layout get_layer_transfer_geometry."""
    if "DSV4-HMA-GROUP-SPEC: 217719" in src:
        return src
    old = (
        "    def _get_layer_transfer_geometry(\n"
        "        self, layer_name: str, remote_num_blocks: int | None = None\n"
        "    ) -> LayerTransferGeometry:\n"
        "        return get_layer_transfer_geometry(\n"
        "            layer_name,\n"
        "            self.kv_caches[layer_name],\n"
        "            self.layer_to_spec,\n"
        "            remote_num_blocks,\n"
        "        )\n"
    )
    new = (
        "    def _get_layer_transfer_geometry(\n"
        "        self, layer_name: str, remote_num_blocks: int | None = None\n"
        "    ) -> LayerTransferGeometry:\n"
        "        # DSV4-HMA-GROUP-SPEC: 217719. Fold MLA 3D to the HMA group\n"
        "        # page (g0=256), not UniformType per-layer inners (64/2).\n"
        "        _gbs = _dsv4_group_spec_page(\n"
        "            layer_name,\n"
        "            getattr(self, \"_dsv4_layer_to_group\", {}) or {},\n"
        "            getattr(self, \"_dsv4_group_spec_bs\", None) or [],\n"
        "        )\n"
        "        geom = get_layer_transfer_geometry(\n"
        "            layer_name,\n"
        "            self.kv_caches[layer_name],\n"
        "            self.layer_to_spec,\n"
        "            remote_num_blocks,\n"
        "            spec_bs_override=_gbs or None,\n"
        "        )\n"
        "        if _gbs:\n"
        "            if not getattr(self, \"_dsv4_logged_fold\", None):\n"
        "                self._dsv4_logged_fold = set()\n"
        "            if layer_name not in self._dsv4_logged_fold:\n"
        "                self._dsv4_logged_fold.add(layer_name)\n"
        "                if geom.block_size == _gbs:\n"
        "                    logger.info(\n"
        "                        \"[dsv4-hma] fold-ok layer=%s spec_bs=%d "
        "geom_bs=%d num_blocks=%d shape=%s\",\n"
        "                        layer_name, _gbs, geom.block_size,\n"
        "                        geom.num_blocks,\n"
        "                        tuple(self.kv_caches[layer_name].shape),\n"
        "                    )\n"
        "                else:\n"
        "                    logger.error(\n"
        "                        \"[dsv4-hma] fold-miss layer=%s spec_bs=%d "
        "geom_bs=%d num_blocks=%d shape=%s\",\n"
        "                        layer_name, _gbs, geom.block_size,\n"
        "                        geom.num_blocks,\n"
        "                        tuple(self.kv_caches[layer_name].shape),\n"
        "                    )\n"
        "        return geom\n"
    )
    if old not in src:
        raise ValueError("anchor missing: _get_layer_transfer_geometry")
    return src.replace(old, new, 1)


def _patch_offsets_spec_override(src: str) -> str:
    """Same group page on the layout offsets path (not only the wrapper)."""
    if "spec_bs_override=_dsv4_group_spec_page(" in src:
        return src
    old = (
        "            layer_to_spec=self.layer_to_spec,\n"
        "            local_block_ids=local_block_ids,\n"
    )
    new = (
        "            layer_to_spec=self.layer_to_spec,\n"
        "            spec_bs_override=_dsv4_group_spec_page(\n"
        "                layer_name,\n"
        "                getattr(self, \"_dsv4_layer_to_group\", {}) or {},\n"
        "                getattr(self, \"_dsv4_group_spec_bs\", None) or [],\n"
        "            ) or None,\n"
        "            local_block_ids=local_block_ids,\n"
    )
    n = src.count(old)
    if n != 1:
        raise ValueError(
            f"anchor missing: offsets layer_to_spec kwargs (found {n})"
        )
    return src.replace(old, new, 1)


def _upgrade_group_spec(src: str) -> str:
    """217718 retry: group page fold. Idempotent on a GROUP-SPEC tree."""
    if "def _dsv4_group_page(" not in src:
        src = _replace_specmap_helpers(src)
    old_unpack = (
        "        self._dsv4_layer_to_group, self._dsv4_group_clip = (\n"
        "            _dsv4_build_layer_to_group(kv_cache_config)\n"
        "        )  # DSV4-HMA-SPECMAP\n"
    )
    new_unpack = (
        "        self._dsv4_layer_to_group, self._dsv4_group_clip, "
        "self._dsv4_group_spec_bs = (\n"
        "            _dsv4_build_layer_to_group(kv_cache_config)\n"
        "        )  # DSV4-HMA-SPECMAP DSV4-HMA-GROUP-SPEC\n"
    )
    if old_unpack in src:
        src = src.replace(old_unpack, new_unpack, 1)
    src = _upgrade_geom_skip_groupspec(src)
    src = _patch_geometry_override(src)
    src = _patch_offsets_spec_override(src)
    return src


LAYOUT_GEOM_SIG_OLD = """def get_layer_transfer_geometry(
    layer_name: str,
    kv_cache: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
    remote_num_blocks: int | None = None,
) -> LayerTransferGeometry:"""

LAYOUT_GEOM_SIG_NEW = """def get_layer_transfer_geometry(
    layer_name: str,
    kv_cache: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
    remote_num_blocks: int | None = None,
    spec_bs_override: int | None = None,
) -> LayerTransferGeometry:"""

LAYOUT_OFFSETS_GEOM_OLD = """    geometry = get_layer_transfer_geometry(
        layer_name, kv_cache, layer_to_spec, remote_num_blocks
    )"""

LAYOUT_OFFSETS_GEOM_NEW = """    geometry = get_layer_transfer_geometry(
        layer_name, kv_cache, layer_to_spec, remote_num_blocks,
        spec_bs_override=spec_bs_override,
    )"""

LAYOUT_OFFSETS_ARG_OLD = """    remote_num_blocks: int,
    merge_fn: Callable["""

LAYOUT_OFFSETS_ARG_NEW = """    remote_num_blocks: int,
    spec_bs_override: int | None = None,
    merge_fn: Callable["""

LAYOUT_FLOOR_FROM_DIVISIBLE_OLD = """            and spec_bs % block_size == 0
            and num_blocks % (spec_bs // block_size) == 0
        ):
            kbpb = spec_bs // block_size
            num_blocks = num_blocks // kbpb
            block_stride = stride[0] * kbpb
            block_len = block_len * kbpb
            block_size = spec_bs
"""

LAYOUT_FLOOR_FROM_DIVISIBLE_NEW = """            and spec_bs % block_size == 0
        ):
            kbpb = spec_bs // block_size
            # DSV4-HMA-GEOM-FLOOR: 217732. Packed dim[0] was not
            # divisible by kbpb (64->256: 4; 2->256: 128). Still map
            # each logical id to kbpb kernel rows; drop the remainder.
            if num_blocks >= kbpb:
                num_blocks = num_blocks // kbpb
                block_stride = stride[0] * kbpb
                block_len = block_len * kbpb
                block_size = spec_bs
"""


def patch_layout(path: str) -> int:
    """Fold MLA 3D kernel pages onto the HMA group page in moriio_layout.py."""
    if not os.path.isfile(path):
        print(f"[dsv4-hma] {LAYOUT_REL} not found -- skipping layout fold.")
        return 0
    src = open(path).read()
    applied: list[str] = []

    if LAYOUT_GEOM_SIG_OLD in src:
        src = src.replace(LAYOUT_GEOM_SIG_OLD, LAYOUT_GEOM_SIG_NEW, 1)
        applied.append("geom-sig")
    elif "spec_bs_override: int | None = None" not in src.split(
        "def get_layer_transfer_geometry", 1
    )[-1][:500]:
        print(
            f"[dsv4-hma] ERROR: get_layer_transfer_geometry signature "
            f"anchor missing in {path}.",
            file=sys.stderr,
        )
        return 1

    if LAYOUT_MLA_3D_OLD in src:
        src = src.replace(LAYOUT_MLA_3D_OLD, LAYOUT_MLA_3D_NEW, 1)
        applied.append("mla-3d-fold")
    elif LAYOUT_FLOOR_FROM_DIVISIBLE_OLD in src:
        src = src.replace(
            LAYOUT_FLOOR_FROM_DIVISIBLE_OLD, LAYOUT_FLOOR_FROM_DIVISIBLE_NEW, 1
        )
        applied.append("mla-3d-floor")
    elif LAYOUT_MARKER in src and "spec_bs = int(getattr(spec" in src:
        src = src.replace(
            "spec_bs = int(getattr(spec, \"block_size\", 0) or 0)",
            "spec_bs = int(spec_bs_override or 0) or int("
            "getattr(spec, \"block_size\", 0) or 0)",
            1,
        )
        applied.append("mla-3d-override")
    elif "spec_bs = int(spec_bs_override" not in src:
        print(
            f"[dsv4-hma] ERROR: MLA 3D geometry anchor missing in {path}.",
            file=sys.stderr,
        )
        return 1

    if LAYOUT_OFFSETS_ARG_OLD in src:
        src = src.replace(LAYOUT_OFFSETS_ARG_OLD, LAYOUT_OFFSETS_ARG_NEW, 1)
        applied.append("offsets-sig")
    if LAYOUT_OFFSETS_GEOM_OLD in src:
        src = src.replace(LAYOUT_OFFSETS_GEOM_OLD, LAYOUT_OFFSETS_GEOM_NEW, 1)
        applied.append("offsets-call")

    if LAYOUT_MARKER not in src:
        print("[dsv4-hma] ERROR: layout marker missing after fold patch.", file=sys.stderr)
        return 1
    if "spec_bs = int(spec_bs_override" not in src:
        print("[dsv4-hma] ERROR: spec_bs_override unused in 3D fold.", file=sys.stderr)
        return 1
    if "spec_bs_override=spec_bs_override" not in src:
        print("[dsv4-hma] ERROR: offsets did not pass spec_bs_override.", file=sys.stderr)
        return 1
    if "DSV4-HMA-GEOM-FLOOR" not in src:
        print("[dsv4-hma] ERROR: floor fold missing after layout patch.", file=sys.stderr)
        return 1
    src2 = _patch_layout_align(src)
    if src2 != src:
        applied.append("align-tail")
        src = src2
    if not applied:
        print(f"[dsv4-hma] layout already folded with floor in {path} -- no-op.")
        return 0
    tmp = path + ".dsv4geom"
    with open(tmp, "w") as f:
        f.write(src)
    os.replace(tmp, path)
    try:
        import py_compile

        py_compile.compile(path, doraise=True)
    except Exception as e:  # noqa: BLE001
        print(f"[dsv4-hma] ERROR: layout py_compile failed: {e}", file=sys.stderr)
        return 1
    print(f"[dsv4-hma] patched layout fold ({', '.join(applied)}) in {path}")
    return 0


def _specmap_helper_block() -> str:
    start = HELPERS.find("def _dsv4_inner_spec")
    if start < 0:
        raise ValueError("HELPERS missing _dsv4_inner_spec")
    return HELPERS[start:]


def _replace_specmap_helpers(src: str) -> str:
    """Replace lookup+pick (any era) with specmap helpers."""
    new = _specmap_helper_block()
    pat = re.compile(
        r"def _dsv4_(?:inner_spec|blocks_per_sw|build_layer_to_group|lookup_group)\("
        r".*?"
        r"def _dsv4_pick_group_ids\(.*?\n"
        r"    return (?:picked|\[int\(x\) for x in picked\]|list\(block_ids\[gi\] or \[\]\))\n"
        r"(?:\n+_DSV4_ALIGN_TAIL_LOGS = \[0\]\n.*?\n    return local_ids\[-n_r:\], remote_ids\n)?",
        re.S,
    )
    m = pat.search(src)
    if not m:
        raise ValueError("anchor missing: specmap helpers")
    return src[: m.start()] + new + src[m.end() :]


def _upgrade_layer_map(src: str) -> str:
    """Swap FA=0/SWA=1 name loop for specmap builder."""
    if "DSV4-HMA-SPECMAP" in src:
        return src
    old = (
        "        self._dsv4_layer_to_group = {}\n"
        "        for _gi, _g in enumerate(\n"
        "            getattr(kv_cache_config, \"kv_cache_groups\", None) or []\n"
        "        ):\n"
        "            for _ln in getattr(_g, \"layer_names\", None) or []:\n"
        "                self._dsv4_layer_to_group[_ln] = _gi\n"
        "        logger.info(\n"
        "            \"[dsv4-hma] worker kv_groups=%d layer_map=%d\",\n"
        "            len(getattr(kv_cache_config, \"kv_cache_groups\", None) or []),\n"
        "            len(self._dsv4_layer_to_group),\n"
        "        )\n"
    )
    new = (
        "        self._dsv4_layer_to_group, self._dsv4_group_clip, "
        "self._dsv4_group_spec_bs = (\n"
        "            _dsv4_build_layer_to_group(kv_cache_config)\n"
        "        )  # DSV4-HMA-SPECMAP DSV4-HMA-GROUP-SPEC\n"
    )
    if old not in src:
        raise ValueError("anchor missing: layer_map loop")
    return src.replace(old, new, 1)


def _upgrade_pick_calls(src: str) -> str:
    """Pass group_clip into pick at write + offsets sites."""
    src2, n = re.subn(
        r'^([ \t]+)_l2g = getattr\(self, "_dsv4_layer_to_group", \{\}\) or \{\}\n'
        r"(?![ \t]+_clip = )",
        r'\1_l2g = getattr(self, "_dsv4_layer_to_group", {}) or {}\n'
        r'\1_clip = getattr(self, "_dsv4_group_clip", None)\n',
        src,
        flags=re.M,
    )
    src = src2
    replacements = (
        (
            "layer_name, meta.local_block_ids, _l2g\n",
            "layer_name, meta.local_block_ids, _l2g, _clip\n",
        ),
        (
            "layer_name, meta.remote_block_ids, _l2g\n",
            "layer_name, meta.remote_block_ids, _l2g, _clip\n",
        ),
        (
            "_dsv4_pick_group_ids(layer_name, local_block_ids, _l2g)",
            "_dsv4_pick_group_ids(layer_name, local_block_ids, _l2g, _clip)",
        ),
        (
            "_dsv4_pick_group_ids(layer_name, remote_block_ids, _l2g)",
            "_dsv4_pick_group_ids(layer_name, remote_block_ids, _l2g, _clip)",
        ),
    )
    for old, new in replacements:
        if old in src and new not in src:
            src = src.replace(old, new)
    return src


def _upgrade_write_pair_log(src: str) -> str:
    """INFO once per layer: group, n_local, n_remote, block_size."""
    if "[dsv4-hma] write layer=%s group=%s n_local=%d" in src:
        return src
    # Indent of the debug block varies; match generically.
    m = re.search(
        r"^([ \t]*)logger\.debug\(\n"
        r"[ \t]*\"\[dsv4-hma\] write layer=%s local=%d remote=%d\",\n"
        r"[ \t]*layer_name,\n"
        r"[ \t]*len\(_local\),\n"
        r"[ \t]*len\(_remote\),\n"
        r"[ \t]*\)\n",
        src,
        re.M,
    )
    if not m:
        return src
    ind = m.group(1)
    new = (
        f"{ind}_gi = _dsv4_lookup_group(\n"
        f"{ind}    layer_name, _l2g,\n"
        f"{ind}    len(meta.local_block_ids) if meta.local_block_ids else 1,\n"
        f"{ind})\n"
        f"{ind}if not getattr(self, \"_dsv4_logged_pair\", None):\n"
        f"{ind}    self._dsv4_logged_pair = set()\n"
        f"{ind}if layer_name not in self._dsv4_logged_pair:\n"
        f"{ind}    self._dsv4_logged_pair.add(layer_name)\n"
        f"{ind}    _bs = None\n"
        f"{ind}    try:\n"
        f"{ind}        _bs = self._get_layer_transfer_geometry("
        f"layer_name).block_size\n"
        f"{ind}    except Exception:\n"
        f"{ind}        pass\n"
        f"{ind}    logger.info(\n"
        f"{ind}        \"[dsv4-hma] write layer=%s group=%s n_local=%d "
        f"n_remote=%d block_size=%s\",\n"
        f"{ind}        layer_name, _gi, len(_local), len(_remote), _bs,\n"
        f"{ind}    )\n"
        f"{ind}    if _gi is None:\n"
        f"{ind}        logger.error(\n"
        f"{ind}            \"[dsv4-hma] unmatched layer=%s; skip transfer\",\n"
        f"{ind}            layer_name,\n"
        f"{ind}        )\n"
    )
    return src[: m.start()] + new + src[m.end() :]


def _ensure_specmap_helpers(src: str) -> str:
    if "def _dsv4_build_layer_to_group(" in src and "unmatched name -> []" in src:
        return src
    if "def _dsv4_lookup_group(" in src:
        return _replace_specmap_helpers(src)
    m = re.search(
        r"def _dsv4_block_id_count\(block_ids\) -> int:\n"
        r".*?return len\(block_ids\)\n",
        src,
        re.S,
    )
    if not m:
        raise ValueError("anchor missing: _dsv4_block_id_count for specmap insert")
    return src[: m.end()] + "\n" + _specmap_helper_block() + src[m.end() :]


def _upgrade_specmap(src: str) -> str:
    src = _ensure_specmap_helpers(src)
    src = _upgrade_layer_map(src)
    src = _upgrade_pick_calls(src)
    src = _upgrade_write_pair_log(src)
    return src


def _refresh_pick_helpers(src: str) -> str:
    """Upgrade helpers on a tree that already has HMA."""
    if "DSV4-HMA-SPECMAP" in src and "def _dsv4_build_layer_to_group(" in src:
        return src
    return _ensure_specmap_helpers(src)


def _write_connector(path: str, src: str, applied: list[str]) -> int:
    tmp = path + ".dsv4hma"
    with open(tmp, "w") as f:
        f.write(src)
    os.replace(tmp, path)
    try:
        import py_compile

        py_compile.compile(path, doraise=True)
    except Exception as e:  # noqa: BLE001
        print(f"[dsv4-hma] ERROR: py_compile failed: {e}", file=sys.stderr)
        return 1
    print(f"[dsv4-hma] patched ({', '.join(applied)}) in {path}")
    return 0


def patch_connector(path: str) -> int:
    src = open(path).read()
    hma_on = MARKER in src and "request_finished_all_groups" in src
    offsets_on = "DSV4-HMA: 217685" in src
    specmap_on = "DSV4-HMA-SPECMAP" in src
    geom_skip_on = "geom-skip layer=" in src
    groupspec_on = "DSV4-HMA-GROUP-SPEC: 217719" in src
    if hma_on and offsets_on and specmap_on and geom_skip_on and groupspec_on:
        orig = src
        try:
            src = _upgrade_align_write(src)
        except ValueError as e:
            print(f"[dsv4-hma] ERROR: {e}", file=sys.stderr)
            return 1
        if src == orig:
            print(f"[dsv4-hma] already patched in {path} -- no-op.")
            return 0
        return _write_connector(path, src, ["align-write-220216"])

    base_py = os.path.normpath(
        os.path.join(os.path.dirname(path), "..", "base.py")
    )
    if not os.path.isfile(base_py):
        print(f"[dsv4-hma] ERROR: {base_py} not found.", file=sys.stderr)
        return 1
    base_src = open(base_py).read()
    if "class SupportsHMA" not in base_src:
        print(
            "[dsv4-hma] ERROR: SupportsHMA missing in kv_connector/v1/base.py. "
            "Cannot enable HMA on this image.",
            file=sys.stderr,
        )
        return 1

    if hma_on:
        applied: list[str] = []
        try:
            src2 = _upgrade_specmap(src)
            if src2 != src:
                applied.append("specmap")
            src = src2
            src2 = _patch_offsets_pick(src)
            if src2 != src:
                applied.append("offsets-pick")
            src = src2
            src2 = _patch_geom_skip(src)
            if src2 != src:
                applied.append("geom-skip")
            src = src2
            src2 = _upgrade_group_spec(src)
            if src2 != src:
                applied.append("groupspec")
            src = src2
            src2 = _upgrade_align_write(src)
            if src2 != src:
                applied.append("align-write-220216")
            src = src2
        except ValueError as e:
            print(f"[dsv4-hma] ERROR: {e}", file=sys.stderr)
            return 1
        if "DSV4-HMA-SPECMAP" not in src:
            print("[dsv4-hma] ERROR: specmap missing after upgrade.", file=sys.stderr)
            return 1
        if "DSV4-HMA-GROUP-SPEC: 217719" not in src:
            print("[dsv4-hma] ERROR: groupspec missing after upgrade.", file=sys.stderr)
            return 1
        if not applied:
            print(f"[dsv4-hma] already patched in {path} -- no-op.")
            return 0
        if "DSV4-HMA: 217685" not in src:
            print("[dsv4-hma] ERROR: 217685 offsets pick missing after patch.", file=sys.stderr)
            return 1
        if "geom-skip layer=" not in src:
            print("[dsv4-hma] ERROR: geom-skip missing after upgrade.", file=sys.stderr)
            return 1
        if "def _dsv4_align_write_ids(" not in src:
            print("[dsv4-hma] ERROR: align-write missing after upgrade.", file=sys.stderr)
            return 1
        return _write_connector(path, src, applied)

    orig = src
    applied = []

    try:
        src = _add_supports_hma_import(src)
        applied.append("import")
    except ValueError as e:
        print(f"[dsv4-hma] ERROR: {e}", file=sys.stderr)
        return 1

    if "def _dsv4_all_group_block_ids" not in src:
        needle = "class MoRIIOConnector(KVConnectorBase_V1"
        idx = src.find(needle)
        if idx < 0:
            print("[dsv4-hma] ERROR: class MoRIIOConnector not found.", file=sys.stderr)
            return 1
        src = src[:idx] + HELPERS + src[idx:]
        applied.append("helpers")

    try:
        src = _replace_once(
            src,
            "class MoRIIOConnector(KVConnectorBase_V1):",
            "class MoRIIOConnector(KVConnectorBase_V1, SupportsHMA):  # DSV4-HMA",
            "class bases",
        )
        applied.append("bases")
    except ValueError as e:
        print(f"[dsv4-hma] ERROR: {e}", file=sys.stderr)
        return 1

    try:
        src = _replace_once(
            src,
            "            self.connector_scheduler: MoRIIOConnectorScheduler | None = (\n"
            "                MoRIIOConnectorScheduler(vllm_config, self.engine_id)\n"
            "            )",
            "            self.connector_scheduler: MoRIIOConnectorScheduler | None = (\n"
            "                MoRIIOConnectorScheduler(\n"
            "                    vllm_config, self.engine_id, kv_cache_config\n"
            "                )\n"
            "            )",
            "scheduler kv_cache_config",
        )
        applied.append("scheduler-ctor")
    except ValueError:
        try:
            src = _replace_once(
                src,
                "MoRIIOConnectorScheduler(vllm_config, self.engine_id)",
                "MoRIIOConnectorScheduler(vllm_config, self.engine_id, kv_cache_config)",
                "scheduler kv_cache_config (compact)",
            )
            applied.append("scheduler-ctor-compact")
        except ValueError as e:
            print(f"[dsv4-hma] ERROR: {e}", file=sys.stderr)
            return 1

    req_fin_old = """    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)
"""
    req_fin_new = """    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        # DSV4-HMA: FA+SWA groups. Do not drop the SWA group.
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(
            request, [list(g) for g in block_ids]
        )
"""
    try:
        src = _replace_once(src, req_fin_old, req_fin_new, "request_finished_all_groups")
        applied.append("all_groups")
    except ValueError:
        m = re.search(
            r"(assert self\.connector_scheduler is not None\n"
            r"[ \t]+return self\.connector_scheduler\.request_finished\("
            r"request, block_ids\)\n)",
            src,
        )
        if not m:
            print("[dsv4-hma] ERROR: request_finished_all_groups anchor missing.", file=sys.stderr)
            return 1
        insert = (
            m.group(1)
            + "\n    def request_finished_all_groups(\n"
            "        self,\n"
            '        request: "Request",\n'
            "        block_ids: tuple[list[int], ...],\n"
            "    ) -> tuple[bool, dict[str, Any] | None]:\n"
            "        # DSV4-HMA: FA+SWA groups. Do not drop the SWA group.\n"
            "        assert self.connector_scheduler is not None\n"
            "        return self.connector_scheduler.request_finished(\n"
            "            request, [list(g) for g in block_ids]\n"
            "        )\n"
        )
        src = src[: m.start()] + insert + src[m.end() :]
        applied.append("all_groups-re")

    try:
        src = _replace_once(
            src,
            "    def __init__(self, vllm_config: VllmConfig, engine_id: str):\n"
            "        self.vllm_config = vllm_config\n",
            "    def __init__(\n"
            "        self,\n"
            "        vllm_config: VllmConfig,\n"
            "        engine_id: str,\n"
            '        kv_cache_config: "KVCacheConfig | None" = None,\n'
            "    ):\n"
            "        self.vllm_config = vllm_config\n"
            "        self.kv_cache_config = kv_cache_config\n",
            "scheduler __init__",
        )
        applied.append("sched-init")
    except ValueError:
        m = re.search(
            r"class MoRIIOConnectorScheduler:.*?"
            r"def __init__\(self, vllm_config: VllmConfig, engine_id: str\):\n"
            r"([ \t]+)self\.vllm_config = vllm_config\n",
            src,
            re.S,
        )
        if not m:
            print("[dsv4-hma] ERROR: scheduler __init__ anchor missing.", file=sys.stderr)
            return 1
        src = (
            src[: m.start()]
            + src[m.start() : m.end()].replace(
                "def __init__(self, vllm_config: VllmConfig, engine_id: str):",
                'def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig | None" = None):',
                1,
            ).replace(
                "self.vllm_config = vllm_config\n",
                "self.vllm_config = vllm_config\n"
                + m.group(1)
                + "self.kv_cache_config = kv_cache_config\n",
                1,
            )
            + src[m.end() :]
        )
        applied.append("sched-init-re")

    # Replace every get_block_ids()[0] in this file (WRITE notify + save + READ).
    old_g0 = "blocks.get_block_ids()[0]"
    new_g0 = "_dsv4_all_group_block_ids(blocks)"
    n_g0 = src.count(old_g0)
    if n_g0 < 2:
        print(
            f"[dsv4-hma] ERROR: expected >=2 get_block_ids()[0], found {n_g0}.",
            file=sys.stderr,
        )
        return 1
    src = src.replace(old_g0, new_g0)
    applied.append(f"get_block_ids x{n_g0}")

    # delay_free on nested groups (len(tuple)==n_groups is not "has blocks").
    n_delay = len(re.findall(r"delay_free_blocks = len\(computed_block_ids\) > 0", src))
    if n_delay != 1:
        print(
            f"[dsv4-hma] ERROR: delay_free_blocks assignment count={n_delay}.",
            file=sys.stderr,
        )
        return 1
    src = re.sub(
        r"delay_free_blocks = len\(computed_block_ids\) > 0",
        "delay_free_blocks = _dsv4_block_id_count(computed_block_ids) > 0  # DSV4-HMA",
        src,
        count=1,
    )
    applied.append("delay_free")

    if src.count("self.layer_to_spec = build_layer_to_spec(kv_cache_config)") != 1:
        print("[dsv4-hma] ERROR: worker layer_to_spec assignment missing/duplicate.", file=sys.stderr)
        return 1
    src = src.replace(
        "self.layer_to_spec = build_layer_to_spec(kv_cache_config)\n",
        "self.layer_to_spec = build_layer_to_spec(kv_cache_config)\n"
        "        # DSV4-HMA-SPECMAP: spec + layer_names, not FA=0/SWA=1.\n"
        "        self.kv_cache_config = kv_cache_config\n"
        "        self._dsv4_layer_to_group, self._dsv4_group_clip, "
        "self._dsv4_group_spec_bs = (\n"
        "            _dsv4_build_layer_to_group(kv_cache_config)\n"
        "        )  # DSV4-HMA-SPECMAP DSV4-HMA-GROUP-SPEC\n",
        1,
    )
    applied.append("layer_map")

    wm = re.search(
        r"def _write_blocks_for_req\(.*?(?=\n[ \t]*def |\nclass |\Z)",
        src,
        re.S,
    )
    if not wm:
        print("[dsv4-hma] ERROR: _write_blocks_for_req not found.", file=sys.stderr)
        return 1
    wbody = wm.group(0)
    if "local_block_ids=_local" in wbody:
        applied.append("write-pick (already)")
    else:
        if "local_block_ids=meta.local_block_ids" not in wbody:
            print(
                "[dsv4-hma] ERROR: _write_blocks_for_req missing "
                "local_block_ids=meta.local_block_ids.",
                file=sys.stderr,
            )
            return 1
        mcall = re.search(r"^([ \t]*)self\.schedule_write_blocks\(", wbody, re.M)
        if not mcall:
            print(
                "[dsv4-hma] ERROR: schedule_write_blocks not in "
                "_write_blocks_for_req.",
                file=sys.stderr,
            )
            return 1
        ind = mcall.group(1)
        pick = (
            f"{ind}_l2g = getattr(self, \"_dsv4_layer_to_group\", {{}}) or {{}}\n"
            f"{ind}_clip = getattr(self, \"_dsv4_group_clip\", None)\n"
            f"{ind}_local = _dsv4_pick_group_ids(\n"
            f"{ind}    layer_name, meta.local_block_ids, _l2g, _clip\n"
            f"{ind})\n"
            f"{ind}_remote = _dsv4_pick_group_ids(\n"
            f"{ind}    layer_name, meta.remote_block_ids, _l2g, _clip\n"
            f"{ind})\n"
            f"{ind}_gi = _dsv4_lookup_group(\n"
            f"{ind}    layer_name, _l2g,\n"
            f"{ind}    len(meta.local_block_ids) if meta.local_block_ids else 1,\n"
            f"{ind})\n"
            f"{ind}if not getattr(self, \"_dsv4_logged_pair\", None):\n"
            f"{ind}    self._dsv4_logged_pair = set()\n"
            f"{ind}if layer_name not in self._dsv4_logged_pair:\n"
            f"{ind}    self._dsv4_logged_pair.add(layer_name)\n"
            f"{ind}    _bs = None\n"
            f"{ind}    try:\n"
            f"{ind}        _bs = self._get_layer_transfer_geometry("
            f"layer_name).block_size\n"
            f"{ind}    except Exception:\n"
            f"{ind}        pass\n"
            f"{ind}    logger.info(\n"
            f"{ind}        \"[dsv4-hma] write layer=%s group=%s n_local=%d "
            f"n_remote=%d block_size=%s\",\n"
            f"{ind}        layer_name, _gi, len(_local), len(_remote), _bs,\n"
            f"{ind}    )\n"
            f"{ind}    if _gi is None:\n"
            f"{ind}        logger.error(\n"
            f"{ind}            \"[dsv4-hma] unmatched layer=%s; skip transfer\",\n"
            f"{ind}            layer_name,\n"
            f"{ind}        )\n"
            f"{ind}_local, _remote = _dsv4_align_write_ids("
            f"layer_name, _local, _remote)\n"
        )
        # Insert above the original call. Do not rewrite the call's indent
        # (217683: pick + extra 8 spaces on schedule_write_blocks -> IndentationError).
        wbody2 = wbody[: mcall.start()] + pick + wbody[mcall.start() :]
        wbody2 = wbody2.replace(
            "local_block_ids=meta.local_block_ids",
            "local_block_ids=_local",
            1,
        )
        wbody2 = wbody2.replace(
            "remote_block_ids=meta.remote_block_ids",
            "remote_block_ids=_remote",
            1,
        )
        src = src[: wm.start()] + wbody2 + src[wm.end() :]
        applied.append("write-pick")

    # Gate skipped .attn. HMA must transfer it with per-group ids.
    gate_all = """            # DSV4-HMA: transfer every registered cache, including block-64
            # .attn. Indexer is not registered. Per-group block ids + mixed-bs
            # offsets replace the 217546 skip.
            logger.info(
                "[dsv4-hma] wait_for_save transferring %d registered layers",
                len(self.kv_caches),
            )
            for layer_name, kv_layer in self.kv_caches.items():
                self.save_kv_layer(metadata, layer_name, kv_layer, None)
"""
    if "wait_for_save transferring %d registered layers" in src:
        applied.append("wait_for_save-all (already)")
    elif "wait_for_save skipped %d non-transfer layers" in src:
        src2, nsub = re.subn(
            r"[ \t]*_n_skip = 0\n"
            r".*?"
            r'\[dsv4-gate\] wait_for_save skipped %d non-transfer layers",\n'
            r"[ \t]*_n_skip,\n"
            r"[ \t]*\)\n",
            gate_all,
            src,
            count=1,
            flags=re.S,
        )
        if nsub != 1:
            print(
                "[dsv4-hma] ERROR: gate wait_for_save skip block failed to replace.",
                file=sys.stderr,
            )
            return 1
        src = src2
        applied.append("wait_for_save-all")
    elif re.search(
        r"for layer_name, kv_layer in self\.kv_caches\.items\(\):\s*"
        r"self\.save_kv_layer\(metadata, layer_name, kv_layer, None\)",
        src,
    ):
        # Vanilla dump-all already transfers every registered layer. Do not
        # inject a log (217683 first-match insert used bad indent).
        applied.append("wait_for_save-vanilla")
    else:
        print(
            "[dsv4-hma] ERROR: wait_for_save loop anchor missing.",
            file=sys.stderr,
        )
        return 1

    # Count every registered layer toward WRITE completion (incl. .attn).
    gate_count_mla = """        self.num_transfer_layers = sum(
            1
            for _ln in self.kv_caches
            if self._get_layer_transfer_geometry(_ln).block_size == self.block_size
        ) or self.num_layers
"""
    gate_count_idx = """        self.num_transfer_layers = sum(
            1
            for _ln in self.kv_caches
            if ".indexer." in _ln
            or (
                not _ln.endswith(".attn")
                and self._get_layer_transfer_geometry(_ln).block_size
                == self.block_size
            )
        ) or self.num_layers
"""
    hma_count = """        # DSV4-HMA: .attn (block 64) is a transfer layer.
        self.num_transfer_layers = self.num_layers
"""
    if gate_count_idx in src:
        src = src.replace(gate_count_idx, hma_count, 1)
        applied.append("num_transfer_layers")
    elif gate_count_mla in src:
        src = src.replace(gate_count_mla, hma_count, 1)
        applied.append("num_transfer_layers")
    elif "DSV4-HMA: .attn (block 64) is a transfer layer" in src:
        applied.append("num_transfer_layers (already)")

    try:
        src = _patch_offsets_pick(src)
        applied.append("offsets-pick")
        src = _patch_geom_skip(src)
        applied.append("geom-skip")
        src = _upgrade_group_spec(src)
        applied.append("groupspec")
        src2 = _upgrade_align_write(src)
        if src2 != src:
            applied.append("align-write-220216")
        src = src2
    except ValueError as e:
        print(f"[dsv4-hma] ERROR: {e}", file=sys.stderr)
        return 1

    if MARKER not in src:
        print("[dsv4-hma] ERROR: marker missing after patch.", file=sys.stderr)
        return 1
    if "class MoRIIOConnector(KVConnectorBase_V1, SupportsHMA)" not in src:
        print("[dsv4-hma] ERROR: class still without SupportsHMA.", file=sys.stderr)
        return 1
    if "request_finished_all_groups" not in src:
        print("[dsv4-hma] ERROR: request_finished_all_groups missing.", file=sys.stderr)
        return 1
    if "blocks.get_block_ids()[0]" in src:
        print(
            "[dsv4-hma] ERROR: get_block_ids()[0] still present "
            "(would drop the SWA group).",
            file=sys.stderr,
        )
        return 1

    if "DSV4-HMA: 217685" not in src:
        print("[dsv4-hma] ERROR: 217685 offsets pick missing after patch.", file=sys.stderr)
        return 1
    if "DSV4-HMA-SPECMAP" not in src:
        print("[dsv4-hma] ERROR: specmap missing after patch.", file=sys.stderr)
        return 1
    if "geom-skip layer=" not in src:
        print("[dsv4-hma] ERROR: geom-skip missing after patch.", file=sys.stderr)
        return 1
    if "DSV4-HMA-GROUP-SPEC: 217719" not in src:
        print("[dsv4-hma] ERROR: groupspec missing after patch.", file=sys.stderr)
        return 1
    if "def _dsv4_align_write_ids(" not in src:
        print("[dsv4-hma] ERROR: align-write missing after patch.", file=sys.stderr)
        return 1

    if src == orig:
        print("[dsv4-hma] ERROR: no changes written.", file=sys.stderr)
        return 1

    return _write_connector(path, src, applied)


def _selftest() -> int:
    """Specmap: exact group, no FA=0 guess, SWA clip, nested ids flatten."""
    import ast
    from types import SimpleNamespace

    class _Log:
        def info(self, *a, **k):
            return None

        def debug(self, *a, **k):
            return None

        def error(self, *a, **k):
            return None

        def warning(self, *a, **k):
            return None

    ns: dict = {"logger": _Log()}
    exec(HELPERS, ns)  # noqa: S102
    pick = ns["_dsv4_pick_group_ids"]
    build = ns["_dsv4_build_layer_to_group"]

    class _Spec:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _Uni(_Spec):
        pass

    groups = [
        SimpleNamespace(
            layer_names=["model.layers.2.attn"],
            kv_cache_spec=_Spec(block_size=256),
        ),
        SimpleNamespace(
            layer_names=["model.layers.3.attn"],
            kv_cache_spec=_Spec(block_size=64, sliding_window=128),
        ),
        SimpleNamespace(
            layer_names=["model.layers.2.attn.compressor.state_cache"],
            kv_cache_spec=_Spec(block_size=4, sliding_window=4),
        ),
        SimpleNamespace(
            layer_names=["model.layers.0.attn.swa_cache"],
            kv_cache_spec=_Spec(block_size=64, sliding_window=128),
        ),
        SimpleNamespace(
            layer_names=["model.layers.2.attn.indexer.k_cache"],
            kv_cache_spec=_Uni(
                block_size=64,
                kv_cache_specs={
                    "model.layers.2.attn.indexer.k_cache": _Spec(block_size=64),
                    "model.layers.4.attn.indexer.k_cache": _Spec(block_size=64),
                },
            ),
        ),
    ]
    l2g, clip, gbs = build(SimpleNamespace(kv_cache_groups=groups))
    assert l2g["model.layers.2.attn"] == 0
    assert l2g["model.layers.3.attn"] == 1
    assert l2g["model.layers.2.attn.compressor.state_cache"] == 2
    assert l2g["model.layers.0.attn.swa_cache"] == 3
    assert l2g["model.layers.4.attn.indexer.k_cache"] == 4
    assert clip[1] == 3  # cdiv(128,64)+1
    assert gbs[0] == 256
    assert gbs[1] == 64
    uni = _Uni(
        block_size=256,
        kv_cache_specs={
            "model.layers.2.attn": _Spec(block_size=64),
            "model.layers.3.attn": _Spec(block_size=2),
            "model.layers.2.attn.indexer.k_cache": _Spec(block_size=256),
        },
    )
    assert ns["_dsv4_group_page"](uni) == 256
    assert ns["_dsv4_group_spec_page"]("model.layers.2.attn", l2g, gbs) == 256
    nested = [[1], [0, 0, 6, 7, 8], [3], [9, 10, 11], [12]]
    assert pick("model.layers.2.attn", nested, l2g, clip) == [1]
    assert pick("model.layers.3.attn", nested, l2g, clip) == [6, 7, 8]
    assert pick("model.layers.0.attn.swa_cache", nested, l2g, clip) == [9, 10, 11]
    assert pick(
        "model.layers.2.attn.compressor.state_cache", nested, l2g, clip
    ) == [3]
    # Unmatched must not steal group 0 (217691 heuristic).
    assert (
        pick(
            "model.layers.0.attn.swa_cache",
            nested,
            {"model.layers.0.attn": 1},
            None,
        )
        == []
    )
    assert all(
        isinstance(x, int)
        for x in pick("model.layers.2.attn", nested, l2g, clip)
    )
    # 218699: drop prefix 0 before clip. Decode [0]+16 real, clip 17 -> 16.
    clip17 = [0, 0, 17]
    decode_g2 = [[], [], [0] + list(range(10, 26))]
    assert pick(
        "model.layers.2.attn.compressor.state_cache",
        decode_g2,
        {"model.layers.2.attn.compressor.state_cache": 2},
        clip17,
    ) == list(range(10, 26))
    # 218768: group-0 clip=0. Block 0 is a real page (curl dest [0]).
    clip0 = [0, 3, 3]
    assert pick(
        "model.layers.0.attn",
        [[0], [9, 10, 11], [3]],
        {"model.layers.0.attn": 0},
        clip0,
    ) == [0]
    assert pick(
        "model.layers.0.attn",
        [[0, 1, 2], [9, 10, 11], [3]],
        {"model.layers.0.attn": 0},
        clip0,
    ) == [0, 1, 2]
    align = ns["_dsv4_align_write_ids"]
    a, b = align("x", list(range(17)), list(range(16)))
    assert a == list(range(1, 17)) and b == list(range(16))
    a, b = align("x", list(range(17)), list(range(17)))
    assert a == list(range(17)) and b == list(range(17))
    # 220216: empty remote is the WRITE hint, not a skip.
    a, b = align("x", [1, 2, 3], [])
    assert a == [1, 2, 3] and b == []
    a, b = align("model.layers.0.attn", [0], [])
    assert a == [0] and b == []
    a, b = align("x", [], [1])
    assert a == [] and b == [1]
    src = (
        "        return compute_block_transfer_offsets(\n"
        "            layer_name=layer_name,\n"
        "            kv_cache=self.kv_caches[layer_name],\n"
        "            layer_to_spec=self.layer_to_spec,\n"
        "            local_block_ids=local_block_ids,\n"
        "            remote_block_ids=remote_block_ids,\n"
        "        )\n"
    )
    out = _patch_offsets_pick(src)
    assert "DSV4-HMA: 217685" in out
    assert "_dsv4_pick_group_ids(layer_name, remote_block_ids, _l2g, _clip)" in out
    assert "local_block_ids, remote_block_ids = _dsv4_align_write_ids(" in out
    assert _patch_offsets_pick(out) == out
    era_218699 = (
        OLD_PICK_AFTER_INT
        + "        logger.debug(\n"
        + '            "[dsv4-hma] offsets layer=%s local=%d remote=%d",\n'
        + "            layer_name, len(local_block_ids), len(remote_block_ids),\n"
        + "        )\n"
        + '            logger.error(\n'
        + '                "[dsv4-hma] unmatched layer=%s; skip transfer",\n'
        + "                layer_name,\n"
        + "            )\n"
        + "            self.schedule_write_blocks(\n"
    )
    era_up = _upgrade_align_write(era_218699)
    assert "def _dsv4_align_write_ids(" in era_up
    assert "local_block_ids, remote_block_ids = _dsv4_align_write_ids(" in era_up
    assert "_dsv4_align_write_ids(layer_name, _local, _remote)" in era_up
    assert NEWER_PICK_AFTER_INT in era_up
    assert NEW_PICK_AFTER_INT not in era_up
    assert OLD_PICK_AFTER_INT not in era_up
    assert "clip-tail hint" in era_up
    assert "clip-tail skip" not in era_up
    assert ALIGN_EMPTY_HINT in HELPERS
    assert ALIGN_EMPTY_SKIP not in HELPERS
    assert _upgrade_align_write(era_up) == era_up
    era_skip = era_up.replace(ALIGN_EMPTY_HINT, ALIGN_EMPTY_SKIP, 1)
    assert ALIGN_EMPTY_SKIP in era_skip
    era_fixed = _upgrade_align_write(era_skip)
    assert ALIGN_EMPTY_HINT in era_fixed
    assert ALIGN_EMPTY_SKIP not in era_fixed
    assert _upgrade_align_write(era_fixed) == era_fixed
    assert "DSV4-HMA-ALIGN-TAIL" in _patch_layout_align(LAYOUT_LONGER_RAISE)
    assert _patch_layout_align(LAYOUT_LONGER_TAIL) == LAYOUT_LONGER_TAIL
    old_pick = (
        "def _dsv4_block_id_count(block_ids) -> int:\n"
        "    return len(block_ids)\n"
        "def _dsv4_lookup_group(layer_name: str, layer_to_group: dict, n_groups: int) -> int:\n"
        "    return 0\n"
        "def _dsv4_pick_group_ids(layer_name: str, block_ids, "
        "layer_to_group: dict) -> list:\n"
        '    """Select this layer\'s KV-group block ids (nested) or the flat list."""\n'
        "    if not block_ids:\n"
        "        return []\n"
        "    if not isinstance(block_ids[0], (list, tuple)):\n"
        "        return list(block_ids)\n"
        "    gi = layer_to_group.get(layer_name)\n"
        "    if gi is None:\n"
        "        gi = 0\n"
        "    return list(block_ids[gi] or [])\n"
        "        self._dsv4_layer_to_group = {}\n"
        "        for _gi, _g in enumerate(\n"
        '            getattr(kv_cache_config, "kv_cache_groups", None) or []\n'
        "        ):\n"
        "            for _ln in getattr(_g, \"layer_names\", None) or []:\n"
        "                self._dsv4_layer_to_group[_ln] = _gi\n"
        "        logger.info(\n"
        '            "[dsv4-hma] worker kv_groups=%d layer_map=%d",\n'
        "            len(getattr(kv_cache_config, \"kv_cache_groups\", None) or []),\n"
        "            len(self._dsv4_layer_to_group),\n"
        "        )\n"
        '        _l2g = getattr(self, "_dsv4_layer_to_group", {}) or {}\n'
        "            local_block_ids = _dsv4_pick_group_ids("
        "layer_name, local_block_ids, _l2g)\n"
    )
    upgraded = _upgrade_specmap(old_pick)
    assert "DSV4-HMA-SPECMAP" in upgraded
    assert "def _dsv4_build_layer_to_group(" in upgraded
    assert "_l2g, _clip)" in upgraded
    assert _fold_mla_3d(1024, 64, 256) == (256, 256, 4)
    assert _fold_mla_3d(1024, 2, 256) == (8, 256, 128)
    assert _fold_mla_3d(101, 64, 256) == (25, 256, 4)
    assert _fold_mla_3d(3, 64, 256) == (3, 64, 1)
    assert _fold_mla_3d(64, 64, 64) == (64, 64, 1)
    geom = _patch_geom_skip(out)
    assert "geom-skip layer=" in geom
    assert "_dsv4_group_spec_page(" in geom
    assert _patch_geom_skip(geom) == geom
    geom_fn = (
        "    def _get_layer_transfer_geometry(\n"
        "        self, layer_name: str, remote_num_blocks: int | None = None\n"
        "    ) -> LayerTransferGeometry:\n"
        "        return get_layer_transfer_geometry(\n"
        "            layer_name,\n"
        "            self.kv_caches[layer_name],\n"
        "            self.layer_to_spec,\n"
        "            remote_num_blocks,\n"
        "        )\n"
    )
    stub_out = _patch_geometry_override(geom_fn)
    assert "DSV4-HMA-GROUP-SPEC: 217719" in stub_out
    assert "fold-ok layer=" in stub_out
    assert "num_blocks=%d shape=%s" in stub_out
    assert _patch_geometry_override(stub_out) == stub_out
    off = (
        "            layer_to_spec=self.layer_to_spec,\n"
        "            local_block_ids=local_block_ids,\n"
    )
    assert "spec_bs_override=_dsv4_group_spec_page(" in _patch_offsets_spec_override(
        off
    )
    assert LAYOUT_MARKER in LAYOUT_MLA_3D_NEW
    assert LAYOUT_MLA_3D_OLD not in LAYOUT_MLA_3D_NEW
    assert "spec_bs_override" in LAYOUT_MLA_3D_NEW
    assert "DSV4-HMA-GEOM-FLOOR" in LAYOUT_MLA_3D_NEW
    ast.parse("def _geom():\n" + LAYOUT_MLA_3D_NEW)
    src_floor = LAYOUT_FLOOR_FROM_DIVISIBLE_OLD.replace(
        LAYOUT_FLOOR_FROM_DIVISIBLE_OLD, LAYOUT_FLOOR_FROM_DIVISIBLE_NEW, 1
    )
    assert "DSV4-HMA-GEOM-FLOOR" in src_floor
    assert "num_blocks % (spec_bs // block_size)" not in src_floor
    assert "spec_bs_override" in LAYOUT_GEOM_SIG_NEW
    ast.parse("def _align():\n" + LAYOUT_LONGER_TAIL)
    print("[dsv4-hma] selftest OK")
    return 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        return _selftest()
    if len(sys.argv) != 2:
        print(
            f"usage: {sys.argv[0]} <vllm_install_dir> | --selftest",
            file=sys.stderr,
        )
        return 2
    vllm_dir = sys.argv[1]
    path = os.path.join(vllm_dir, CONN_REL)
    if not os.path.isfile(path):
        print(f"[dsv4-hma] {CONN_REL} not found under {vllm_dir} -- skipping.")
        return 0
    rc = patch_connector(path)
    if rc != 0:
        return rc
    layout_path = os.path.join(vllm_dir, LAYOUT_REL)
    return patch_layout(layout_path)


if __name__ == "__main__":
    sys.exit(main())
