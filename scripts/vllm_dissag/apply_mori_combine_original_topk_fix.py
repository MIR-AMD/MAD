#!/usr/bin/env python3
"""Fix MoRI combine() using dispatched topk_ids (vLLM EP32 garbage).

Source: amd-weisun/vllm @ fix/mori-combine-original-topk-ids
  commit e25bf1826f9a8dc66d7312f046375e8c9491d638 (2026-08-17)

MoriPrepareAndFinalize.finalize() was calling mori_op.combine() with the
topk_ids modular_kernel already replaced with DISPATCHED (post-prepare) ids.
Those ids describe OTHER ranks' tokens that landed on this rank's local
experts, not the original routing of THIS rank's tokens. combine() then
gathers from the wrong nodes. Worse at EP32 (4 nodes) than EP16 (2 nodes).

Qwen3-30B-A3B (Li / Wei Sun, OCI): EP32 probe 3/10 and GSM8K 18.95% before;
10/10 and 89.31% after. EP8/EP16 were already fine. Hunyuan needs more than
this patch; GLM 4P/2D 216534 rank-0 Who isWho matches the EP32 signature.

Matches acb0f1dc mori.py. Idempotent. Missing file -> skip. Found-old that
fails to apply is a hard error.

Usage: apply_mori_combine_original_topk_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "model_executor/layers/fused_moe/prepare_finalize/mori.py"

OLD_INIT = """        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
"""
NEW_INIT = """        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
        # Original (pre-dispatch) topk_ids, stashed in prepare() for use in
        # finalize() -- see the comment in prepare() for why.
        self._original_topk_ids: torch.Tensor | None = None
"""

OLD_PREPARE = """        assert not apply_router_weight_on_input, (
            "mori does not support apply_router_weight_on_input=True now."
        )
        scale = None
"""
NEW_PREPARE = """        assert not apply_router_weight_on_input, (
            "mori does not support apply_router_weight_on_input=True now."
        )
        # combine() needs the ORIGINAL (pre-dispatch) topk_ids, not the
        # dispatched ids that modular_kernel.py's forward() substitutes in
        # for the expert GEMM stage. MoRI's combine kernel uses these ids
        # (via tokenIndices) to decide which remote nodes' partial results
        # to gather for each of THIS rank's own tokens -- the dispatched
        # ids describe OTHER ranks' tokens that were routed here, not the
        # original routing of this rank's own tokens, so passing them to
        # combine() causes it to gather from the wrong nodes. Stash the
        # original ids here; finalize() below uses this instead of the
        # topk_ids it's given (which is also the post-dispatch value, for
        # the same reason).
        self._original_topk_ids = topk_ids
        scale = None
"""

OLD_COMBINE = """        result = self.mori_op.combine(
            fused_expert_output,
            None,
            topk_ids,
        )[0]
"""
NEW_COMBINE = """        result = self.mori_op.combine(
            fused_expert_output,
            None,
            self._original_topk_ids,
        )[0]
"""


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[mori-combine] {REL} not found under {sys.argv[1]} -- skipping.")
        return 0

    src = open(path).read()
    if "self._original_topk_ids" in src and "self._original_topk_ids," in src:
        print(f"[mori-combine] already patched in {path} -- no-op.")
        return 0

    if OLD_COMBINE not in src:
        print(
            f"[mori-combine] ERROR: combine(topk_ids) anchor missing in {path}.",
            file=sys.stderr,
        )
        return 1
    if OLD_PREPARE not in src:
        print(
            f"[mori-combine] ERROR: prepare() stash anchor missing in {path}.",
            file=sys.stderr,
        )
        return 1
    if OLD_INIT not in src:
        print(
            f"[mori-combine] ERROR: __init__ anchor missing in {path}.",
            file=sys.stderr,
        )
        return 1

    src = src.replace(OLD_INIT, NEW_INIT, 1)
    src = src.replace(OLD_PREPARE, NEW_PREPARE, 1)
    src = src.replace(OLD_COMBINE, NEW_COMBINE, 1)
    if "self._original_topk_ids," not in src:
        print(f"[mori-combine] ERROR: post-write combine() still uses topk_ids in {path}.",
              file=sys.stderr)
        return 1
    open(path, "w").write(src)
    print(f"[mori-combine] patched: combine() uses original topk_ids (Wei Sun e25bf182) in {path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
