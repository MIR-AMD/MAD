#!/usr/bin/env python3
"""Log the shape MoRI hands the experts, once per process. Default OFF.

436392 attributed the MoRI decode regression: of 301.81 ms of GPU kernel time
per decode step (against a 302.69 ms measured ITL, i.e. all of it),

    _matmul_ogs_NNT_bf16xbf16xmxfp4   240.28 ms/step   79.6%
    activation-shaped ops              35.39 ms/step   11.7%
    MoRI dispatch + combine             1.86 ms/step    0.6%

so MoRI's comm is not the cost -- the MoE expert stack is. The mechanism is
believed to be that `prepare()` returns MoRI's dispatch output, a view of
`(max_num_tokens_to_recv(), hidden_dim)` = `world_size * max_num_inp_token_per_rank`
= 8192 x 4096 at mnbt 1024 / EP8, which modular_kernel then uses as `M` for the
whole expert stack. One real token, 8192 rows of work.

That last step is an INFERENCE from the mnbt correlation (ITL = 20.1 +
0.2816*mnbt, R^2 0.996) plus the kernel split. This patch measures it instead,
because the fix is expensive enough to be worth one cheap confirmation.

`--profiler-config.torch_profiler_record_shapes=true` does NOT answer this:
record_shapes annotates CPU op events, and matmul_ogs is launched from Python
with no aten op carrying the activation shape. So log it at the source.

Fires once per process, on the first prepare() after arming. It reads
`total_recv`, a device tensor, so it forces one host sync -- once, not per
layer. Prints:

    a1          what vLLM handed MoRI            (real tokens this rank)
    dispatch    what MoRI handed back            (the padded M)
    total_recv  rows actually carrying a token
    waste       dispatch rows per useful row

Expected at mnbt 1024 / EP8 / con=1 if the inference is right: a1 ~1 row,
dispatch 8192 rows. If dispatch comes back small, the padding theory is wrong
and the expert cost is coming from somewhere else -- stop and re-measure.

    MORI_SHAPE_PROBE   set to 1 to arm. Unset (default) costs one `if` on the
                       first prepare() call and nothing thereafter.

Anchors in prepare(), which neither apply_mori_combine_original_topk_fix.py
(__init__, finalize) nor apply_mori_profiler_dump.py (import, __init__,
finalize) touches, so this composes with both in any order. `os` is imported
function-locally for the same reason.

Idempotent. Missing file -> skip. Found-old that fails to apply is a hard
error.

Usage: apply_mori_shape_probe.py <vllm_install_dir>
"""
import os
import sys

REL = "model_executor/layers/fused_moe/prepare_finalize/mori.py"

MARKER = "_mori_shape_probe"

OLD = """        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=dispatch_recv_token_num, expert_num_tokens_cpu=None
        )
"""
NEW = '''        if not MoriPrepareAndFinalize._mori_shape_probe_done:
            self._mori_shape_probe(a1, dispatch_a1, dispatch_recv_token_num)

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=dispatch_recv_token_num, expert_num_tokens_cpu=None
        )
'''

# Class attribute, not instance: there is one MoriPrepareAndFinalize per MoE
# layer (43 on Flash), and one line per layer would bury the point.
OLD_FLAG = """    def __init__(
        self,
        mori_op: mori.ops.EpDispatchCombineOp,
"""
NEW_FLAG = """    _mori_shape_probe_done = False

    def __init__(
        self,
        mori_op: mori.ops.EpDispatchCombineOp,
"""

OLD_TAIL = """    def finalize(
        self,
        output: torch.Tensor,
"""
NEW_TAIL = '''    def _mori_shape_probe(self, a1, dispatch_a1, dispatch_recv_token_num) -> None:
        """Report real-tokens-in vs padded-rows-out, once per process."""
        import os

        # Disarm first, and for the whole class: if anything below raises we
        # want one logged failure, not one per layer per token.
        MoriPrepareAndFinalize._mori_shape_probe_done = True
        if os.environ.get("MORI_SHAPE_PROBE", "") != "1":
            return
        try:
            cfg = getattr(self.mori_op, "config", None)
            cap = getattr(cfg, "max_num_inp_token_per_rank", -1)
            world = getattr(cfg, "world_size", -1)
            rows = int(dispatch_a1.shape[0])
            # Device tensor -> one host sync, once per process.
            try:
                recv = int(dispatch_recv_token_num.sum().item())
            except Exception:
                recv = -1
            logger.warning(
                "[mori-shape] a1=%s -> dispatch=%s | total_recv=%d | "
                "capacity=%d world_size=%d capacity*world_size=%d | "
                "waste=%s rows per useful row",
                tuple(a1.shape),
                tuple(dispatch_a1.shape),
                recv,
                cap,
                world,
                cap * world if cap > 0 and world > 0 else -1,
                f"{rows / recv:.0f}" if recv > 0 else "n/a",
            )
        except Exception:
            logger.exception("[mori-shape] probe failed; continuing")

    def finalize(
        self,
        output: torch.Tensor,
'''


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[mori-shape] {REL} not found under {sys.argv[1]} -- skipping.")
        return 0

    src = open(path).read()
    if MARKER in src:
        print(f"[mori-shape] already patched in {path} -- no-op.")
        return 0

    for name, anchor in (
        ("probe-site", OLD),
        ("flag", OLD_FLAG),
        ("method", OLD_TAIL),
    ):
        n = src.count(anchor)
        if n != 1:
            print(
                f"[mori-shape] ERROR: {name} anchor found {n} times in {path}, "
                f"expected 1.",
                file=sys.stderr,
            )
            return 1

    src = src.replace(OLD_FLAG, NEW_FLAG, 1)
    src = src.replace(OLD, NEW, 1)
    src = src.replace(OLD_TAIL, NEW_TAIL, 1)

    if src.count(MARKER) < 3:
        print(
            f"[mori-shape] ERROR: post-write marker count is "
            f"{src.count(MARKER)}, expected >= 3, in {path}.",
            file=sys.stderr,
        )
        return 1

    import ast

    try:
        ast.parse(src)
    except SyntaxError as exc:
        print(f"[mori-shape] ERROR: patched file does not parse: {exc}",
              file=sys.stderr)
        return 1

    open(path, "w").write(src)
    print(f"[mori-shape] patched: dispatch shape probe wired into {path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
