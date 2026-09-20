#!/usr/bin/env python3
"""Hand the experts the rows that can hold a token, not the whole buffer. OFF by default.

THE BUG (436457, measured end to end -- not inferred)

MoRI's `dispatch()` returns a `_cached_view` over a preallocated symmetric
buffer, shaped `(max_num_tokens_to_recv(), hidden_dim)`, where

    max_num_tokens_to_recv() = world_size * max_num_inp_token_per_rank

and vLLM wires `max_num_inp_token_per_rank = moe.max_num_tokens =
--max-num-batched-tokens`. That shape does not depend on how many tokens
exist. The shape probe caught it in both cells of 436457:

    mnbt=1024   a1=(1024, 4096) -> dispatch=(8192, 4096)   capacity=1024 ws=8
    mnbt= 256   a1=( 256, 4096) -> dispatch=(2048, 4096)   capacity= 256 ws=8

`modular_kernel._do_fused_experts` then takes `M_full` straight off that
tensor (`moe_problem_size` returns `M = a1.size(0)`) and sizes the entire
expert stack from it. So at con=1 decode, one real token per rank buys 8192
rows of MXFP4 expert GEMM:

    _matmul_ogs_NNT_bf16xbf16xmxfp4   1950.06 ms   79.8% of GPU kernel time
    activation-shaped ops (silu/reduce/gather/topk_sum)   ~11.7%
    MoRI dispatch + combine             15.29 ms    0.6%

Cutting mnbt 4x cut M 4x (8192 -> 2048) and the GEMM followed at 4.82x, with
the launch count *identical* at 688 and triton re-autotuning its M-tile from
128 to 64. MoRI's comm stayed flat. The padded buffer is the regression; MoRI
is not.

THE FIX

Each source rank can send this rank at most its own token count -- that is
precisely why MoRI sizes recv-per-rank at `max_num_inp_token_per_rank`. So

    bound = world_size * max_tokens_on_any_rank_this_forward

is a hard upper bound on the rows that can carry a token, it is known on the
HOST before the forward, and it is identical on every rank. Trim the dispatch
outputs to it and `M_full` becomes `bound`.

Decode at con=1: `bound = 8 * 1` = 8 rows instead of 8192.
Prefill/profile_run at 1024 tokens/rank: `bound = 8 * 1024` = 8192 = the full
buffer, so those forwards are byte-identical and untouched. That is the whole
point -- the padding is CORRECT at prefill scale (436457 measured `waste=1`,
zero waste, on three of eight ranks during profile_run), and it is only decode
that was paying prefill's bill.

WHY NOT THE OTHER TWO CANDIDATES

`max_num_inp_token_per_rank` (a global capacity clamp) is what 436392 tried and
it is UNRUNNABLE: `determine_available_memory` -> `profile_run` pushes a full
mnbt-sized batch through the MoE at boot, so capacity must be >= mnbt before
any traffic exists. Both capped cells came back NOT_READY.

MoRI's own `max_total_recv_tokens` config field (default 0 = disabled, and vLLM
never sets it) caps the same product one level down, but it is equally static
and "if the actual received token count exceeds the derived limit, the kernel
currently asserts" -- so it hits the identical boot wall. Neither knob can
express "big for prefill, small for decode"; a per-forward slice can, and it
costs no extra buffers.

SAFETY

- The bound is an upper bound, never an estimate. Slicing keeps a prefix, and
  received tokens are packed contiguously into `[0, total_recv)` -- MoRI's own
  `launch_local_expert_count` walks the recv indices with `total_ptr` as the
  extent, so there is nothing live past it.
- `combine()` reads `fused_expert_output` through that same recv mapping, so it
  stays in range for any `bound >= total_recv`.
- `modular_kernel` asserts `topk_ids.size(0) == a1.size(0)`, so ids, weights
  and scales are trimmed to the same length as the activations.
- Rank-consistent by construction: the bound comes from
  `DPMetadata.num_tokens_across_dp_cpu`, a CPU tensor holding every DP rank's
  count, reduced with `.max()`. Dispatch is collective; a per-rank bound would
  desynchronize it. If that metadata is absent we do NOT guess -- we skip the
  trim and say so once.
- No host sync on the hot path. The bound is host-side integer arithmetic, so
  cudagraph capture is unaffected (each captured batch size gets its own
  constant bound).

    MORI_TRIM_DISPATCH   set to 1 to arm. Unset (default) is byte-identical to
                         stock: one dict lookup per prepare() and no slicing.
    MORI_TRIM_CHECK      verify `total_recv <= bound` on the first N prepare()
                         calls. Costs one host sync per checked call, so this is
                         a tripwire for bringing the fix up, not a default. On
                         violation it logs ERROR and SKIPS the trim for that
                         call, so a wrong bound degrades to stock behaviour
                         instead of a wrong answer. Skipped while a cudagraph is
                         capturing, where syncing is illegal.

Anchors the dispatch call site and `def prepare(`, neither of which
apply_mori_combine_original_topk_fix.py touches (__init__, the
apply_router_weight_on_input assert, the combine call), so the two compose in
either order against the same file. `os` and the forward-context import are
function-local for the same reason.

Idempotent. Missing file -> skip. Found-old that fails to apply is a hard
error.

Usage: apply_mori_trim_dispatch.py <vllm_install_dir>
"""
import os
import sys

REL = "model_executor/layers/fused_moe/prepare_finalize/mori.py"

MARKER = "_mori_trim_rows"

# The dispatch call site. One occurrence, and no other patcher rewrites it.
OLD_DISPATCH = """        ) = self.mori_op.dispatch(a1, topk_weights, scale, topk_ids)
"""
NEW_DISPATCH = """        ) = self.mori_op.dispatch(a1, topk_weights, scale, topk_ids)

        # MoRI hands back its whole preallocated recv buffer
        # (world_size * max_num_inp_token_per_rank rows). Keep only the rows
        # that can hold a token this forward -- see
        # apply_mori_trim_dispatch.py.
        _trim = self._mori_trim_rows(a1, dispatch_a1, dispatch_recv_token_num)
        if _trim is not None:
            dispatch_a1 = dispatch_a1[:_trim]
            dispatch_ids = dispatch_ids[:_trim]
            if dispatch_weights is not None:
                dispatch_weights = dispatch_weights[:_trim]
            if dispatch_scale is not None:
                dispatch_scale = dispatch_scale[:_trim]
"""

OLD_PREPARE_DEF = """    def prepare(
        self,
        a1: torch.Tensor,
"""
NEW_PREPARE_DEF = '''    _mori_trim_checked = 0
    _mori_trim_warned = False

    def _mori_trim_rows(self, a1, dispatch_a1, dispatch_recv_token_num):
        """Rows of MoRI's dispatch buffer that can carry a token, or None.

        None means "leave the buffer alone", which is stock behaviour.
        """
        import os

        if os.environ.get("MORI_TRIM_DISPATCH", "") != "1":
            return None

        rows = dispatch_a1.shape[0]
        cfg = getattr(self.mori_op, "config", None)
        world = int(getattr(cfg, "world_size", 0) or 0)
        if world <= 0:
            return None

        per_rank = self._mori_tokens_per_rank()
        if per_rank <= 0:
            return None

        bound = world * per_rank
        # >= rows is the prefill/profile_run case: the buffer is already the
        # right size, so slicing would be a no-op view. Return None so those
        # forwards stay byte-identical to stock.
        if bound >= rows:
            return None

        try:
            limit = int(os.environ.get("MORI_TRIM_CHECK", "") or 0)
        except ValueError:
            limit = 0
        if 0 < limit and MoriPrepareAndFinalize._mori_trim_checked < limit:
            if not self._mori_trim_bound_holds(
                dispatch_recv_token_num, bound, rows, world, per_rank
            ):
                return None

        return bound

    def _mori_tokens_per_rank(self) -> int:
        """Max tokens on ANY rank this forward, or 0 if we cannot know it.

        Must be identical on every rank -- dispatch is collective. Returns 0
        (do not trim) rather than a local guess, because a bound that differed
        per rank would desynchronize the collective.
        """
        try:
            from vllm.forward_context import get_forward_context

            dp_meta = getattr(get_forward_context(), "dp_metadata", None)
            counts = getattr(dp_meta, "num_tokens_across_dp_cpu", None)
            if counts is not None:
                return int(counts.max())
        except Exception:
            pass
        if not MoriPrepareAndFinalize._mori_trim_warned:
            MoriPrepareAndFinalize._mori_trim_warned = True
            logger.warning(
                "[mori-trim] MORI_TRIM_DISPATCH=1 but "
                "DPMetadata.num_tokens_across_dp_cpu is unavailable, so the "
                "per-rank token bound cannot be made rank-consistent. "
                "Leaving the dispatch buffer untrimmed (stock behaviour)."
            )
        return 0

    def _mori_trim_bound_holds(
        self, dispatch_recv_token_num, bound, rows, world, per_rank
    ) -> bool:
        """Tripwire: does the received token count really fit in `bound`?

        Forces a host sync, so it runs for the first MORI_TRIM_CHECK calls only
        and never under cudagraph capture.
        """
        import os

        import torch

        try:
            if torch.cuda.is_current_stream_capturing():
                return True
        except Exception:
            pass

        MoriPrepareAndFinalize._mori_trim_checked += 1
        try:
            recv = int(dispatch_recv_token_num.sum().item())
        except Exception:
            logger.exception("[mori-trim] could not read total_recv; not trimming")
            return False

        if recv > bound:
            # Fail safe, not loud: skipping the trim gives the stock (correct,
            # slow) answer, where trimming past total_recv would silently drop
            # tokens.
            logger.error(
                "[mori-trim] BOUND VIOLATED: total_recv=%d > bound=%d "
                "(world_size=%d * tokens_per_rank=%d). Not trimming. The "
                "one-copy-per-destination-rank assumption does not hold here "
                "-- do not ship this fix until that is understood.",
                recv,
                bound,
                world,
                per_rank,
            )
            return False

        logger.warning(
            "[mori-trim] check %d/%s ok: total_recv=%d <= bound=%d, "
            "M %d -> %d (%.0fx less expert work)",
            MoriPrepareAndFinalize._mori_trim_checked,
            os.environ.get("MORI_TRIM_CHECK", "?"),
            recv,
            bound,
            rows,
            bound,
            rows / bound if bound else 0.0,
        )
        return True

    def prepare(
        self,
        a1: torch.Tensor,
'''


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[mori-trim] {REL} not found under {sys.argv[1]} -- skipping.")
        return 0

    src = open(path).read()
    if MARKER in src:
        print(f"[mori-trim] already patched in {path} -- no-op.")
        return 0

    for name, anchor in (
        ("dispatch-site", OLD_DISPATCH),
        ("prepare-def", OLD_PREPARE_DEF),
    ):
        n = src.count(anchor)
        if n != 1:
            print(
                f"[mori-trim] ERROR: {name} anchor found {n} times in {path}, "
                f"expected 1.",
                file=sys.stderr,
            )
            return 1

    src = src.replace(OLD_PREPARE_DEF, NEW_PREPARE_DEF, 1)
    src = src.replace(OLD_DISPATCH, NEW_DISPATCH, 1)

    if src.count(MARKER) < 2:
        print(
            f"[mori-trim] ERROR: post-write marker count is "
            f"{src.count(MARKER)}, expected >= 2, in {path}.",
            file=sys.stderr,
        )
        return 1

    import ast

    try:
        ast.parse(src)
    except SyntaxError as exc:
        print(f"[mori-trim] ERROR: patched file does not parse: {exc}",
              file=sys.stderr)
        return 1

    open(path, "w").write(src)
    print(f"[mori-trim] patched: per-forward dispatch trim wired into {path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
