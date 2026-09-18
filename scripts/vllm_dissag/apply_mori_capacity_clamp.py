#!/usr/bin/env python3
"""Decouple MoRI's buffer capacity from --max-num-batched-tokens. Default OFF.

436340 measured, on DSV4-Flash EP8 at con=1 (one real token per rank):

    mnbt 1024 -> ITL 307.03 ms      mnbt 256 -> ITL  82.45 ms
    mnbt  512 -> ITL 171.06 ms      mnbt 128 -> ITL  60.69 ms

    fit  ITL = 20.1 ms + 0.2816 * mnbt   (R^2 = 0.996)

The intercept is the allgather baseline (22.9 ms measured), so essentially ALL
of the MoRI excess is the mnbt-proportional term; and the slope, 35.2 us per
token-slot per rank, predicts the EP-width ladder it was never fitted to
(EP8 311 vs 306 measured, EP16 602 vs 592). One constant explains the
magnitude and the scaling together.

But mnbt is the WRONG knob to conclude from, because it moves two things at
once. vLLM wires a single scheduler value into both:

    --max-num-batched-tokens
      -> SchedulerConfig.max_num_batched_tokens
      -> FusedMoEConfig.max_num_tokens            (fused_moe/layer.py:343)
      -> all2all_utils.py  max_num_tokens_per_dp_rank=moe.max_num_tokens
      -> MoriAll2AllManager._make_all2all_kwargs  (all2all.py:977)
         max_num_inp_token_per_rank=max_num_tokens_per_dp_rank
      -> EpDispatchCombineConfig

so lowering mnbt shrinks the prefill chunk AND MoRI's symmetric buffers. This
patch cuts that link: MORI_CAP_TOKENS overrides ONLY the MoRI capacity and
leaves the scheduler alone. Holding mnbt at 1024 while capping MoRI is the
experiment the sweep could not run -- if ITL still falls, capacity is causal
and prefill chunking is exonerated.

In MoRI the value fans out as
    MaxNumTokensToSendPerRank() = maxNumInpTokenPerRank
    MaxNumTokensToRecv()        = worldSize * MaxNumTokensToRecvPerRank()
and that product is exactly the `mnbt * world_size` law the fit found. Two
consumers scale with it and this patch does not distinguish them -- MoRI's own
comm kernels, and the expert GEMM, since dispatch() hands back a view of
(max_num_tokens_to_recv(), hidden_dim) = 8192 x 4096 at mnbt 1024 / EP8 that
MoriPrepareAndFinalize.prepare() passes straight to rocm_aiter_fused_experts.
Use a torch profiler trace to tell those apart.

DIAGNOSTIC, NOT YET A PRODUCT KNOB. MoRI cannot accept more tokens in one
forward than its capacity: dispatch_combine.py:506-534 warns and the write
side drops, and the v2 op raises outright. So the cap must be >= the largest
per-rank token count in ANY forward, prefill included. At con=1 with isl=32
that is 32 tokens, so a cap of 64 is safe while mnbt stays 1024 -- but the
same cap with a 1024-token prefill chunk would silently drop tokens. Shipping
this means a per-phase capacity (a second handle sized for decode, selected on
token count), not a global clamp; MoriAll2AllManager.handle_cache already keys
handles on kwargs, so a second one costs only its buffers.

    MORI_CAP_TOKENS   per-rank MoRI buffer capacity, in tokens. Unset, empty,
                      0, negative, or >= the value vLLM computed all mean
                      "leave it alone", so an unset run is byte-identical.

Idempotent. Missing file -> skip. Found-old that fails to apply is a hard
error.

Usage: apply_mori_capacity_clamp.py <vllm_install_dir>
"""
import os
import sys

REL = "distributed/device_communicators/all2all.py"

MARKER = "_mori_capacity_tokens"

# Unique in the file: the DeepEP managers spell their own field
# `num_max_dispatch_tokens_per_rank`, so this is MoRI's line and only MoRI's.
OLD_FIELD = """            max_num_inp_token_per_rank=max_num_tokens_per_dp_rank,
"""
NEW_FIELD = """            max_num_inp_token_per_rank=_mori_capacity_tokens(
                max_num_tokens_per_dp_rank
            ),
"""

OLD_CLASS = """class MoriAll2AllManager(All2AllManagerBase):
"""
NEW_CLASS = '''def _mori_capacity_tokens(default: int) -> int:
    """MoRI's per-rank buffer capacity, overridable via MORI_CAP_TOKENS.

    `default` is what vLLM computed, i.e. moe.max_num_tokens, i.e.
    --max-num-batched-tokens. See apply_mori_capacity_clamp.py for why that is
    the wrong number for decode and why overriding it is a diagnostic rather
    than a product knob.

    Only ever lowers. An override that is unparseable, non-positive, or not
    actually smaller is ignored with a warning, so a misconfigured run behaves
    like an unpatched one instead of silently changing the cell.
    """
    import os

    raw = os.environ.get("MORI_CAP_TOKENS", "").strip()
    if not raw:
        return default
    try:
        cap = int(raw)
    except ValueError:
        logger.warning(
            "[mori-cap] MORI_CAP_TOKENS=%r is not an integer -- ignoring, "
            "keeping max_num_inp_token_per_rank=%d",
            raw,
            default,
        )
        return default
    if cap <= 0 or cap >= default:
        logger.info(
            "[mori-cap] MORI_CAP_TOKENS=%d does not lower the computed "
            "capacity %d -- keeping %d",
            cap,
            default,
            default,
        )
        return default
    # Loud on purpose. A capped run can drop tokens if any single forward
    # hands a rank more than `cap`, so this must never be mistaken for a
    # default configuration when reading a log after the fact.
    logger.warning(
        "[mori-cap] max_num_inp_token_per_rank %d -> %d via MORI_CAP_TOKENS. "
        "DIAGNOSTIC: MoRI drops tokens beyond its capacity, so this is only "
        "valid while every forward stays at or under %d tokens per rank.",
        default,
        cap,
        cap,
    )
    return cap


class MoriAll2AllManager(All2AllManagerBase):
'''


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[mori-cap] {REL} not found under {sys.argv[1]} -- skipping.")
        return 0

    src = open(path).read()
    if MARKER in src:
        print(f"[mori-cap] already patched in {path} -- no-op.")
        return 0

    for name, anchor, want in (
        ("field", OLD_FIELD, 1),
        ("class", OLD_CLASS, 1),
    ):
        n = src.count(anchor)
        if n != want:
            print(
                f"[mori-cap] ERROR: {name} anchor found {n} times in {path}, "
                f"expected {want}.",
                file=sys.stderr,
            )
            return 1

    src = src.replace(OLD_CLASS, NEW_CLASS, 1)
    src = src.replace(OLD_FIELD, NEW_FIELD, 1)

    if src.count(MARKER) < 2:
        print(
            f"[mori-cap] ERROR: post-write marker count is "
            f"{src.count(MARKER)}, expected >= 2, in {path}.",
            file=sys.stderr,
        )
        return 1

    import ast

    try:
        ast.parse(src)
    except SyntaxError as exc:
        print(f"[mori-cap] ERROR: patched file does not parse: {exc}",
              file=sys.stderr)
        return 1

    open(path, "w").write(src)
    print(f"[mori-cap] patched: MORI_CAP_TOKENS override wired into {path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
