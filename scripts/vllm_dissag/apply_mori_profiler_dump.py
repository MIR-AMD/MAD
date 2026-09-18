#!/usr/bin/env python3
"""Read out MoRI's kernel profiler from vLLM. Diagnostic, default OFF.

MoRI's dispatch/combine kernels carry named BEGIN/END spans (Perfetto slots
like `EpCombineSyncBarrier`), but two separate things have to be true to see
them and only the first is documented:

  1. the JIT build must carry -DENABLE_PROFILER, i.e. `ENABLE_PROFILER=1`
     (mori/jit/config.py:271), and
  2. SOMEBODY has to copy the device trace buffer back and serialize it.

vLLM never does (2). `MoriPrepareAndFinalize` calls dispatch() and combine()
and drops the buffer on the floor, so `ENABLE_PROFILER=1` alone compiles the
instrumentation and then shows you nothing -- which is what made the 302 ms
EP8 / 592 ms EP16 decode regression un-attributable. This patch does (2).

The pieces already exist and are found, not invented:
  mori.ops.EpDispatchCombineOp.get_debug_time_buf()  -> the device buffer
  mori.kernel_profiler.export_to_perfetto(buf, path) -> Perfetto JSON, with
      slot names auto-discovered from mori.cpp's *Slots enums
  mori/_jit-sources/tools/profiler/analyze_ep_kernel_trace.py   (one rank)
  mori/_jit-sources/tools/profiler/aggregate_ep_kernel_trace.py (all ranks,
      wants a 'traces/trace_rank_*.json' glob)

Enable with MORI_PROFILE_DIR. Unset = not one extra branch taken per token
beyond an `is None` test, so this is safe to leave in the patch stack.

    MORI_PROFILE_DIR    output dir. Unset (default) disables everything.
    MORI_PROFILE_AT     dump on the Nth finalize() call. Default 1500.
                        finalize() runs once per MoE layer per forward step,
                        so N/(MoE layers) is the step you land on: GLM-5 has
                        ~75 MoE layers (0-2 are dense) -> step ~20;
                        DSV4-Flash has ~40 -> step ~37. Both are inside
                        steady-state decode for a 32-token generation.
    MORI_PROFILE_RANKS  "0" (default) or "all". Each rank writes its own
                        ~1 GiB-backed buffer, so "all" is 8x the cost and is
                        only worth it when hunting a straggler rank.

Cost when it fires: one device->host copy of the trace buffer plus a numpy
parse, inline in the decode loop. It stalls that one step for seconds. It is
one-shot per process by construction -- do not read the ITL of a run that
dumped, and do not compare it to a run that did not.

MoRI's JIT cache is keyed on the profiler flag, so turning ENABLE_PROFILER on
does not silently reuse non-instrumented kernels. It does mean a cold JIT
build (~10 min) on the first cell.

Idempotent. Missing file -> skip. Found-old that fails to apply is a hard
error. Anchors are chosen to compose with
apply_mori_combine_original_topk_fix.py in either order.

Usage: apply_mori_profiler_dump.py <vllm_install_dir>
"""
import os
import sys

REL = "model_executor/layers/fused_moe/prepare_finalize/mori.py"

MARKER = "_mori_profile_dump"

# Anchor on the single line, not the pair the combine patcher uses, so that
# patcher can insert its own field after it either before or after this runs.
OLD_INIT = """        self.use_fp8_dispatch = use_fp8_dispatch
"""
NEW_INIT = """        self.use_fp8_dispatch = use_fp8_dispatch
        # Which MoRI kernel a model actually drives is decided here and is not
        # readable from config.json or from the JIT build log -- MoRI's
        # WRAP_ALL_TYPES_BOOL3_ENTRY macro emits the fp4, fp8 and bf16 variants
        # of every kernel into one translation unit, so seeing
        # `EpCombineIntraNodeKernel_fp4_p2p_stdmoe` compile says nothing about
        # what runs. all2all_utils.py sets quant_dtype to
        # quant_config.quant_dtype when use_fp8_dispatch, else to moe.in_dtype.
        # Log it once so cross-model comparisons can be checked, not assumed.
        logger.info_once(
            "[mori-dispatch] use_fp8_dispatch=%s op_config=%s",
            use_fp8_dispatch,
            getattr(mori_op, "config", None),
        )
        # MoRI kernel profiler readout -- see apply_mori_profiler_dump.py.
        # Unset MORI_PROFILE_DIR (the default) leaves this None and the only
        # cost in finalize() is one `is not None` test per layer per token.
        self._mori_profile_dir = os.environ.get("MORI_PROFILE_DIR") or None
        self._mori_profile_at = int(os.environ.get("MORI_PROFILE_AT", "1500"))
        self._mori_profile_ranks = os.environ.get("MORI_PROFILE_RANKS", "0")
        self._mori_profile_calls = 0
        self._mori_profile_done = False
        if self._mori_profile_dir is not None:
            logger.info(
                "[mori-profile] armed: dir=%s at_call=%d ranks=%s "
                "(needs ENABLE_PROFILER=1 in the JIT build)",
                self._mori_profile_dir,
                self._mori_profile_at,
                self._mori_profile_ranks,
            )
"""

OLD_IMPORT = """import mori
import torch
"""
NEW_IMPORT = """import os

import mori
import torch
"""

OLD_FINALIZE = """        output.copy_(result[:num_token])
"""
NEW_FINALIZE = '''        output.copy_(result[:num_token])
        if self._mori_profile_dir is not None and not self._mori_profile_done:
            self._mori_profile_calls += 1
            if self._mori_profile_calls >= self._mori_profile_at:
                self._mori_profile_dump()

    def _mori_profile_dump(self) -> None:
        """Serialize MoRI's kernel trace for this rank, once, then disarm.

        Disarm FIRST: if anything below raises we want a single logged
        failure, not one per layer per token for the rest of the run.
        """
        self._mori_profile_done = True
        try:
            from mori.kernel_profiler import export_to_perfetto

            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 0
            )
            if self._mori_profile_ranks != "all" and rank != 0:
                return

            buf = self.mori_op.get_debug_time_buf()
            if buf is None or buf.numel() == 0:
                logger.warning(
                    "[mori-profile] rank %d: trace buffer is empty. The JIT "
                    "cache was almost certainly built without "
                    "ENABLE_PROFILER=1.",
                    rank,
                )
                return

            os.makedirs(self._mori_profile_dir, exist_ok=True)
            path = os.path.join(
                self._mori_profile_dir, f"trace_rank_{rank}.json"
            )
            # Warns and returns without writing if no events decoded, which
            # is the other way an uninstrumented build shows up.
            export_to_perfetto(buf, filename=path)
            logger.info(
                "[mori-profile] rank %d: wrote %s after %d finalize calls. "
                "Analyze with mori/_jit-sources/tools/profiler/"
                "aggregate_ep_kernel_trace.py '%s/trace_rank_*.json'",
                rank,
                path,
                self._mori_profile_calls,
                self._mori_profile_dir,
            )
        except Exception:
            logger.exception("[mori-profile] dump failed; continuing")
'''


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[mori-profile] {REL} not found under {sys.argv[1]} -- skipping.")
        return 0

    src = open(path).read()
    if MARKER in src:
        print(f"[mori-profile] already patched in {path} -- no-op.")
        return 0

    for name, anchor in (
        ("import", OLD_IMPORT),
        ("__init__", OLD_INIT),
        ("finalize", OLD_FINALIZE),
    ):
        if anchor not in src:
            print(
                f"[mori-profile] ERROR: {name} anchor missing in {path}.",
                file=sys.stderr,
            )
            return 1

    src = src.replace(OLD_IMPORT, NEW_IMPORT, 1)
    src = src.replace(OLD_INIT, NEW_INIT, 1)
    src = src.replace(OLD_FINALIZE, NEW_FINALIZE, 1)

    if src.count(MARKER) < 2:
        print(
            f"[mori-profile] ERROR: post-write marker count is "
            f"{src.count(MARKER)}, expected >= 2, in {path}.",
            file=sys.stderr,
        )
        return 1

    import ast

    try:
        ast.parse(src)
    except SyntaxError as exc:
        print(f"[mori-profile] ERROR: patched file does not parse: {exc}",
              file=sys.stderr)
        return 1

    open(path, "w").write(src)
    print(f"[mori-profile] patched: kernel trace readout wired into {path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
