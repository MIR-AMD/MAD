#!/bin/bash
# Dump Python stacks from every process in a wedged vLLM container.
#
# Written for the EP16 allgather_reducescatter deadlock (436083): the server
# takes exactly 2 requests and then stops scheduling, with /health still 200 and
# no error on any rank. Nothing in the logs says where the ranks are, so the
# stacks are the only evidence.
#
# Runs INSIDE the container so py-spy shares the PID namespace; the container is
# already --privileged with SYS_PTRACE so no extra caps are needed. py-spy is NOT
# in the v0290 image and the compute nodes can reach pypi, so install on first use.
#
# Process selection is by comm, NOT `pgrep -x python3`: vLLM retitles everything
# via setproctitle, so the engines are `VLLM::EngineCore_DP0`,
# `VLLM::Worker_DP0_EP0`, `VLLM::DPCoordinator`, and the API server is `vllm`.
# A python3-only match returns nothing at all.
#
# Run it on BOTH nodes -- the interesting asymmetry in 436083 was between ranks
# on the master, and the child was needed to prove the other 8 were merely
# waiting.
#
#   srun --jobid=$JOB --overlap -N1 -n1 --nodelist=$NODE \
#       dump_wedged_stacks.sh <container> [out-file]
set -u
C="${1:?container name required}"
OUT="${2:-/run_logs/stacks_$(hostname -s).txt}"

docker exec "$C" bash -c "
set -u
which py-spy >/dev/null 2>&1 || pip install -q --root-user-action=ignore py-spy
{
  echo '########################################################'
  echo \"# host=\$(hostname -s)  container=$C  utc=\$(date -u +%FT%TZ)\"
  echo '########################################################'
  ps -eo pid=,comm= | awk '\$2 ~ /^(vllm|VLLM::)/ {print \$1}' | sort -n | while read -r p; do
    [ -r /proc/\$p/cmdline ] || continue
    name=\$(cat /proc/\$p/comm 2>/dev/null)
    wchan=\$(cat /proc/\$p/wchan 2>/dev/null)
    echo
    echo \"===== pid=\$p comm=\$name wchan=\${wchan:-?} =====\"
    timeout 90 py-spy dump --pid \$p 2>&1 | sed 's/^/    /'
  done
} > '$OUT' 2>&1
echo \"wrote $OUT (\$(wc -l < '$OUT') lines)\"
"
