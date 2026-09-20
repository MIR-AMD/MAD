#!/bin/bash
# Host leftover teardown. Run on each allocated node BEFORE docker run.
#
# scancel does not run connector_cleanup_ports / the trailing docker stop
# (SIGKILL skips those). The next job must prerun this. 217459: Hy3 engines
# Ready, then ZMQ :36367 EADDRINUSE; cancel left host-net listeners for 217463.
#
# This is the host (fuser exists). The vLLM image has neither ss nor fuser.
#
# Kill every container on the node. These jobs are --exclusive: the allocation
# is the only occupant, and leftovers from a previous scancel are exactly what
# we have to clear. Same pattern as run_xPyD_models.slurm teardown
# (`docker ps -q | xargs docker stop`). A name/label filter would miss unnamed
# --rm leftovers that still pin 10001/36367.
set -u

echo "[prerun-cleanup] $(hostname) begin"

_ids=$(docker ps -aq 2>/dev/null || true)
if [[ -n "${_ids}" ]]; then
    echo "[prerun-cleanup] docker kill ${_ids}"
    # kill, not stop: cancelled jobs leave host-net listeners until SIGKILL.
    docker kill ${_ids} 2>/dev/null || true
    docker rm -f ${_ids} 2>/dev/null || true
fi
if [[ -n "${DOCKER_CONT_NAME:-}" ]]; then
    docker rm -f "${DOCKER_CONT_NAME}" 2>/dev/null || true
fi

# Same port set as run_xPyD_models.slurm srun (PD proxy + MoRIIO + barrier).
_ports="2222 5000 15000 2584 5557 9711 10001 13345 14600 18001 20005 29000 30000 36367 61005 61555 8405"
for _p in ${_ports}; do
    fuser -k "${_p}/tcp" 2>/dev/null || true
done
sleep 2

for _p in 10001 36367; do
    _hit=$(fuser "${_p}/tcp" 2>/dev/null || true)
    if [[ -n "${_hit}" ]]; then
        echo "[prerun-cleanup] WARN tcp ${_p} still busy pids=${_hit}"
        fuser -v "${_p}/tcp" 2>/dev/null || true
    else
        echo "[prerun-cleanup] tcp ${_p} free"
    fi
    hex=$(printf '%04X' "${_p}")
    awk -v p="$hex" 'NR > 1 {
        n = split($2, a, ":")
        if (n >= 2 && toupper(a[n]) == p) print "[prerun-cleanup]", FILENAME, $2, "st="$4, "inode="$10
    }' /proc/net/tcp /proc/net/tcp6 2>/dev/null || true
done
echo "[prerun-cleanup] $(hostname) done"
