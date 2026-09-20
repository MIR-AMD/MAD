#!/bin/bash
# Host leftover teardown. Run on each allocated node BEFORE docker run.
#
# scancel does not run connector_cleanup_ports / the trailing docker stop
# (SIGKILL skips those). The next job must prerun this. 217459: Hy3 engines
# Ready, then ZMQ :36367 EADDRINUSE; cancel left host-net listeners for 217463.
#
# This is the host (fuser exists). The vLLM image has neither ss nor fuser.
#
# Docker kill is name-filtered to this stack's leftovers, not every container
# on the node:
#   container_${MODEL}_${JOBID}   run_xPyD_models.slurm
#   dsv4coloc-*                   run_dsv4_colocated_tp8.slurm
#   dsv4ep16-*                    run_dsv4_ep16_a2a.slurm
#   dsv4sweep-* / dsv4hold-*      run_dsv4_tp_ep_sweep.slurm
# Unrelated host-net sidecars are left alone. fuser still frees our listen
# ports if an unnamed leftover is holding one.
set -u

echo "[prerun-cleanup] $(hostname) begin"

_ours() {
    case "$1" in
        container_*|dsv4coloc-*|dsv4ep16-*|dsv4sweep-*|dsv4hold-*) return 0 ;;
        *) return 1 ;;
    esac
}

# docker ps --filter name= is a substring match; walk names ourselves so a
# container named "my_container_backup" is not collateral.
while read -r _id _name; do
    [[ -z "${_id}" ]] && continue
    if _ours "${_name}"; then
        echo "[prerun-cleanup] docker kill ${_name} (${_id})"
        # kill, not stop: cancelled jobs leave host-net listeners until SIGKILL.
        docker kill "${_id}" 2>/dev/null || true
        docker rm -f "${_id}" 2>/dev/null || true
    fi
done < <(docker ps -a --format '{{.ID}} {{.Names}}' 2>/dev/null || true)

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
