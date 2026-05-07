#!/usr/bin/env bash
# run_experiments.sh — run MPPI experiments in parallel across CUDA devices
#
# Usage:
#   bash run_experiments.sh                          # all GPUs, all envs
#   bash run_experiments.sh --devices 0 1            # specific GPUs
#   bash run_experiments.sh --continue               # skip completed
#   bash run_experiments.sh --devices 0 1 --continue

set -euo pipefail
trap 'echo ""; echo "Interrupted — killing all jobs..."; kill $(jobs -p) 2>/dev/null; exit 1' INT TERM

# ── config ────────────────────────────────────────────────────────────────────
SCRIPT="mjp_mppi.py"
METHOD="all"
LOG_DIR="logs"
DONE_FILE="logs/completed.txt"

ENVS=(
    # playground — simple
    "PointMass"
    "ReacherEasy"
    "ReacherHard"
    "BallInCup"
    # playground — medium
    "CheetahRun"
    "HopperStand"
    "HopperHop"
    "WalkerStand"
    "WalkerWalk"
    "WalkerRun"
    "FingerSpin"
    "SwimmerSwimmer6"
    # gcrl — locomotion
    "ant"
    "ant_ball"
    # gcrl — mazes
    "ant_u_maze"
    "ant_big_maze"
    # gcrl — manipulation
    "arm_reach"
    "arm_push_easy"
    "arm_push_hard"
    "arm_grasp"
    "arm_binpick_easy"
    "arm_binpick_hard"
    "FishSwim"
    # playground — heavy
    "HumanoidStand"
    "HumanoidWalk"
    "HumanoidRun"
    "humanoid"
)

TOTAL=${#ENVS[@]}

# ── parse flags ───────────────────────────────────────────────────────────────
CONTINUE=false
DEVICES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --continue)
            CONTINUE=true
            shift
            ;;
        --devices)
            shift
            while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do
                DEVICES+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# default: GPU 0
if [[ ${#DEVICES[@]} -eq 0 ]]; then
    DEVICES=(0)
fi

if [[ ${#DEVICES[@]} -eq 0 ]]; then
    echo "No CUDA devices found."
    exit 1
fi

N_GPUS=${#DEVICES[@]}

# ── setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
touch "$DONE_FILE"

# ── helpers ───────────────────────────────────────────────────────────────────
is_done()  { grep -qxF "$1" "$DONE_FILE"; }
mark_done() { echo "$1" >> "$DONE_FILE"; }

# ── build queue: envs to actually run ─────────────────────────────────────────
QUEUE=()
for ENV in "${ENVS[@]}"; do
    if [[ "$CONTINUE" == true ]] && is_done "$ENV"; then
        continue
    fi
    QUEUE+=("$ENV")
done

N_QUEUE=${#QUEUE[@]}
N_SKIP=$((TOTAL - N_QUEUE))

# ── state tracking ────────────────────────────────────────────────────────────
# pid_of[device]  → PID of job running on that device (-1 = free)
# env_of[device]  → env name running on that device
declare -A pid_of
declare -A env_of
declare -A last_check_of
for g in "${DEVICES[@]}"; do
    pid_of[$g]=-1
    env_of[$g]=""
done

COMPLETED=0
FAILED=0
FAILED_ENVS=()

# ── display ───────────────────────────────────────────────────────────────────
print_status() {
    local done_total=$((COMPLETED + FAILED + N_SKIP))
    echo -ne "\r  [${done_total}/${TOTAL}] done=${COMPLETED} failed=${FAILED} skipped=${N_SKIP}  |"
    for g in "${DEVICES[@]}"; do
        if [[ ${pid_of[$g]} -eq -1 ]]; then
            echo -ne "  GPU${g}:idle"
        else
            echo -ne "  GPU${g}:${env_of[$g]}"
        fi
    done
    echo -ne "  "
}

# ── main loop ─────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MPPI experiments  [${TOTAL} envs → ${N_QUEUE} to run]"
echo "  devices: ${DEVICES[*]}  method=${METHOD}"
[[ "$CONTINUE" == true ]] && echo "  --continue: ${N_SKIP} already completed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

Q_IDX=0   # next env to dispatch from QUEUE

while true; do

    # ── reap finished jobs ────────────────────────────────────────────────────
    for g in "${DEVICES[@]}"; do
        PID=${pid_of[$g]}
        [[ $PID -eq -1 ]] && continue

        if ! kill -0 "$PID" 2>/dev/null; then
            # process finished — check exit code
            wait "$PID" && EXIT=0 || EXIT=$?
            ENV="${env_of[$g]}"

            if [[ $EXIT -eq 0 ]]; then
                mark_done "$ENV"
                COMPLETED=$((COMPLETED + 1))
            else
                FAILED=$((FAILED + 1))
                FAILED_ENVS+=("$ENV")
                echo "$ENV" >> "${LOG_DIR}/failed.txt"
            fi

            pid_of[$g]=-1
            env_of[$g]=""
        fi
        # check for naconmax overflow every 30s
        if [[ ${pid_of[$g]} -ne -1 ]]; then
            NOW=$(date +%s)
            if (( NOW - last_check_of[$g] >= 30 )); then
                last_check_of[$g]=$NOW
                if grep -q "naconmax" "${LOG_DIR}/${env_of[$g]}.out" 2>/dev/null; then
                    ENV="${env_of[$g]}"
                    echo ""
                    echo "  [OVERFLOW] ${ENV} — naconmax too small, killing (see ${LOG_DIR}/${ENV}.out)"
                    kill -- -"${pid_of[$g]}" 2>/dev/null
                    FAILED=$((FAILED + 1))
                    FAILED_ENVS+=("$ENV")
                    echo "$ENV" >> "${LOG_DIR}/failed.txt"
                    pid_of[$g]=-1
                    env_of[$g]=""
                fi
            fi
        fi
    done

    # ── dispatch new jobs onto free devices ───────────────────────────────────
    for g in "${DEVICES[@]}"; do
        [[ ${pid_of[$g]} -ne -1 ]] && continue   # device busy
        [[ $Q_IDX -ge $N_QUEUE   ]] && continue   # queue empty

        ENV="${QUEUE[$Q_IDX]}"
        Q_IDX=$((Q_IDX + 1))

        setsid bash -c "CUDA_VISIBLE_DEVICES=$g python $SCRIPT \
            --env $ENV --method $METHOD \
            2> ${LOG_DIR}/${ENV}.err | \
            while IFS= read -r line; do \
                buf+=(\"\$line\"); \
                [[ \${#buf[@]} -gt 10 ]] && buf=(\"\${buf[@]:1}\"); \
                printf '%s\n' \"\${buf[@]}\" > ${LOG_DIR}/${ENV}.out; \
            done" &

        pid_of[$g]=$!
        env_of[$g]="$ENV"
        last_check_of[$g]=$(date +%s)
    done

    print_status

    # ── exit condition: queue exhausted + all devices idle ────────────────────
    ALL_IDLE=true
    for g in "${DEVICES[@]}"; do
        [[ ${pid_of[$g]} -ne -1 ]] && ALL_IDLE=false && break
    done
    [[ $ALL_IDLE == true && $Q_IDX -ge $N_QUEUE ]] && break

    sleep 2
done

# ── final summary ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  done=${COMPLETED}  skipped=${N_SKIP}  failed=${FAILED}"
if [[ $FAILED -gt 0 ]]; then
    echo "  failed envs:"
    for e in "${FAILED_ENVS[@]}"; do
        echo "    ${e}  →  ${LOG_DIR}/${e}.err"
    done
    exit 1
fi
exit 0
