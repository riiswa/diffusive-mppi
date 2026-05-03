#!/usr/bin/env bash
# run_experiments.sh — run all MPPI experiments lightest → heaviest
#
# Usage:
#   bash run_experiments.sh                # run all
#   bash run_experiments.sh --continue     # skip already completed envs

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
SCRIPT="mjp_mppi.py"
METHOD="all"
LOG_DIR="logs"
DONE_FILE="logs/completed.txt"

# envs ordered lightest → heaviest
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
    "FingerTurnEasy"
    "FingerTurnHard"
    "FishSwim"
    "SwimmerSwimmer6"
    "SwimmerSwimmer15"
    # playground — heavy
    "HumanoidStand"
    "HumanoidWalk"
    "HumanoidRun"
    # gcrl — locomotion
    "ant"
    "ant_ball"
    "ant_push"
    "humanoid"
    # gcrl — mazes
    "ant_u_maze"
    "ant_big_maze"
    "ant_hardest_maze"
    "humanoid_u_maze"
    "humanoid_big_maze"
    "humanoid_hardest_maze"
    # gcrl — manipulation
    "arm_reach"
    "arm_push_easy"
    "arm_push_hard"
    "arm_grasp"
    "arm_binpick_easy"
    "arm_binpick_easy_eef"
    "arm_binpick_hard"
)

TOTAL=${#ENVS[@]}

# ── flags ─────────────────────────────────────────────────────────────────────
CONTINUE=false
for arg in "$@"; do
    [[ "$arg" == "--continue" ]] && CONTINUE=true
done

# ── setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
touch "$DONE_FILE"

# ── helpers ───────────────────────────────────────────────────────────────────
is_done() {
    grep -qxF "$1" "$DONE_FILE"
}

mark_done() {
    echo "$1" >> "$DONE_FILE"
}

# ── main loop ─────────────────────────────────────────────────────────────────
COMPLETED=0
SKIPPED=0
FAILED=0

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MPPI experiments  [${TOTAL} envs]  method=${METHOD}"
[[ "$CONTINUE" == true ]] && echo "  --continue: skipping already completed envs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for i in "${!ENVS[@]}"; do
    ENV="${ENVS[$i]}"
    IDX=$((i + 1))
    PREFIX="[${IDX}/${TOTAL}]"

    # skip if already done and --continue is set
    if [[ "$CONTINUE" == true ]] && is_done "$ENV"; then
        echo "${PREFIX} SKIP  ${ENV}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo -n "${PREFIX} RUN   ${ENV} ... "

    python "$SCRIPT" \
        --env    "$ENV"    \
        --method "$METHOD" \
        > "${LOG_DIR}/${ENV}.out" \
        2> "${LOG_DIR}/${ENV}.err"

    EXIT=$?

    if [[ $EXIT -eq 0 ]]; then
        mark_done "$ENV"
        COMPLETED=$((COMPLETED + 1))
        echo "OK"
    else
        FAILED=$((FAILED + 1))
        echo "FAILED  (see ${LOG_DIR}/${ENV}.err)"
    fi

done

# ── summary ───────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  done=${COMPLETED}  skipped=${SKIPPED}  failed=${FAILED}"
[[ $FAILED -gt 0 ]] && echo "  failed envs: check logs/*.err" && exit 1
exit 0