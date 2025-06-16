#!/bin/bash

### === CONFIGURATION ===

PATTERN_LIST=("directional" "rail" "punctual" "LED" "specular" "harmonic" "grid")
SKIP_LIGHT_LOAD=("grid" "directional")

# Required
PS_IMAGES_PATHS=""
BASE_OUT_PATH=""

# Optional paths (set to empty string "" to omit them)
MESHROOM_PROJECT=""
ALIGNED_IMAGE_PATH=""
GEOMETRY_PATH=""
POSE_PATH=""
BLACK_IMAGE_PATH=""

# Optimization and control (set to empty string "" to omit them)
DELTA=0.01
LEARNING_RATE=0.001
ITERATIONS=1000
TQDM_REFRESH=1

# Photometric stereo (set to empty string "" to omit them)
STEP=21
SLICE_I=0
PS_CHUNCK_NUMBER=100

# Backend (set to empty string "" to omit them)
BACKEND="gpu" 

### === EXECUTION ===

PREV_OUT_PATH=""

for PATTERN in "${PATTERN_LIST[@]}"; do
    echo "=== Processing pattern: $PATTERN ==="

    OUT_PATH="$BASE_OUT_PATH/${PATTERN}_1"

    CMD="python main.py"

    # Required
    CMD+=" --pattern $PATTERN"
    CMD+=" --ps_images_paths $PS_IMAGES_PATHS"
    CMD+=" --out_path \"$OUT_PATH\""


    # Optional paths
    [ -n "$MESHROOM_PROJECT" ] && CMD+=" --meshroom_project \"$MESHROOM_PROJECT\""
    [ -n "$ALIGNED_IMAGE_PATH" ] && CMD+=" --aligned_image_path \"$ALIGNED_IMAGE_PATH\""
    [ -n "$GEOMETRY_PATH" ] && CMD+=" --geometry_path \"$GEOMETRY_PATH\""
    [ -n "$POSE_PATH" ] && CMD+=" --pose_path \"$POSE_PATH\""
    [ -n "$BLACK_IMAGE_PATH" ] && CMD+=" --black_image_path \"$BLACK_IMAGE_PATH\""
    [ -n "$DELTA" ] && CMD+=" --delta $DELTA"
    [ -n "$LEARNING_RATE" ] && CMD+=" --learning_rate $LEARNING_RATE"
    [ -n "$ITERATIONS" ] && CMD+=" --iterations $ITERATIONS"
    [ -n "$TQDM_REFRESH" ] && CMD+=" --tqdm_refresh $TQDM_REFRESH"
    [ -n "$STEP" ] && CMD+=" --step $STEP"
    [ -n "$SLICE_I" ] && CMD+=" --slice_i $SLICE_I"
    [ -n "$PS_CHUNCK_NUMBER" ] && CMD+=" --ps_chunck_number $PS_CHUNCK_NUMBER"
    [ -n "$BACKEND" ] && CMD+=" --backend $BACKEND"

    # Load previous light unless it's directional or grid
    if [[ ! " ${SKIP_LIGHT_LOAD[@]} " =~ " ${PATTERN} " ]] && [ -n "$PREV_OUT_PATH" ]; then
        CMD+=" --loaded_light_folder \"$PREV_OUT_PATH\""
    fi

    echo "Running: $CMD"
    eval $CMD

    # Update previous output path
    if [ "$PATTERN" != "grid" ]; then
        PREV_OUT_PATH="$OUT_PATH"
    fi
done
