#!/bin/bash

CONFIG_FILE=${1:-config.sh}
source "$CONFIG_FILE"

### === EXECUTION ===

PREV_OUT_PATH=""

cp "$0" "$BASE_OUT_PATH/$(basename "$0")"
[ -f "$CONFIG_FILE" ] && cp "$CONFIG_FILE" "$BASE_OUT_PATH/$(basename "$CONFIG_FILE")"

echo "Using config file: $CONFIG_FILE"

for PATTERN in "${PATTERN_LIST[@]}"; do
    echo "=== Processing pattern: $PATTERN ==="

    CMD="python main.py"

    # Required
    CMD+=" --ps_images_paths $PS_IMAGES_PATHS"
    
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
    [ -n "$PS_CHUNCK_NUMBER" ] && CMD+=" --ps_chunck_number $PS_CHUNCK_NUMBER"
    

    CMD_ESTIM="$CMD"
    OUT_PATH="$BASE_OUT_PATH/light/${PATTERN}"
    CMD_ESTIM+=" --pattern $PATTERN"
    CMD_ESTIM+=" --out_path \"$OUT_PATH\""
    CMD_ESTIM+=" --backend gpu"
    CMD_ESTIM+=" --slice_i 0"
    [ -n "$ESTIM_STEP" ] && CMD_ESTIM+=" --step $ESTIM_STEP"
    [ "$PATTERN" = "grid" ] && [ -n "$PIXEL_STEP" ] && CMD_ESTIM+=" --pixel_step $PIXEL_STEP"
    if [[ ! " ${SKIP_LIGHT_LOAD[@]} " =~ " ${PATTERN} " ]] && [ -n "$PREV_OUT_PATH" ]; then
        CMD_ESTIM+=" --loaded_light_folder \"$PREV_OUT_PATH\""
    fi

    echo "Running light estimation: $CMD_ESTIM"
    eval $CMD_ESTIM

    PREV_OUT_PATH="$OUT_PATH"

if [[ " ${SOLVE_PS[@]} " =~ " ${PATTERN} " ]]; then
    for (( SLICE_I=0; SLICE_I<PS_STEP*PS_STEP; SLICE_I++ )); do
        CMD_PS="$CMD"
        SLICE_DIR=$(printf "slice_%05d" "$SLICE_I")
        OUT_PATH="$BASE_OUT_PATH/ps/${PATTERN}/${SLICE_DIR}"
        CMD_PS+=" --pattern PS"
        CMD_PS+=" --out_path \"$OUT_PATH\""
        CMD_PS+=" --loaded_light_folder \"$PREV_OUT_PATH\""
        [ -n "$PS_STEP" ] && CMD_PS+=" --step $PS_STEP"
        CMD_PS+=" --slice_i $SLICE_I"
        CMD_PS+=" --backend cpu"
        CMD_PS+=" --skip_export images lightmaps light misc"

        echo "Running PS (slice $SLICE_I): $CMD_PS"
        eval $CMD_PS
    done
fi



done
