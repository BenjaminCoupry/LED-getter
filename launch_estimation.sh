#!/bin/bash

CONFIG_FILE=${1:-config.sh}
source "$CONFIG_FILE"

### === EXECUTION ===


mkdir -p "$BASE_OUT_PATH"
[ -f "$CONFIG_FILE" ] && cp "$CONFIG_FILE" "$BASE_OUT_PATH/$(basename "$CONFIG_FILE")"

echo "Using config file: $CONFIG_FILE"

for PATTERN in "${PATTERN_LIST[@]}"; do
    echo "=== Processing pattern: $PATTERN ==="

    CMD="python estimate.py"

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
    ESTIM_OUT_PATH="$BASE_OUT_PATH/light/${PATTERN}"
    CMD_ESTIM+=" --pattern $PATTERN"
    CMD_ESTIM+=" --out_path \"$ESTIM_OUT_PATH\""
    CMD_ESTIM+=" --backend gpu"
    CMD_ESTIM+=" --slice_i 0"
    [ -n "$ESTIM_STEP" ] && CMD_ESTIM+=" --step $ESTIM_STEP"
    [ "$PATTERN" = "grid" ] && [ -n "$PIXEL_STEP" ] && CMD_ESTIM+=" --pixel_step $PIXEL_STEP"
    if [[ ! " ${SKIP_LIGHT_LOAD[@]} " =~ " ${PATTERN} " ]] && [ -n "$PREV_OUT_PATH" ]; then
        CMD_ESTIM+=" --loaded_light_folder \"$PREV_OUT_PATH\""
    fi

    echo "Running light estimation: $CMD_ESTIM"

    mkdir -p "$ESTIM_OUT_PATH"
    ESTIM_SCRIPT_PATH="$ESTIM_OUT_PATH/estim.sh"
    echo "#!/bin/bash" > "$ESTIM_SCRIPT_PATH"
    chmod +x "$ESTIM_SCRIPT_PATH"
    echo  "$CMD_ESTIM" >> "$ESTIM_SCRIPT_PATH"
    echo "" >> "$ESTIM_SCRIPT_PATH"
    
    eval $CMD_ESTIM

    PREV_OUT_PATH="$ESTIM_OUT_PATH"

    PSS_OUT_PATH="$BASE_OUT_PATH/ps/${PATTERN}"

    mkdir -p "$PSS_OUT_PATH"
    PS_SCRIPT_PATH="$PSS_OUT_PATH/ps.sh"
    echo "#!/bin/bash" > "$PS_SCRIPT_PATH"
    chmod +x "$PS_SCRIPT_PATH"    

    if [[ " ${SOLVE_PS[@]} " =~ " ${PATTERN} " ]]; then
        TO_MERGE=""
        TOTAL_SLICES=$((PS_STEP * PS_STEP - 1))
        for (( SLICE_I=0; SLICE_I<=TOTAL_SLICES; SLICE_I++ )); do
            CMD_PS="$CMD"
            SLICE_DIR=$(printf "slice_%05d" "$SLICE_I")
            PS_OUT_PATH="$PSS_OUT_PATH/${SLICE_DIR}"
            CMD_PS+=" --pattern PS"
            CMD_PS+=" --out_path \"$PS_OUT_PATH\""
            CMD_PS+=" --loaded_light_folder \"$PREV_OUT_PATH\""
            [ -n "$PS_STEP" ] && CMD_PS+=" --step $PS_STEP"
            CMD_PS+=" --slice_i $SLICE_I"
            CMD_PS+=" --backend cpu"
            CMD_PS+=" --skip_export images lightmaps light misc"

            TO_MERGE+=" \"$PS_OUT_PATH/values/values.npz\""

            echo "$CMD_PS" >> "$PS_SCRIPT_PATH"
            echo "" >> "$PS_SCRIPT_PATH"
            echo "Running PS (slice $SLICE_I/$TOTAL_SLICES): $CMD_PS"
            eval $CMD_PS
        done

        CMD_MERGE="python estimate.py --backend cpu --out_path \"$PSS_OUT_PATH/merged\" --paths $TO_MERGE"

        echo "$CMD_MERGE" >> "$PS_SCRIPT_PATH"
        echo "" >> "$PS_SCRIPT_PATH"
        echo "Merging PS results: $CMD_MERGE"
        eval $CMD_MERGE

    fi

done
