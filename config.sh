#!/bin/bash

### === CONFIGURATION ===

PATTERN_LIST=("directional" "rail" "punctual" "LED" "specular" "harmonic" "grid")
SOLVE_PS=("directional" "rail" "punctual" "LED" "specular" "harmonic" "grid")
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
ITERATIONS=10000
TQDM_REFRESH=1

#Light estimation (set to empty string "" to omit them)
PIXEL_STEP="800"

# Photometric stereo (set to empty string "" to omit them)
ESTIM_STEP=10
PS_STEP=4
PS_CHUNCK_NUMBER=100

# Start with some loaded light (set to empty string "" to omit them)
PREV_OUT_PATH=""