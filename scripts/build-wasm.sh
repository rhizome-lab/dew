#!/usr/bin/env bash
# Build WASM profiles for dew playground
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WASM_CRATE="$SCRIPT_DIR/../crates/dew-wasm"
OUT_DIR="$SCRIPT_DIR/../playground/dist/wasm"

# Profiles: name -> cargo features
declare -A PROFILES=(
    ["core"]="core"
    ["linalg"]="linalg"
    ["graphics"]="graphics"
    ["signal"]="signal"
    ["full"]="full"
)

build_profile() {
    local name="$1"
    local features="$2"
    local out="$OUT_DIR/$name"

    echo "Building profile: $name (features: $features)"

    local start_time=$SECONDS

    cd "$WASM_CRATE"
    wasm-pack build \
        --target web \
        --out-dir "$out" \
        --no-default-features \
        --features "console_error_panic_hook,$features" \
        2>&1 | grep -v "^\[INFO\]" || true

    local elapsed=$((SECONDS - start_time))

    local size=$(ls -lh "$out"/*.wasm 2>/dev/null | awk '{print $5}')
    echo "  -> $size in ${elapsed}s"
    echo
}

main() {
    mkdir -p "$OUT_DIR"

    echo "=== Building WASM profiles ==="
    echo

    local total_start=$SECONDS

    for profile in "${!PROFILES[@]}"; do
        build_profile "$profile" "${PROFILES[$profile]}"
    done

    local total_elapsed=$((SECONDS - total_start))

    echo "=== Summary ==="
    echo "Total time: ${total_elapsed}s"
    echo
    echo "Profile sizes:"
    for profile in "${!PROFILES[@]}"; do
        local size=$(ls -lh "$OUT_DIR/$profile"/*.wasm 2>/dev/null | awk '{print $5}')
        printf "  %-12s %s\n" "$profile:" "$size"
    done
}

main "$@"
