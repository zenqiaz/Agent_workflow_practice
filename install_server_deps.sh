#!/usr/bin/env bash
# Install server-side dependencies for local ORCA mode.
# Usage: bash install_server_deps.sh /path/to/orca [--with-molsimplify]
#
# Prerequisites:
#   - Python 3.11 or 3.12 venv already activated
#   - ORCA 6.1.1 downloaded from https://orcaforum.kofo.mpg.de
#   - pip install -r requirements.txt already done

set -e

ORCA_DIR="${1:-}"
WITH_MOLSIMPLIFY=false
for arg in "$@"; do
    [[ "$arg" == "--with-molsimplify" ]] && WITH_MOLSIMPLIFY=true
done

# --- OPI ---
if [[ -z "$ORCA_DIR" ]]; then
    echo "Usage: bash install_server_deps.sh /path/to/orca [--with-molsimplify]"
    echo "  /path/to/orca  — directory containing the ORCA binary"
    exit 1
fi

echo "==> Installing OPI from $ORCA_DIR"
OPI_WHEEL=$(ls "$ORCA_DIR"/opi*.whl 2>/dev/null | head -1)
OPI_DIR="$ORCA_DIR/opi"
if [[ -n "$OPI_WHEEL" ]]; then
    pip install "$OPI_WHEEL"
elif [[ -d "$OPI_DIR" ]]; then
    pip install "$OPI_DIR"
else
    echo "ERROR: Could not find opi*.whl or opi/ in $ORCA_DIR"
    echo "       Check the ORCA distribution for the OPI package."
    exit 1
fi

# --- molSimplify (optional) ---
if [[ "$WITH_MOLSIMPLIFY" == true ]]; then
    echo "==> Installing molSimplify"
    ARCH=$(uname -m)
    OS=$(uname -s)
    if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
        echo "    Detected macOS Apple Silicon — using tensorflow-macos"
        pip install tensorflow-macos tensorflow-metal
    else
        echo "    Detected $OS $ARCH — using standard tensorflow"
        pip install tensorflow
    fi
    pip install molSimplify
fi

echo ""
echo "==> Done. Add to your .env:"
echo "    MCP_MODE=local"
echo "    MCP_SERVER_CMD=python server_with_product.py"
echo "    OPI_ORCA=$ORCA_DIR/orca"
