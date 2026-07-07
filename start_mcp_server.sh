#!/bin/bash
# MCP server startup
if ! command -v python3 >/dev/null 2>&1; then
  apt-get install -y -qq python3 python3-venv 2>/dev/null
fi
export LD_LIBRARY_PATH=/data/zhang/ollama/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export ORCA_DIR=/data/zhang/ollama/orca/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg
export OPI_ORCA=$ORCA_DIR
export PATH=$ORCA_DIR:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export XTB_DIR=/data/zhang/ollama/xtb-dist
export XTBEXE=$XTB_DIR/bin/xtb
export XTBPATH=$XTB_DIR/share/xtb
export PATH=$XTB_DIR/bin:$PATH
cd /data/zhang/ollama/nbo_agent
exec /data/zhang/ollama/QCagent_venv/bin/python server_with_product.py
