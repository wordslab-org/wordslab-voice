#!/bin/bash

# This script starts unmute without Docker to reuse a maximum of disk space

source /root/.bashrc 
cd /home/workspace/wordslab-voice/kyutai-unmute

./unmute/dockerless/start_frontend.sh
./unmute/dockerless/start_backend.sh
# ./unmute/dockerless/start_llm.sh      # replaced by the running ollama server
./unmute/dockerless/start_stt.sh        # Needs 2.5GB of vram
./unmute/dockerless/start_tts.sh        # Needs 5.3GB of vram