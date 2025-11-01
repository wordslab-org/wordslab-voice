#!/bin/bash

# ollama llm warm up

# stt        # Needs 2.5GB of vram
# tts        # Needs 5.3GB of vram
# llm        # Needs 11GB of vram for gemma3:12b => OK with 24GB

ollama run gemma3:12b hello --verbose

# This script starts unmute with Docker commpose and uses the local ollama instance for the llm

export HUGGING_FACE_HUB_TOKEN=hf_eq...TO

cd ./unmute
docker compose up