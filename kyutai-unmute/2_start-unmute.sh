#!/bin/bash

# ollama llm warm up

# stt        # Needs 2.5 GB of vram
# tts        # Needs 5.3 GB of vram
# llm        # Needs 11  GB of vram for gemma3:12b 

# VRAM used: 21078MiB /  24564MiB

ollama run gemma3:12b hello --verbose

# This script starts unmute with Docker commpose and uses the local ollama instance for the llm

export HUGGING_FACE_HUB_TOKEN=hf_eq...TO

cd ./unmute
docker compose up