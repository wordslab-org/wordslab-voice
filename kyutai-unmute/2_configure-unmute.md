# Unmute repo configuration for local execution

1. Edit ./unmute/dockerless/start_frontend.sh

Replace the last line

> pnpm dev

with

> HOST=0.0.0.0 PORT=$USER_APP1_PORT pnpm dev

Then access the frontend at: $USER_APP1_URL

2. Make sure that no app is already launched using the following ports

- 8000 - Backend 
- 8090 - STT
- 8089 - TTS

3. Prepare a LLM server with the local ollama server

> ollama run gemma3:12b

4. Edit ./unmute/dockerless/start_backend.sh

Replace the last line

> uv run uvicorn unmute.main_websocket:app --reload --host 0.0.0.0 --port 8000 --ws-per-message-deflate=false

with

> LLM_SERVER="http://127.0.0.1:11434" KYUTAI_LLM_MODEL="gemma3:12b" KYUTAI_LLM_API_KEY="ollama" uv run uvicorn unmute.main_websocket:app --reload --host 0.0.0.0 --port 8000 --ws-per-message-deflate=false

Note : all URLs are configured in

> ./unmute/unmute/kyutai_constants.py


