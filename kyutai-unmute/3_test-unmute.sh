#!/bin/bash

# Configure the services for parallel conversations
# Max 4 workers for a single backend container

cd unmute
vi services/moshi-server/configs/stt.toml 
# batch_size = 1          <--- make this bigger to support more parallel conversations
# batch_size = 4
vi services/moshi-server/configs/tts.toml 
# batch_size = 2          <--- make this bigger to support more parallel conversations
# batch_size = 4

docker compose down
docker compose build
docker compose up

# Restart the service without SSL certificates

cd unmute
vi docker-compose.yml
# Comment out the three lines
#      - "--entrypoints.websecure.http.tls=true"
#      - "traefik.http.routers.frontend.tls=true"
#      - "traefik.http.routers.backend.tls=true"

docker compose down
docker compose up

# wait until the services are ready

uv run unmute/loadtest/loadtest_client.py --server-url ws://localhost:8883/api --n-workers 4 --n-conversations 12