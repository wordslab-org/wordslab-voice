#!/bin/bash

# clone the github repo
git clone https://github.com/kyutai-labs/unmute.git

# copy the custom docker compose config
cp docker-compose.yml ./unmute
mkdir -p ./unmute/traefik-config
cp tls.yml ./unmute/traefik-config

# build the containers

export HUGGING_FACE_HUB_TOKEN=hf_eq...TO

cd ./unmute
docker compose build