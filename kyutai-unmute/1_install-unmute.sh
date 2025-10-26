#!/bin/bash

# This script installs the prerequisites for kyutai-labs Unmute
# https://github.com/kyutai-labs/unmute/tree/main?tab=readme-ov-file#running-without-docker

# uv is already installed in wordslab-notebooks

# cargo for Rust -> moshi-server
curl https://sh.rustup.rs -sSf | sh

# pnpm for typescript -> frontend
curl -fsSL https://get.pnpm.io/install.sh | sh -

# cuda is already installed in wordslab-notebooks

# clone the github repo
git clone https://github.com/kyutai-labs/unmute.git