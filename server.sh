#!/usr/bin/env bash
cd /
GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone https://github.com/DDantalion/CSE232B.git
cd CSE232B
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda update -n base -c defaults conda -y
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba
cd vllm_cache_bench
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
chmod +x setup_env.sh
./setup_env.sh
