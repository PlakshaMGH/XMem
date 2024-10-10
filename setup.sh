#!/bin/bash

sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y
sudo apt install -y unzip
sudo apt-get install -y git-lfs

sudo -v ; curl https://rclone.org/install.sh | sudo bash

rclone config 

rclone copy remote:endovis/endo17/data/frames /workspace/data/frames --progress --transfers 32
rclone copy remote:endovis/endo17/data/masks /workspace/data/masks --progress --transfers 32

git clone https://github.com/PlakshaMGH/XMem.git /workspace/XMem

cd /workspace/XMem
pip install -r requirements.txt

git lfs pull
# pip install -e .