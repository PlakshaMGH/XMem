#!/bin/bash

sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y
sudo apt install -y unzip
sudo apt-get install -y git-lfs

sudo -v ; curl https://rclone.org/install.sh | sudo bash

rclone config create remote b2 account <b2_accound_id> key <b2_account_key>

rclone copy remote:endovis/endo17/data/frames /workspace/data/frames --progress --transfers 32
rclone copy remote:endovis/endo17/data/masks/train/type_masks /workspace/data/masks/train/type_masks --progress --transfers 32
rclone copy remote:endovis/endo17/data/masks/test/type_masks /workspace/data/masks/test/type_masks --progress --transfers 32

git clone https://github.com/PlakshaMGH/XMem.git /workspace/XMem

cd /workspace/XMem
pip install -r requirements.txt

# opencv from conda for H264 encoding
conda install -c conda-forge opencv -y

wget https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth -O ./artifacts/XMem.pth

sleep 1

wandb login <wandb_api_key>

torchrun --nproc_per_node=2 train_endovis.py --subset-string "1" --run-name "Patient_1" \
    --run-id "e17type1" --project-name "DataVar_XMem_E17_Type"

python test_endovis.py --subset-string "9, 10" --train-set "1" \
    --run-id "e17type1" --project-name "DataVar_XMem_E17_Type"

# Training on Two Patients 
torchrun --nproc_per_node=2 train_endovis.py --subset-string "1,2" --run-name "Patient_1-2" \
    --run-id "e17type2" --project-name "DataVar_XMem_E17_Type"

python test_endovis.py --subset-string "9, 10" --train-set "1-2" \
    --run-id "e17type2" --project-name "DataVar_XMem_E17_Type"

# Training on Four Patients
torchrun --nproc_per_node=2 train_endovis.py --subset-string "1,2,3,4" --run-name "Patient_1-4" \
    --run-id "e17type4" --project-name "DataVar_XMem_E17_Type"

python test_endovis.py --subset-string "9, 10" --train-set "1-4" \
    --run-id "e17type4" --project-name "DataVar_XMem_E17_Type"

# Training on Eight Patients
torchrun --nproc_per_node=2 train_endovis.py --subset-string "1,2,3,4,5,6,7,8" --run-name "Patient_1-8" \
    --run-id "e17type8" --project-name "DataVar_XMem_E17_Type"

python test_endovis.py --subset-string "9, 10" --train-set "1-8" \
    --run-id "e17type8" --project-name "DataVar_XMem_E17_Type"