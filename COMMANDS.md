# Commands used during the Ablation Study

## Endovis17

### rclone

```bash
rclone copy remote:endovis/endo17/data/frames ~/workspace/data/frames --progress --transfers 32
rclone copy remote:endovis/endo17/data/masks ~/workspace/data/masks --progress --transfers 32
```

### Training
```bash
torchrun --nproc_per_node=1 train_endovis.py --subset-string "1" --run-name "Patient_1" --run-id "e17bin-p1" --project-name "DataVar_XMem_E17_Bin"
```

### Testing
```bash
python test_endovis.py --subset-string "9,10" --train-set "1"  --run-id "e17bin-p1" \
--project-name "DataVar_XMem_E17_Bin"
```
