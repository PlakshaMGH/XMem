from pathlib import Path

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from dataset import EndoVisDataset
from transforms import get_transforms
from util.configuration import config, init_logger
from model.trainer import XMemTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.benchmark = True
print(f"Training on {device}.")

_, long_id = init_logger()

MAIN_FOLDER = Path("./data/EndoVis17")
TRAIN_VIDEOS_PATH = MAIN_FOLDER / "frames/train"
TRAIN_MASKS_PATH = MAIN_FOLDER / "masks/train/binary_masks"
SAVE_DIR = Path("./artifacts/saved_models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

transforms_dict = get_transforms()

train_dataset = EndoVisDataset(
    TRAIN_VIDEOS_PATH,
    TRAIN_MASKS_PATH,
    num_iterations=config.num_iterations,
    batch_size=config.batch_size,
    max_jump=20,
    num_frames=8,
    max_num_obj=1,
    transform=transforms_dict,
    subset=[1]
)

train_loader = DataLoader(
    train_dataset,
    config.batch_size,
    num_workers=config.num_workers,
    drop_last=True,
)

model = XMemTrainer(
    config,
    # logger=logger,
    save_path=SAVE_DIR / long_id,
    local_rank=0,
    world_size=1,
).train()

model.load_network("./artifacts/pretrained_weights/XMem.pth")

total_epochs = config.num_iterations
print(f"Training model for {total_epochs} epochs.")
## Train Loop
model.train()

pbar = tqdm(range(total_epochs), unit="Epoch")
for epoch in pbar:
    for data in train_loader:
        data["rgb"] = data["rgb"].cuda()
        data["first_frame_gt"] = data["first_frame_gt"].cuda()
        data["cls_gt"] = data["cls_gt"].cuda()
        data["selector"] = data["selector"].cuda()
        loss = model.do_pass(data, epoch)
        pbar.set_postfix(loss=loss)

    if epoch % 100 == 0:
        model.save_network(epoch)

model.save_network(total_epochs)