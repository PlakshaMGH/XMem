import torch
from pathlib import Path
from dataset import EndoVisDataset, im_mean
from util.configuration import config, init_logger
from util.helpers import reseed
from model.trainer import XMemTrainer
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.benchmark = True
print(f"Training on {device}.")

MAIN_FOLDER = Path("./data/EndoVis17")
TRAIN_VIDEOS_PATH = MAIN_FOLDER / "frames/train"
TRAIN_MASKS_PATH = MAIN_FOLDER / "masks/train/binary_masks"
SAVE_DIR = Path("./artifacts/saved_models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Set seed to ensure the same initialization
reseed(42)

logger, long_id = init_logger()

# These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
pair_im_transform = transforms.Compose([
    transforms.ColorJitter(0.01, 0.01, 0.01, 0),
])

pair_imgt_transform = transforms.Compose([
    transforms.RandomAffine(15, shear=10),
])

# These transform are the same for all pairs in the sampled sequence
all_im_transform = transforms.Compose([
    transforms.ColorJitter(0.1, 0.03, 0.03, 0),
    transforms.RandomGrayscale(0.05),
])

all_imgt_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00))
])

train_dataset = EndoVisDataset(
    TRAIN_VIDEOS_PATH,
    TRAIN_MASKS_PATH,
    num_iterations=100,#config["iterations"],
    batch_size=2,#config["batch_size"],
    max_jump=config["max_skip_value"],
    num_frames=config["num_frames"],
    max_num_obj=config["max_num_obj"],
    im_tran=pair_im_transform,
    imgt_tran=pair_imgt_transform,
    all_im_tran=all_im_transform,
    all_imgt_tran=all_imgt_transform,
)

train_loader = DataLoader(
    train_dataset,
    config["batch_size"],
    num_workers=config["num_workers"],
    drop_last=True,
)

model = XMemTrainer(
    config,
    logger=logger,
    save_path=SAVE_DIR / long_id,
    local_rank=0,
    world_size=1,
).train()

model.load_network("./artifacts/pretrained_weights/XMem.pth")

total_epochs = config["iterations"]
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