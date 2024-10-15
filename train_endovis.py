from pathlib import Path

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as distributed

import typer

from dataset import EndoVisDataset
from transforms import get_transforms
from util.configuration import config, init_logger
from model.trainer import XMemTrainer

def main(subset_string: str = "1,2,3,4,5,6,7,8"):

    # Init distributed environment
    distributed.init_process_group(backend="nccl")
    print(f'CUDA Device count: {torch.cuda.device_count()}')

    if config.benchmark:
        torch.backends.cudnn.benchmark = True

    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)

    print(f'I am rank {local_rank} in this world of size {world_size}!')

    # Model related
    if local_rank == 0:
        # Logging
        logger, long_id = init_logger(do_logging=True, existing_run=None)

        # Construct the rank 0 model
        model = XMemTrainer(config.to_dict(), logger=logger, 
                        save_path=Path('saves', long_id, 'iteration') if long_id is not None else None, 
                        local_rank=local_rank, world_size=world_size).train()
    else:
        # Construct model for other ranks
        model = XMemTrainer(config.to_dict(), local_rank=local_rank, world_size=world_size).train()
        logger = None

    # loading pretrained weights
    model.load_network("./artifacts/XMem.pth")

    MAIN_FOLDER = Path("../data")
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
        subset=[int(i) for i in subset_string.split(',')]
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    train_loader = DataLoader(
        train_dataset,
        config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        sampler=train_sampler,
        pin_memory=True,
    )

    total_iterations = config.num_iterations
    iteration_per_gpu = total_iterations // world_size
    print(f"Training model for {total_iterations} total iterations and {iteration_per_gpu} iterations per GPU.")
    ## Train Loop
    model.train()

    iter_pbar = tqdm(train_loader, disable=local_rank!=0)
    for iteration, data in enumerate(iter_pbar, start=1):
        train_sampler.set_epoch(iteration) 
        total_loss: float = model.do_pass(data, iteration*world_size)

        # update progress bar
        iter_pbar.set_postfix(total_loss=total_loss)

    distributed.destroy_process_group()

    # end the logger
    if logger:
        logger.finish()

    print("Training complete!")

if __name__ == "__main__":
    typer.run(main)