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

def get_loader(subset_string: str, num_iterations: int, max_jump: int, world_size: int, local_rank: int):

    MAIN_FOLDER = Path("../data")
    TRAIN_VIDEOS_PATH = MAIN_FOLDER / "frames/train"
    TRAIN_MASKS_PATH = MAIN_FOLDER / "masks/train/type_masks"

    transforms_dict = get_transforms()

    train_dataset = EndoVisDataset(
        TRAIN_VIDEOS_PATH,
        TRAIN_MASKS_PATH,
        num_iterations=num_iterations,
        batch_size=config.batch_size,
        max_jump=max_jump,
        num_frames=config.num_frames,
        max_num_obj=config.max_num_obj,
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

    return train_sampler, train_loader

def main(subset_string: str = "1,2,3,4,5,6,7,8", run_name: str = "Patient_1",
         run_id: str = "abcd1xx4", project_name: str = "DataVar_XMem_E17_Type"):

    # Init distributed environment
    distributed.init_process_group(backend="nccl")
    print(f'CUDA Device count: {torch.cuda.device_count()}')

    if config.benchmark:
        torch.backends.cudnn.benchmark = True

    config.exp_id = run_name

    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)

    print(f'I am rank {local_rank} in this world of size {world_size}!')

    # Model related
    if local_rank == 0:
        # Logging
        logger = init_logger(run_name, run_id, project_name, config, do_logging=True)

        # Construct the rank 0 model
        model = XMemTrainer(config.to_dict(), logger=logger,
                        save_path=Path('saves', run_id, 'iteration') if run_id is not None else None, 
                        local_rank=local_rank, world_size=world_size).train()
    else:
        # Construct model for other ranks
        model = XMemTrainer(config.to_dict(), local_rank=local_rank, world_size=world_size).train()
        logger = None

    # loading pretrained weights
    model.load_network("./artifacts/XMem.pth")

    # saving the starting point as iteration 0
    if local_rank == 0:
        model.save_network(0)

    
    SAVE_DIR = Path("./artifacts/saved_models")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    total_iterations = config.num_iterations
    steps = config.steps
    max_skip_value = config.max_skip_value

    assert sum(steps) == total_iterations, "Total iterations must be equal to the sum of steps"
    assert len(max_skip_value) == len(steps), "Length of max_skip_value must be equal to the length of steps"

    iteration_per_gpu = total_iterations // world_size
    print(f"Training model for {total_iterations} total iterations and {iteration_per_gpu} iterations per GPU.")
    ## Train Loop
    model.train()

    tqdm_iter = tqdm(total=total_iterations)
    iteration = 1
    for step, max_skip in zip(steps, max_skip_value):
        print(f"Training for {step} iterations with max skip value {max_skip}")
        train_sampler, train_loader = get_loader(subset_string, step, max_skip, world_size, local_rank)

        for epoch, data in enumerate(train_loader):
            train_sampler.set_epoch(epoch) 
            total_loss: float = model.do_pass(data, iteration*world_size)
            iteration += 1
            # update progress bar
            tqdm_iter.set_postfix(total_loss=total_loss)
            tqdm_iter.update(1)
    tqdm_iter.close()

    distributed.destroy_process_group()

    # end the logger
    if logger:
        logger.finish()
    
    # save the final model
    if local_rank == 0:
        model.save_network(total_iterations-1)

    print("Training complete!")

if __name__ == "__main__":
    typer.run(main)