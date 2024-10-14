import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU

from util.configuration import test_config, init_logger
from model.network import XMem
from inference.data.mask_mapper import MaskMapper
from inference.data.video_reader import VideoReader
from inference.inference_core import InferenceCore

logger, _ = init_logger(do_logging=False, existing_run=True) # Set to True to resume logging on latest run
mapper = MaskMapper()

def test_patient(frames_path, masks_path, processor, size=-1):
    vid_reader = VideoReader(frames_path, frames_path, masks_path, size=size)

    total_iou = 0
    num_frames = len(vid_reader)
    iou_list = []
    dice_list = []

    for data in tqdm(vid_reader, total=num_frames):
        frame = data['rgb'].cuda()
        mask = data['mask']
        info = data['info']
        frame_name = info['frame']
        shape = info['shape'] # original size
        need_resize = info['need_resize']
        idx = info['idx']

        # Map possibly non-continuous labels to continuous ones
        mask, labels = mapper.convert_mask(mask)
        mask = torch.Tensor(mask).cuda()
        if need_resize:
            # resize_mask requires a batch dimension (B*C*H*W) C->classes
            mask = vid_reader.resize_mask(mask.unsqueeze(0))[0]

        if idx == 0:
            processor.set_all_labels(list(mapper.remappings.values()))
            mask_to_use = mask
            labels_to_use = labels
            # IoU and Dice init
            IoU = MeanIoU(num_classes=len(labels_to_use))
            Dice = GeneralizedDiceScore(num_classes=len(labels_to_use))
        else:
            mask_to_use = None
            labels_to_use = None

        # Run the model on this frame
        with torch.no_grad():
            prob = processor.step(frame, mask_to_use, labels_to_use, end=(idx==num_frames-1))
        
        # Upsample to original size if needed
        if need_resize:
            prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]

        # Calculate IoU and Dice
        iou_list.append(IoU(prob, mask))
        dice_list.append(Dice(prob, mask))

    return iou_list, dice_list

def main():
    # Load the latest model
    saves_dir = Path("./saves")
    model_dirs = sorted([d for d in saves_dir.iterdir() if d.is_dir()], reverse=True)
    if not model_dirs:
        raise ValueError("No model directories found in ./saves")
    
    latest_model_dir = model_dirs[0]
    model_files = sorted(latest_model_dir.glob("*.pth"), key=os.path.getmtime, reverse=False)
    
    if not model_files:
        raise ValueError(f"No .pth files found in {latest_model_dir}")

    best_model_path = None
    best_avg_iou = 0

    MAIN_FOLDER = Path("../data")
    TEST_VIDEOS_PATH = MAIN_FOLDER / "frames/test"
    TEST_MASKS_PATH = MAIN_FOLDER / "masks/test/binary_masks"

    for model_file in model_files:
        print(f"Testing model: {model_file}")
        
        # Load model
        network = XMem(test_config.to_dict(), model_file).eval().to(device)
        processor = InferenceCore(network, config=test_config.to_dict())

        # Test for each patient
        patient_ious = []
        for patient_id in TEST_VIDEOS_PATH.iterdir():
            if patient_id.is_dir():
                frames_path = TEST_VIDEOS_PATH / patient_id.name
                masks_path = TEST_MASKS_PATH / patient_id.name
                print(f"Testing patient {patient_id.name}")
                
                patient_iou = test_patient(frames_path, masks_path, processor)
                patient_ious.append(patient_iou)
                print(f"Patient {patient_id.name} IoU: {patient_iou:.4f}")

        avg_iou = np.mean(patient_ious)
        print(f"Average IoU: {avg_iou:.4f}")

        if avg_iou > best_avg_iou:
            best_avg_iou = avg_iou
            best_model_path = model_file

    # Save the best model
    if best_model_path:
        best_save_path = saves_dir / "best.pth"
        torch.save(torch.load(best_model_path), best_save_path)
        logger.log_model(best_save_path, name=f'best.pth')
        print(f"Best model saved to {best_save_path}")
        print(f"Best average IoU: {best_avg_iou:.4f}")
    else:
        print("No best model found.")

if __name__ == "__main__":
    main()