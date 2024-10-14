import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2

from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU

from util.configuration import test_config, init_logger
from model.network import XMem
from inference.data.mask_mapper import MaskMapper
from inference.data.video_reader import VideoReader
from inference.inference_core import InferenceCore

# logger, _ = init_logger(do_logging=False, existing_run=True) # Set to True to resume logging on latest run
mapper = MaskMapper()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def torch_prob_to_one_hot_torch(prob, num_classes):
    mask = torch.argmax(prob, dim=0)
    numpy_mask = mask.cpu().numpy()
    mask = F.one_hot(mask, num_classes=num_classes+1).permute(2, 0, 1)
    return mask[1:], numpy_mask

def color_map(pred_mask: np.ndarray, gt_mask: np.ndarray):
    # Intersection of pred_mask and gt_mask: True Positive
    true_positive = np.bitwise_and(pred_mask, gt_mask)
    # Only Pred not GT: False Positive
    false_positive = np.bitwise_and(pred_mask, np.bitwise_not(gt_mask))
    # Only GT not Pred: False Negative
    false_negative = np.bitwise_and(np.bitwise_not(pred_mask), gt_mask)

    # Colors
    green = (0, 255, 0)
    red = (255, 0, 0)
    blue = (0, 0, 255)

    # Creating Color Map Image
    h,w = pred_mask.shape[:2]
    color_map = np.zeros((h,w,3), dtype=np.uint8)
    color_map[true_positive!=0] = green
    color_map[false_positive!=0] = red
    color_map[false_negative!=0] = blue

    return color_map

def create_video_from_frames(video_frames):
    video_path = f"./saved_videos/video.mp4"
    sample_frame = list(video_frames.values())[0]
    size1, size2, _ = sample_frame.shape
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (size2, size1), True)
    for frame_name, frame in sorted(video_frames.items(), key=lambda x: x[0]):
        out_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(out_img)
    out.release()


def test_patient(frames_path, masks_path, processor, size=-1):
    vid_reader = VideoReader(frames_path, frames_path, masks_path, size=size, use_all_mask=True)

    total_iou = 0
    num_frames = len(vid_reader)
    iou_list = []
    dice_list = []
    video_frames = {}
    pred_masks = []
    gt_masks = []

    for data in tqdm(vid_reader, total=num_frames):
        frame = data['rgb'].to(device)
        original_mask = data['mask']
        info = data['info']
        frame_name = info['frame']
        shape = info['shape'] # original size
        need_resize = info['need_resize']
        idx = info['idx']

        # Map possibly non-continuous labels to continuous ones
        mask, labels = mapper.convert_mask(original_mask, exhaustive=True)
        mask = torch.Tensor(mask).to(device)
        if need_resize:
            # resize_mask requires a batch dimension (B*C*H*W) C->classes
            resized_mask = vid_reader.resize_mask(mask.unsqueeze(0))[0]
        else:
            resized_mask = mask

        if idx == 0:
            processor.set_all_labels(list(mapper.remappings.values()))
            mask_to_use = resized_mask
            labels_to_use = labels
            # IoU and Dice init
            IoU = MeanIoU(num_classes=len(processor.all_labels)+1, per_class=True, include_background=False)
            Dice = GeneralizedDiceScore(num_classes=len(processor.all_labels)+1, per_class=True, include_background=False)
        else:
            mask_to_use = None
            labels_to_use = None

        # Run the model on this frame
        with torch.no_grad():
            prob = processor.step(frame, mask_to_use, labels_to_use, end=(idx==num_frames-1))

        # Upsample to original size if needed
        if need_resize:
            prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]

        prob, prob_numpy_mask = torch_prob_to_one_hot_torch(prob, len(processor.all_labels))
        pred_masks.append(prob.cpu())
        gt_masks.append(mask.cpu())

        color_mask = color_map(prob_numpy_mask, original_mask)
        video_frames[frame_name] = cv2.addWeighted(data['original_img'], 1, color_mask, 0.5, 0)

    create_video_from_frames(video_frames)

    # Calculate IoU and Dice
    pred_masks = torch.stack(pred_masks, dim=0)
    gt_masks = torch.stack(gt_masks, dim=0)
    meanIoU = IoU(pred_masks, gt_masks)
    meanDice = Dice(pred_masks, gt_masks)

    print(meanIoU)
    print(meanDice)

    return meanIoU.item(), meanDice.item()
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
                
                patient_iou, patient_dice = test_patient(frames_path, masks_path, processor, size=384)
                patient_ious.append(patient_iou)
                print(f"Patient {patient_id.name} IoU: {patient_iou:.4f}")

        avg_iou = np.mean(patient_ious)
        print(f"Average IoU: {avg_iou:.4f}")

        if avg_iou > best_avg_iou:
            best_avg_iou = avg_iou
            best_model_path = model_file

    # Save the best model
    # if best_model_path:
    #     best_save_path = saves_dir / "best.pth"
    #     torch.save(torch.load(best_model_path), best_save_path)
    #     logger.log_model(best_save_path, name=f'best.pth')
    #     print(f"Best model saved to {best_save_path}")
    #     print(f"Best average IoU: {best_avg_iou:.4f}")
    # else:
    #     print("No best model found.")

if __name__ == "__main__":
    main()