import os
import re
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2
import typer

from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU

from util.configuration import test_config, init_logger
from model.network import XMem
from inference.data.mask_mapper import MaskMapper
from inference.data.video_reader import VideoReader
from inference.inference_core import InferenceCore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def torch_prob_to_one_hot_torch(prob, num_classes):
    mask = torch.argmax(prob, dim=0)
    numpy_mask = mask.cpu().numpy()
    mask = F.one_hot(mask, num_classes=num_classes+1).permute(2, 0, 1)
    return mask, numpy_mask

def add_background_mask(mask):
    # Create the background mask
    background = torch.ones_like(mask[0])  # Start with all ones
    for channel in mask:
        background = background & (channel == 0)  # Set to 0 where any channel has 1
    
    # Combine the background mask with the original mask
    return torch.cat([background.unsqueeze(0), mask], dim=0)

def color_map(pred_mask: np.ndarray, gt_mask: np.ndarray):

    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()

    if pred_mask.ndim == 3: # one hot encoded
        pred_mask = np.argmax(pred_mask, axis=0)
    if gt_mask.ndim == 3: # one hot encoded
        gt_mask = np.argmax(gt_mask, axis=0)

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

def create_video_from_frames(video_frames, video_name):
    save_folder = Path("./saved_videos")
    save_folder.mkdir(parents=True, exist_ok=True)
    video_path = save_folder / f"{video_name}.mp4"
    sample_frame = list(video_frames.values())[0]
    size1, size2, _ = sample_frame.shape
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'avc1'), 1, (size2, size1), True)
    for frame_name, frame in sorted(video_frames.items(), key=lambda x: x[0]):
        out_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(out_img)
    out.release()

    return str(video_path)

def create_numpy_video_from_frames(video_frames):
    numpy_video = []
    for frame_name, frame in sorted(video_frames.items(), key=lambda x: x[0]):
        numpy_video.append(frame)
    return np.stack(numpy_video, axis=0)


def test_patient(frames_path, masks_path, processor, mapper, size=-1):
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
        mask = add_background_mask(mask.cpu())
        gt_masks.append(mask)

        color_mask = color_map(prob_numpy_mask, mask)
        video_frames[frame_name] = cv2.addWeighted(data['original_img'], 1, color_mask, 0.5, 0)

    # Calculate IoU and Dice
    pred_masks = torch.stack(pred_masks, dim=0)
    gt_masks = torch.stack(gt_masks, dim=0)
    meanIoU = IoU(pred_masks, gt_masks)
    meanDice = Dice(pred_masks, gt_masks)

    return meanIoU, meanDice, video_frames

def main(subset_string: str = "9,10", train_set: str = "1", run_id: str = "e17type1", project_name: str = "DataVar_XMem_E17_Type"):

    logger = init_logger(None, run_id, project_name, test_config, do_logging=False)
    subset_list = [int(i) for i in subset_string.split(',')]

    # Load the latest model
    saves_dir = Path("./saves")
    model_dirs = sorted([d for d in saves_dir.iterdir() if d.is_dir()], reverse=True)
    if not model_dirs:
        raise ValueError("No model directories found in ./saves")
    
    model_dir = saves_dir / run_id
    model_files = sorted(model_dir.glob("*.pth"), reverse=False)
    
    if not model_files:
        raise ValueError(f"No .pth files found in {model_dir}")

    best_model_path = None
    best_video_frames = None
    best_avg_iou = 0

    MAIN_FOLDER = Path("../data")
    TEST_VIDEOS_PATH = MAIN_FOLDER / "frames/test"
    TEST_MASKS_PATH = MAIN_FOLDER / "masks/test/type_masks"

    torch.autograd.set_grad_enabled(False)

    for model_file in model_files:
        print(f"Testing model: {model_file}")
        iteration_num = int(model_file.stem.split("_")[-1].split(".")[0])
        
        # Load model
        network = XMem(test_config.to_dict(), model_file).to(device).eval()

        # Test for each patient
        patient_ious = []
        video_frames_dict = {}
        for patient_id in TEST_VIDEOS_PATH.iterdir():
            # Initialize mapper for each patient for new labels in each patient
            mapper = MaskMapper()
            processor = InferenceCore(network, config=test_config.to_dict())
            number = int(re.search(r'\d+', patient_id.name).group())
            if patient_id.is_dir() and (number in subset_list):
                frames_path = TEST_VIDEOS_PATH / patient_id.name
                masks_path = TEST_MASKS_PATH / patient_id.name
                print(f"Testing patient {patient_id.name}")
                
                patient_iou_per_class, _, video_frames = test_patient(frames_path, masks_path, processor, mapper, size=384)
                video_frames_dict[patient_id.name] = video_frames

                if patient_iou_per_class.ndim == 0:
                    patient_iou_per_class = patient_iou_per_class.unsqueeze(0)
                for i, iou in enumerate(patient_iou_per_class):
                    logger.log_metrics('test', f'{patient_id.name}_class_{processor.all_labels[i]}', iou, step=iteration_num)

                mean_iou = torch.mean(patient_iou_per_class)
                logger.log_metrics('test', patient_id.name, mean_iou, step=iteration_num)
                patient_ious.append(mean_iou)


        avg_iou = np.mean(patient_ious)
        logger.log_metrics('test', 'avg_iou', avg_iou, step=iteration_num)

        if avg_iou > best_avg_iou:
            best_avg_iou = avg_iou
            best_model_path = model_file
            best_video_frames = video_frames_dict

    # Save the best model
    if best_model_path:
        model_name = f"best_{train_set}_{best_avg_iou*100:.2f}.pth"
        logger.log_metrics('test', 'best_avg_iou', best_avg_iou, step=int(train_set[-1]))
        logger.log_model(best_model_path, name=model_name)
        if best_video_frames:
            for patient_id, video_frames in best_video_frames.items():
                video_path = create_video_from_frames(video_frames, patient_id)
                logger.log_video(video_path, name=patient_id)
    else:
        print("No best model found.")

if __name__ == "__main__":
    typer.run(main)