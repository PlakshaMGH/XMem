import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage import io
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from util.configuration import config, init_logger
from model.network import XMem
from inference.inference_core import InferenceCore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

im_normalization = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

logger, _ = init_logger(long_id=False, existing_run=None)

def resize_mask(mask, size, num_obj):
    mask = mask.unsqueeze(0).unsqueeze(0)
    h, w = mask.shape[-2:]
    min_hw = min(h, w)
    mask = F.interpolate(mask, (int(h/min_hw*size), int(w/min_hw*size)), 
                mode='nearest')
    mask = mask.squeeze(0,1).long()
    return F.one_hot(mask, num_classes=num_obj+1).permute(2,0,1).float()

def calculate_iou(pred, target):
    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def test_patient(frames_path, masks_path, processor, size=-1, num_obj=1):
    image_paths = sorted(frames_path.glob('*.png'))
    mask_paths = sorted(masks_path.glob('*.png'))
    
    total_iou = 0
    num_frames = len(image_paths)

    if size < 0:
        im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])
    else:
        im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
            transforms.Resize(size, interpolation=InterpolationMode.BILINEAR, antialias=False),
        ])

    # First frame
    first_frame = io.imread(image_paths[0])
    first_mask = io.imread(mask_paths[0])
    og_shape = first_frame.shape[:2]

    frame_torch = im_transform(first_frame).to(device)
    
    if size > 0:
        first_mask = torch.tensor(first_mask).to(device)
        first_mask = resize_mask(first_mask, size, num_obj)
    else:
        first_mask = torch.from_numpy(first_mask).unsqueeze(0).float().to(device)
    
    first_mask = first_mask[1:]  # Remove background
    
    with torch.no_grad():
        prediction = processor.step(frame_torch, first_mask)
    
    # Calculate IoU for first frame
    pred_mask = torch.argmax(prediction, dim=0).cpu().numpy()
    true_mask = io.imread(mask_paths[0])
    iou = calculate_iou(pred_mask, true_mask)
    total_iou += iou

    # Process remaining frames
    for image_path, mask_path in tqdm(zip(image_paths[1:], mask_paths[1:]), total=num_frames-1):
        frame = io.imread(image_path)
        frame_torch = im_transform(frame).to(device)
        
        with torch.no_grad():
            prediction = processor.step(frame_torch)
        
        # Upsample to original size if needed
        if size > 0:
            prediction = F.interpolate(prediction.unsqueeze(1), og_shape, mode='nearest', align_corners=False)[:,0]
        
        pred_mask = torch.argmax(prediction, dim=0).cpu().numpy()
        true_mask = io.imread(mask_path)
        
        iou = calculate_iou(pred_mask, true_mask)
        total_iou += iou

    return total_iou / num_frames

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
        model = XMem(config.to_dict()).to(device).eval()
        model_weights = torch.load(model_file)
        model.load_weights(model_weights, init_as_zero_if_needed=True)

        processor = InferenceCore(model, config=config.to_dict())

        # Test for each patient
        patient_ious = []
        for patient_id in TEST_VIDEOS_PATH.iterdir():
            if patient_id.is_dir():
                frames_path = TEST_VIDEOS_PATH / patient_id
                masks_path = TEST_MASKS_PATH / patient_id
                
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
        print(f"Best model saved to {best_save_path}")
        print(f"Best average IoU: {best_avg_iou:.4f}")
    else:
        print("No best model found.")

if __name__ == "__main__":
    main()