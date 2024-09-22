from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from dataset import im_mean
from util.helpers import reseed

def get_transforms():
    # Set seed to ensure the same initialization
    reseed(42)

    transforms_dict = {}
    
    # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
    transforms_dict["single_image"] = transforms.Compose([
        transforms.ColorJitter(0.01, 0.01, 0.01, 0),
    ])

    transforms_dict["pair_image_gt"] = [
        transforms.Compose([
            transforms.RandomAffine(15, shear=10, interpolation=InterpolationMode.BILINEAR, fill=im_mean) # for images
        ]),
        transforms.Compose([
            transforms.RandomAffine(15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0) # for masks
        ])
    ]

    # These transform are the same for all pairs in the sampled sequence
    transforms_dict["seq_image"] = transforms.Compose([
        transforms.ColorJitter(0.1, 0.03, 0.03, 0),
        transforms.RandomGrayscale(0.05),
    ])


    transforms_dict["seq_image_gt"] = [
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.BILINEAR) # for images
        ]),
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST) # for masks
        ])
    ]