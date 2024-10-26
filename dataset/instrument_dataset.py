import os
import re
from os import path

from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision import transforms

from dataset.range_transform import im_normalization, im_mean
from util.helpers import reseed

class EndoVisDataset(Dataset):
    """
    For each sequence:
    - Pick `num_frames` frames
    - Pick `num_objects` objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, im_root, gt_root, max_jump,
                 num_iterations=4000, batch_size=8,
                 subset=None, num_frames=8, max_num_obj=1,
                 transform=None):
        
        self.im_root = im_root # Root directory for Images
        self.gt_root = gt_root # Root directory for ground truth data
        self.max_jump = max_jump # Maximum distance between frames
        self.num_iterations = num_iterations # Number of iterations
        self.batch_size = batch_size # Batch size
        self.num_frames = num_frames # Number of frames to be sampled
        self.max_num_obj = max_num_obj # Maximum number of objects
        self.transform = transform

        # Pre-filtering
        self.subset = subset
        vid_list = sorted(os.listdir(self.im_root))
        self.videos, self.frames = self._filter_videos(vid_list)
        self.batches = self._create_batches(self.videos)

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

        if self.transform:
            self.single_image_T = transform["single_image"] # RGB Image Transformations
            self.pair_image_gt_T = transform["pair_image_gt"] # RGB Image and Ground Truth Transformations
            self.seq_image_T = transform["seq_image"] # RGB Image Transformations for all frames
            self.seq_image_gt_T = transform["seq_image_gt"] # RGB Image and Ground Truth Transformations for all frames

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

    # Pre-filtering based on subset provided
    def _filter_videos(self, video_list):

        # Initialize lists for storing video and frame information
        videos = [] # List of videos
        frames_dict = {} # Dictionary mapping video to its frames

        # Pre-filtering
        for vid in video_list:
            # If looking for a subset of videos only
            if self.subset is not None:
                # get the number from the video name
                number = int(re.search(r'\d+', vid).group())
                if number not in self.subset:
                    continue
            
            # List frames in each video directory
            frames = sorted(os.listdir(os.path.join(self.gt_root, vid)))
            
            # Check if the video has enough frames
            if len(frames) < self.num_frames:
                continue
            
            frames_dict[vid] = frames
            videos.append(vid)
        
        return videos, frames_dict
    
    def _create_batches(self, videos_list):
        batches = []

        while len(batches) < self.num_iterations*self.batch_size:
            video = np.random.choice(videos_list)
            frames = self.frames[video]
            num_frames = self.num_frames
            length = len(frames)
            this_max_jump = min(len(frames), self.max_jump)

            frames_idx = [np.random.randint(length)]
            acceptable_set = set(
                range(max(0, frames_idx[-1]-this_max_jump), 
                      min(length, frames_idx[-1]+this_max_jump+1))
                ).difference(set(frames_idx))
            
            while(len(frames_idx) < num_frames):
                idx = np.random.choice(list(acceptable_set))
                frames_idx.append(idx)
                new_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1)))
                acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))

            frames_idx = sorted(frames_idx)

            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            batches.append((video, frames_idx))

        return batches
            
    def _image_transform(self, frames, frames_idx, vid_im_path, vid_gt_path):
        sequence_seed = np.random.randint(2147483647)
        images = []
        masks = []
        target_objects = []
        for f_idx in frames_idx:
            jpg_name = frames[f_idx]
            png_name = frames[f_idx]
            
            this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
            this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')

            if self.transform:
                # Same Transformation for Image/Mask pairs throughout current sequence
                # due to same seed, the same transformation is applied to all frames
                reseed(sequence_seed)
                this_im = self.seq_image_gt_T[0](this_im)
                this_im = self.seq_image_T(this_im)
                reseed(sequence_seed)
                this_gt = self.seq_image_gt_T[1](this_gt)

                # Different Transformation for Image/Mask pairs in a sequence
                # due to different seed in each iteration, different transformation is applied to each frame
                pairwise_seed = np.random.randint(2147483647) # generate a random seed for each frame
                reseed(pairwise_seed)
                this_im = self.pair_image_gt_T[0](this_im)
                this_im = self.single_image_T(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_image_gt_T[1](this_gt)


            this_im = self.final_im_transform(this_im)
            this_gt = np.array(this_gt)

            images.append(this_im)
            masks.append(this_gt)

        return images, masks
    
    def __getitem__(self, idx):

        video, frames_idx = self.batches[idx]
        info = {}
        info['name'] = str(video)
        info['frames'] = [str(i) for i in frames_idx]
        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]

        images, masks = self._image_transform(frames, frames_idx, vid_im_path, vid_gt_path)

        images = torch.stack(images, 0)

        labels = np.unique(masks[0])
        # Remove background
        labels = labels[labels!=0]

        if len(labels) == 0:
            new_idx = np.random.choice(len(self.batches))
            return self.__getitem__(new_idx)
        else:
            target_objects = labels.tolist()

        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)

        info['num_objects'] = len(target_objects)

        masks = np.stack(masks, 0)
        h, w = masks.shape[1:]

        # Generate one-hot ground-truth
        cls_gt = np.zeros((self.num_frames, h, w), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, h, w), dtype=np.int64)
        for i, l in enumerate(target_objects):
            this_mask = (masks==l)
            cls_gt[this_mask] = i+1
            first_frame_gt[0,i] = (this_mask[0])
        cls_gt = np.expand_dims(cls_gt, 1)

        # 1 if object exist, 0 otherwise
        selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
        selector = torch.FloatTensor(selector)

        data = {
            'rgb': images,
            'first_frame_gt': first_frame_gt,
            'cls_gt': cls_gt,
            'selector': selector,
            'info': info,
            'og_masks': masks
        }

        return data

    def __len__(self):
        return len(self.batches)