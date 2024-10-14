"""
Dumps things to wandb and console
"""
import os
import warnings
import wandb
from torchvision import transforms

def tensor_to_numpy(image):
    image_np = (image.numpy() * 255).astype('uint8')
    return image_np

def detach_to_cpu(x):
    return x.detach().cpu()

def fix_width_trunc(x):
    return ('{:.9s}'.format('{:0.9f}'.format(x)))

class WandbLogger:
    def __init__(self, run_name, id, project_name):
        self.project_name = project_name
        self.run_name = run_name
        if self.run_name == 'NULL':
            self.run_name = 'DEBUG'

        if id is None:
            self.no_log = True
            warnings.warn('W&B Logging has been disabled.')
        else:
            self.no_log = False

            # Initialize W&B run
            self.logger = wandb.init(
                project=project_name,  # Replace with your project name
                name=run_name,
                reinit=True  # Allows multiple runs within the same script
            )

            # Define inverse transformations if needed
            self.inv_im_trans = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )

            self.inv_seg_trans = transforms.Normalize(
                mean=[-0.5 / 0.5],
                std=[1 / 0.5]
            )

            # Log Git information
            # self.log_string('git', git_info)

    def existing_run_name(self):
        return self.logger.name
        
    def get_run(self, existing_run=None):
        
        if existing_run and isinstance(existing_run, str):
            return wandb.init(project=self.project_name, id=existing_run, resume='must')
        
        if self.no_log is False: # logging is already initialized
            return self
        
        wandb_dir = os.path.join(os.getcwd(), 'wandb')
        if not os.path.exists(wandb_dir):
            # no wandb folder exists, create new run
            warnings.warn('W&B directory does not exist...')
            return None
        
        runs = [d for d in os.listdir(wandb_dir) if os.path.isdir(os.path.join(wandb_dir, d))]
        if not runs:
            warnings.warn('No runs found in W&B directory...')
            return None
        
        # find the latest run
        latest_run = max(runs, key=lambda x: os.path.getmtime(os.path.join(wandb_dir, x)))
        latest_run_id = latest_run.split('-')[-1]

        # resuming the latest run
        print(f"Resuming run: {latest_run}")
        self.logger = wandb.init(project=self.project_name, id=latest_run_id, resume='must')

        self.no_log = False

        return self

    def log_scalar(self, tag, x, step):
        if self.no_log:
            warnings.warn('W&B Logging has been disabled.')
            return
        wandb.log({tag: x}, step=step)

    def log_metrics(self, l1_tag, l2_tag, val, step, f=None):
        tag = f"{l1_tag}/{l2_tag}"
        text = f"{self.run_name} - It {step:6d} [{l1_tag.upper()}] [{l2_tag:13}]: {fix_width_trunc(val)}"
        print(text)
        if f is not None:
            f.write(text + '\n')
            f.flush()
        self.log_scalar(tag, val, step)

    def log_im(self, tag, x, step):
        if self.no_log:
            warnings.warn('W&B Logging has been disabled.')
            return
        x = detach_to_cpu(x)
        x = self.inv_im_trans(x)
        x = tensor_to_numpy(x)
        wandb.log({tag: wandb.Image(x)}, step=step)

    def log_cv2(self, tag, x, step):
        if self.no_log:
            warnings.warn('W&B Logging has been disabled.')
            return
        # x = x.transpose((2, 0, 1))  # Convert HWC to CHW
        wandb.log({tag: wandb.Image(x)}, step=step)

    def log_seg(self, tag, x, step):
        if self.no_log:
            warnings.warn('W&B Logging has been disabled.')
            return
        x = detach_to_cpu(x)
        x = self.inv_seg_trans(x)
        x = tensor_to_numpy(x)
        wandb.log({tag: wandb.Image(x)}, step=step)

    def log_gray(self, tag, x, step):
        if self.no_log:
            warnings.warn('W&B Logging has been disabled.')
            return
        x = detach_to_cpu(x)
        x = tensor_to_numpy(x)
        wandb.log({tag: wandb.Image(x, mode='L')}, step=step)

    def log_string(self, tag, x):
        print(f"{tag}: {x}")
        if self.no_log:
            warnings.warn('W&B Logging has been disabled.')
            return
        wandb.log({tag: x})

    def log_model(self, model, name):
        if not self.no_log:
            wandb.log_model(model, name=name)

    def finish(self):
        if not self.no_log:
            wandb.finish()