Original Code: https://github.com/hkchengrex/XMem

# XMem on Surgical Videos

Fine-tuning XMem on different robotic surgery datasets for different segmentation tasks like tools segmentation (binary and multi-class) and anatomy segmentation.

Different Datasets:
- [EndoVis17](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/)
- [EndoVis18](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/)
- MGH Right Lower Lobe Pulmonary Artery (Private)

Changes in the code are done to fine-tune the XMem model with existing pre-trained weights on the above datasets, and do testing on the test splits. The corresponding files for training and testing are `train_endovis.py` and `test_endovis.py`.

We also use Weights and Biases to log and track experiments, to run the training and testing scripts, follow the instructions below:

```bash
wandb login <wandb-api-key>

# Training on One Patient
torchrun --nproc_per_node=2 train_endovis.py --subset-string "1,2,3,4" --run-name "Patient_1" \
    --run-id "e18bin-p1" --project-name "DataVar_XMem_E18_Bin"

# Testing on Test Split
python test_endovis.py --subset-string "17,18,19,20" --train-set "1" \
    --run-id "e18bin-p1" --project-name "DataVar_XMem_E18_Bin"
```

Summary:
* `--nproc_per_node`: Number of GPUs to use for training
* `--subset-string`: Patient IDs to use for training, the folder names must have the number as part of the folder name. For example, we want to train on four patients, the folder names should be *instrument_dataset_01*, *instrument_dataset_02*, *instrument_dataset_03*, *instrument_dataset_04*. The subset string should be **1,2,3,4**. Same for testing.
* `--train-set`: The training set which was used for training the model which is being used for testing, this is used for logging the best model name.
* `--run-id`: The unique ID of the run in the Weights and Biases project.
* `--project-name`: Name of the Weights and Biases project.