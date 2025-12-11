# PRNU-Bench: A Novel Benchmark and Model for PRNU-Based Camera Identification

![](assets/samples_dataset.png)


### Dataset 
In our experiments, we used the PRNU-Bench dataset which is shared here:

https://huggingface.co/datasets/unibuc-cs/PRNU

If you want to use this dataset, you can clone it with:
```
git lfs install
GIT_LFS_SKIP_SMUDGE=0 git clone https://huggingface.co/datasets/unibuc-cs/PRNU
```
### Environment
Create a conda environmet with python 3.9 installed and install the dependencies from requirements.txt
```
conda create -n prnu python=3.9
conda activate prnu
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
### Models
We share the models that we used in our experiments, here:
```
git lfs install
GIT_LFS_SKIP_SMUDGE=0 git clone https://huggingface.co/acroitoru/PRNU-Bench
```

## Evaluation on PRNU-Bench
Update if necessary the paths in ```configs/config.json```.
### Run fingerprint and noise residual extraction for the evaluation phase
```
python prnu_extraction_test.py
```
### Evaluation
By default, the parameters for evaluation should be the following:
```
python test.py --prnu_signals_path test_registered_devices/ --query_path queries --ckpt_paths checkpoints/model_1024.pt,checkpoints/model_1400.pt
```
## Device Registration
To extract the PRNU fingerprint for a custom device, you can

## Training (Optional: only if you want to train your own comparison model)
The scripts are configured to work with the format of PRNU-Bench
### PRNU fingerprint extraction for the pre-training phase

1. Ensure that in config/config.json the paths are correct.
2. Run:
```
python prnu_extraction_train.py
```

### Run the training of the neural-based comparison methods.
```
python train.py --resolution <1400/1024> --prnu_signals_path <path to the previously extracted fingerprints> --query_training_path <path to the previously extracted noise residuals>
```