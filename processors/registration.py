import  prnu
from prnu.functions import extract_single, model_creation, noise_extract_compact, rgb2gray, wiener_dft, zero_mean_total, noise_extract_compact_drunet
from tqdm import tqdm
import argparse
import os
import wandb
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import csv
import gc
import torch
from datasets import Dataset, Value, Sequence, Features, concatenate_datasets, load_from_disk
from processors.dataset import DeviceRegistrationDataset
import json
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from basicsr.models.archs.restormer_arch import Restormer
from basicsr.models.network_unet import UNetRes
import shutil
class Registration():

    def __init__(self):
        # Load JSON config
        with open("configs/registration.json", "r") as f:
            self.config = json.load(f)
            self._create_denoiser()
        
        
    def _create_denoiser(self):
        if self.config['denoiser_type'] == 'restormer':
            network_config = yaml.load(open("configs/GaussianColorDenoising_Restormer.yml", mode='r'), Loader=Loader)
            s = network_config['network_g'].pop('type')
            self.model = Restormer(**network_config['network_g'])
            weights_path = self.config['denoiser_path']
            self.model.load_state_dict(torch.load(weights_path)['params'])
        else:
            self.model = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                    downsample_mode="strideconv", upsample_mode="convtranspose")
            self.model.load_state_dict(torch.load(self.config['denoiser_path']), strict=True)
            self.model.eval()
            for k, v in self.model.named_parameters():
                v.requires_grad = False


    def _prnu_extraction(self, image_paths, devices):
        dataloaders = {}
        for resolution in self.config['resolutions']:
            dataset = DeviceRegistrationDataset(image_paths, devices, resolution)
            dataloader = DataLoader(dataset, num_workers = self.config['num_workers'], batch_size = self.config['batch_size'], shuffle=False)
            dataloaders[resolution] = dataloader
        for resolution, dataloader in dataloaders.items():
            noises = []
            all_prnus_devices = []
            for images, device_name in  dataloader:
                if self.config['denoiser_type'] == 'restormer':
                    normalized_images=images/255.
                    noise_estimated = noise_extract_compact((normalized_images, self.model, 50, 50))
                else:
                    noise_estimated = noise_extract_compact_drunet((images, self.model, 100, 100))
                noises.extend(noise_estimated.cpu().numpy()/255.)
                all_prnus_devices.extend(device_name)
            

            RPsum = np.zeros(noises[0].shape, np.float32)
            for noise in noises:    
                RPsum += noise
            noises = np.array(noises)
            K = RPsum / len(noises)
            K = rgb2gray(K)
            K = zero_mean_total(K)
            K = wiener_dft(K, K.std(ddof=1)).astype(np.float32)
            yield {"device_id": all_prnus_devices[0], "device_name": all_prnus_devices[0], 'resolutions': resolution, "prnu": K}
    
    def _create_dataset(self, image_paths, devices, persist=False):
        features = Features({
                "device_id": Value("string"),
                "device_name": Value("string"),
                'resolutions':Value("int32"),
                "prnu": Sequence(Sequence(Value("float32")))
            })
        if not os.path.exists(self.config["output_database_dir"]):
            os.makedirs(self.config["output_database_dir"], exist_ok=False)
            ds = Dataset.from_generator(generator=lambda: self._prnu_extraction(image_paths, devices), features=features, writer_batch_size = self.config["writer_batch_size"], keep_in_memory=False)
            if persist:
                ds.save_to_disk(self.config["output_database_dir"])
            return ds
        elif is_hf_dataset_dir(self.config["output_database_dir"]):
            new_ds = Dataset.from_generator(generator=lambda: self._prnu_extraction(image_paths, devices), features=features, writer_batch_size = self.config["writer_batch_size"], keep_in_memory=False)
            if persist:
                old_ds = load_from_disk(self.config["output_database_dir"], keep_in_memory=False)
                merged_ds = concatenate_datasets([new_ds, old_ds])
                merged_ds.save_to_disk(self.config["output_database_dir"]+"_tmp")
                shutil.rmtree(self.config["output_database_dir"])
                os.rename(self.config["output_database_dir"]+"_tmp", self.config["output_database_dir"])
                return merged_ds
            return new_ds
        else:
            new_ds = Dataset.from_generator(generator=lambda: self._prnu_extraction(image_paths, devices), features=features, writer_batch_size = self.config["writer_batch_size"], keep_in_memory=False)
            if persist:
                new_ds.save_to_disk(self.config["output_database_dir"])
            return new_ds

    def register_device(self, folder_images, device_name, persist=False):
        image_paths = []
        devices = []
        for image in os.listdir(folder_images):
            if not is_png_truncated(image):
                image_paths.append(os.path.join(root_images, image))
                devices.append(device_name)

        return self._create_dataset(image_paths, devices, persist)

   

    def register_multiple_devices(self, root_folder_devices, individual_persist = False):
        self.model.to("cuda")
        if is_hf_dataset_dir(root_folder_devices):
            dataset = load_from_disk(root_folder_devices)
            dataset = filter_dataset_for_prnu_estimation(dataset)
            unique_devices = list(set(list(dataset['device'])))
            list_datasets = []
            for device in tqdm(unique_devices):
                device_dataset = filter_dataset_by_device(dataset, device)
                image_paths = [os.path.join(root_folder_devices, image_path) for image_path in device_dataset['image_path']]
                devices = device_dataset['device']
                list_datasets.append(self._create_dataset(image_paths, devices, persist=individual_persist))
            if not os.path.exists(self.config["output_database_dir"]):
                new_ds = concatenate_datasets(list_datasets)    
                os.makedirs(self.config["output_database_dir"], exist_ok=False)
                new_ds.save_to_disk(self.config["output_database_dir"])
            elif is_hf_dataset_dir(self.config["output_database_dir"]):
                old_ds = load_from_disk(self.config["output_database_dir"], keep_in_memory=False)
                merged_ds = concatenate_datasets(list_datasets+[old_ds])
                print(merged_ds['device_id'])
                merged_ds.save_to_disk(self.config["output_database_dir"]+"_tmp")
                shutil.rmtree(self.config["output_database_dir"])
                os.rename(self.config["output_database_dir"]+"_tmp", self.config["output_database_dir"]) 
            else:
                new_ds = concatenate_datasets(list_datasets)
                new_ds.save_to_disk(self.config["output_database_dir"])
        self.model.to("cpu")

def is_hf_dataset_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    has_info = os.path.isfile(os.path.join(path, "dataset_info.json"))
    has_state = os.path.isfile(os.path.join(path, "state.json"))
    has_arrow = any(f.endswith(".arrow") for f in os.listdir(path))
    if has_info and has_state and has_arrow:
        return True
    return False


def filter_dataset_by_device(dataset, device):
    devices = list(dataset['device'])
    indices_select = [i for i, d in enumerate(devices) if d==device]
    return dataset.select(indices_select)


def filter_dataset_for_prnu_estimation(dataset):
    views = list(dataset['view'])
    indices_select = [i for i,view in enumerate(views) if 'view_1' in view ]
    return dataset.select(indices_select)


def is_png_truncated(filename):
    with open(filename, 'rb') as f:
        f.seek(-12, 2) 
        end = f.read()
        return b'IEND' not in end
   
if __name__ == "__main__":
    registration_service = Registration()
    registration_service.register_multiple_devices("/home/biodeep/alin/datasets/PRNU")


