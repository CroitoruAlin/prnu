import os
import time
import json
import yaml
import gc
import numpy as np
from datasets import load_from_disk, Features, Value, Array2D, Sequence, Dataset
from einops import rearrange
import torch.nn.functional as F
from torch.utils.data import DataLoader
from processors.dataset import DeviceRegistrationDataset
from basicsr.models.archs.restormer_arch import Restormer
from basicsr.models.network_unet import UNetRes
import torch
from tqdm import tqdm
from prnu import extract_single_drunet, extract_single, aligned_cc, aligned_cc_torch
import prnu
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def filter_dataset_by_resolution(dataset, resolution):
    cols = set(dataset.column_names)
    keep_cols = [c for c in ["device_id", "resolutions", "prnu"] if c in cols]
    ds = dataset.remove_columns([c for c in cols if c not in keep_cols])
    idx = np.nonzero(np.asarray(ds["resolutions"]) == resolution)[0].tolist()
    return ds.select(idx)


def filter_dataset_by_list_of_devices(dataset, device_list):
    cols = set(dataset.column_names)
    keep_cols = [c for c in ["device_id", "resolutions", "prnu"] if c in cols]
    ds = dataset.remove_columns([c for c in cols if c not in keep_cols])
    devs = np.asarray(ds["device_id"])
    mask = np.isin(devs, np.asarray(device_list))
    idx = np.nonzero(mask)[0].tolist()
    return ds.select(idx)


def _stack_prnu_fast(ds, flat_shape=None):

    prnus = None
    
    if flat_shape is not None:
        features = ds.features.copy()
        features["prnu"] = Array2D(shape=(flat_shape,), dtype="float32")
        try:
            ds_fast = ds.cast(Features(features))
            ds_fast = ds_fast.with_format("numpy", columns=["prnu"])
            prnus = ds_fast["prnu"]  
        except Exception:
            prnus = None

    
    if prnus is None:
        try:
            col = ds.data.column("prnu").combine_chunks()
            arr = col.to_numpy()  
            if arr.dtype == object:
                prnus = np.stack(arr.tolist(), axis=0)
            else:
                prnus = arr
        except Exception:
            prnus = None

    if prnus is None:
        ds_np = ds.with_format("numpy", columns=["prnu"])
        prnus = np.stack(ds_np["prnu"], axis=0)
    return prnus, ds['device_id']


class NoiseExtractor:

    def __init__(self):
        with open("configs/config.json", "r") as f:
            self.config = json.load(f)
        self._create_denoiser()

        

    def _create_denoiser(self):
        if self.config.get('denoiser_type') == 'restormer':
            network_config = yaml.load(open("configs/GaussianColorDenoising_Restormer.yml", mode='r'), Loader=Loader)
            network_config['network_g'].pop('type', None)
            self.model = Restormer(**network_config['network_g'])
            weights_path = self.config['denoiser_path']
            self.model.load_state_dict(torch.load(weights_path)['params'])
        else:
            self.model = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                                 downsample_mode="strideconv", upsample_mode="convtranspose")
            self.model.load_state_dict(torch.load(self.config['denoiser_path']), strict=True)
            self.model.eval()
            for _, v in self.model.named_parameters():
                v.requires_grad = False

    def extract_noise(self, image_paths, gt):
        self.model.to("cuda")
        final_scores = []

        
        for r in self.config['resolutions']:

            t0 = time.time()
            dataset = DeviceRegistrationDataset(image_paths, gt if gt is not None else ['test']*len(image_paths), r)
            

            dataloader = DataLoader(dataset, num_workers=self.config['num_workers'],
                                    batch_size=self.config['batch_size'], shuffle=False)
            print("Data setup time:", time.time() - t0)

            resolution_scores = []
            all_query_devices = []

            for images, gt_device in tqdm(dataloader):
                if self.config.get('denoiser_type') == 'drunet':
                    query_noise = extract_single_drunet(images, self.model, 50, 50)
                else:
                    images = images/255.
                    query_noise = extract_single(images, self.model, 50, 50)
                for i, query in enumerate(query_noise):
                    yield {"device_id": gt_device[i], "resolutions": int(r), "query": query.astype(np.float32)}

                


if __name__ == "__main__":
    noise_extractor = NoiseExtractor()
    t0 = time.time()
    train_devices = np.load("train_devices.npy")
    image_dataset = load_from_disk("../datasets/PRNU/", keep_in_memory=False).filter(lambda sample: "view_1" in sample['view'] and sample["device"] in train_devices)
    gt_devices = list(image_dataset['device'])
    image_paths = [os.path.join("../datasets/PRNU/", image_path) for image_path in list(image_dataset['image_path'])]
    

    features = Features({
            "device_id": Value("string"),
            "resolution":Value("int32"),
            "query": Sequence(Sequence(Value("float32")))
        })
    ds = Dataset.from_generator(generator=lambda: noise_extractor.extract_noise(image_paths, gt=gt_devices), features=features, writer_batch_size =400, keep_in_memory=False)
    os.makedirs(noise_extractor.config['output_training_queries'], exist_ok=True)
    ds.save_to_disk(noise_extractor.config['output_training_queries'])

