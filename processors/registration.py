from dataset2 import PRNUImageDataset
import extract_path
import prnu
from prnu.functions import extract_single, model_creation, noise_extract_compact, rgb2gray, wiener_dft, zero_mean_total
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
from datasets import Dataset, Value, Sequence, Features
from denoiser import create_restormer
import json
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from basicsr.models.archs.restormer_arch import Restormer

class Registration():

    def __init__(self, denoiser_path):
        # Load JSON config
        with open("config.json", "r") as f:
            self.config = json.load("configs/registration.json")
        network_config = yaml.load(open("configs/GaussianColorDenoising_Restormer.yml", mode='r'), Loader=Loader)
        s = x['network_g'].pop('type')
        model = Restormer(**x['network_g'])
        weights_path = weights['denoiser_path']
        model.load_state_dict(torch.load)

def write_dictionary(dictionary, filename, output_directory, header=None):
    output_path = os.path.join(output_directory, filename)
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        if header is not None:
            writer.writerow(header)
        for key,value in dictionary.items():
            if type(value) is list:
                writer.writerow([key]+value)
            else:
                writer.writerow([key, value])
    with open(output_path) as f:
        reader = csv.reader(f)
        columns = next(reader)
        data = list(reader)
    table = wandb.Table(data=data, columns=columns)
    wandb.log({"results_per_device": table})

def main():
    """
    Main example script. Load a subset of flatfield and natural images from a dataset.
    For each device compute the fingerprint from all the flatfield images.
    For each natural image compute the noise residual.
    Check the detection performance obtained with cross-correlation and PCE
    :return:
    """
    parser = argparse.ArgumentParser(description='PRNU Computation')
    parser.add_argument("--model_name", default="restormer")
    parser.add_argument('--result_dir', default='./results', type=str, help='Directory for results')
    parser.add_argument('--weights', default='./pretrained_models/gaussian_color_denoising', type=str, help='Path to weights')
    parser.add_argument('--model_type', default="blind", choices=['non_blind','blind'], type=str, help='blind: single model to handle various noise levels. non_blind: separate model for each noise level.')
    parser.add_argument('--sigmas', default='15,25,50', type=str, help='Sigma values')
    parser.add_argument("--sigma_test", type=int, default=15)
    parser.add_argument("--resolution",type=str, default="256")
    parser.add_argument("--batch_size",type=str, default="1")
    parser.add_argument("--channel", type=int, default=None)
    parser.add_argument("--no_prnu_images", type=int, default=-1)
    args = parser.parse_args()
    
    dir_name = f"{args.model_name}_{args.model_type}_{args.sigma_test}_{args.resolution}_{str(args.no_prnu_images)}"
    
    output_dir = os.path.join("results", dir_name)
    os.makedirs(output_dir, exist_ok=True)
    wandb.init(project="PRNU Computation", 
               config=vars(args), 
               name=dir_name)
    prnu_image_paths, prnu_devices, query_image_paths, query_devices, query_image_names, unique_devices = extract_path.extract_path_prnu_ub("/home/fl488644/datasets/PRNU", limit_prnu_images=args.no_prnu_images)#extract_path.extract_path_revision()#
    model = create_restormer(args)
    noises = []
    all_prnus_devices = []
    all_inten_scales = []
    all_saturation = []
    queries = []
    all_query_devices = []
    bs = int(args.batch_size)
    resolution = int(args.resolution)
    query_ds = PRNUImageDataset(query_image_paths, query_devices, target_resolution=int(args.resolution))
    query_dl = DataLoader(query_ds, batch_size=bs, num_workers=8)
    h, w, ch = resolution, resolution, 3
    def prnu_extraction():
        for device in unique_devices:
            noises = []
            all_prnus_devices = []
            prnu_ds = PRNUImageDataset(prnu_image_paths, prnu_devices, target_resolution=int(args.resolution), used_device=device)
            prnu_dl = DataLoader(prnu_ds, batch_size=bs, num_workers=8)
            for images, devices, inten_scale, saturation in tqdm(prnu_dl):
                normalized_images=images/255.
                noise_estimated = noise_extract_compact((normalized_images, model, 50, 50))
                noises.extend(noise_estimated/255.)
                all_prnus_devices.extend(devices)
            
        
            RPsum = np.zeros((h, w, ch), np.float32)

            for noise in noises:    
                RPsum += noise
            noises = np.array(noises)
            K = RPsum / len(noises)#np.mean(noises, axis=0)#
            print(K.shape)
            K = rgb2gray(K)
            K = zero_mean_total(K)
            K = wiener_dft(K, K.std(ddof=1)).astype(np.float32)
            yield {"device_id": device, "resolution": int(args.resolution), "prnu": K}
    # features = Features({
    #         "device_id": Value("string"),
    #         "resolution":Value("int32"),
    #         "prnu": Sequence(Sequence(Value("float32")))
    #     })
    # if not os.path.exists(f"{output_dir}/prnu_median_signals_{args.resolution}"):
    #     ds = Dataset.from_generator(generator=prnu_extraction, features=features, writer_batch_size =400, keep_in_memory=False)
    #     ds.save_to_disk(f"{output_dir}/prnu_median_signals_{args.resolution}")
    def queries_extraction():
        for images, devices, _, _ in tqdm(query_dl):
            normalized_images = images/255.
            residuals = extract_single(normalized_images, model)
            queries = residuals
            for i, query in enumerate(queries):
                yield {"device_id": devices[i], "resolution": int(args.resolution), "query": query.astype(np.float32)}

    
      
    features = Features({
            "device_id": Value("string"),
            "resolution":Value("int32"),
            "query": Sequence(Sequence(Value("float32")))
        })
    ds = Dataset.from_generator(generator=queries_extraction, features=features, writer_batch_size =400, keep_in_memory=False)
    ds.save_to_disk(f"{output_dir}/queries_{args.resolution}")

   
    
if __name__ == "__main__":
    main()


