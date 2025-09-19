import argparse
from datasets import load_from_disk, concatenate_datasets, Dataset, Features, Value, Sequence
from tqdm import tqdm
import random
from processors.registration import Registration
from processors.extract_noise_residuals import NoiseExtractor
import numpy as np
import os
import torch
import gc
def keep_certain_devices(dataset, filter_devices):
    devices = list(dataset['device'])
    indices_select = [i for i, d in enumerate(devices) if d in filter_devices]
    return dataset.select(indices_select)

def keep_certain_view(dataset, view):
    views = list(dataset['view'])
    indices_select = [i for i, v in enumerate(views) if view in v]
    return dataset.select(indices_select)

def main():
    registration = Registration()
    config = registration.config

    training_devices = np.load(config["train_devices"])
    # print(training_devices)
    test_devices = np.load(config["test_devices"])
    dataset = load_from_disk(config["data_path"])

    query_images_paths = []
    query_devices = []

    datasets_registration = []
    unique_devices = set(dataset['device'])
    for device in tqdm(training_devices):
        
        if device not in unique_devices:
            continue
        device_ds = keep_certain_devices(dataset, [device])
        view_ds = keep_certain_view(device_ds, "view_1")
        device_images = [os.path.join(config["data_path"], path) for path in list(view_ds['image_path'])]
        random.shuffle(device_images)
        prnu_extraction_images = device_images[:5]
        query_images_train = device_images[5:]
        query_images_paths.extend(query_images_train)
        query_devices.extend([device]*len(query_images_train))
        ds_registration = registration.register_device(prnu_extraction_images, device, persist=False)
        datasets_registration.append(ds_registration)
    reg_ds = concatenate_datasets(datasets_registration)
    reg_ds.save_to_disk(config["output_prnu_fingerprint"])
    
    registration = None
    gc.collect()
    torch.cuda.empty_cache()

    features = Features({
                "device_id": Value("string"),
                'resolutions':Value("int32"),
                "prnu": Sequence(Sequence(Value("float32")))
    })
    noise_extractor = NoiseExtractor()
    query_ds = Dataset.from_generator(lambda: noise_extractor.extract_noise(query_images_paths, query_devices))
    query_ds.save_to_disk(config['output_training_queries'])

if __name__ == "__main__":
    main()

