import argparse
import random
from processors.registration import Registration
from processors.extract_noise_residuals import NoiseExtractor
import numpy as np
import os
import torch
import gc


def main():
    parser = argparse.ArgumentParser(description='PRNU Computation')
    parser.add_argument("--register_multiple_devices", action="store_true")
    parser.add_argument("--input_path", type=str)
    args = parser.parse_args()
    registration = Registration()
    if args.register_multiple_devices:
        for device_name in os.listdir(args.input_path):
            folder_images = os.path.join(args.input_path, device_name)
            print(f"Registering device {device_name}")
            registration.register_device(folder_images, device_name, persist=True)
    else:
        device_name = os.path.dirname(args.input_path)
        print(f"Registering device {device_name}")
        registration.register_device(args.input_path, device_name, persist=True)
    

if __name__ == "__main__":
    main()
