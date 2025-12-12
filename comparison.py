import argparse
import random
from processors.comparison import Comparison
from processors.extract_noise_residuals import NoiseExtractor
import numpy as np
import os
import torch
import gc
import json

def main():
    parser = argparse.ArgumentParser(description='PRNU Computation')
    parser.add_argument("--infer_device_id", action="store_true")
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--device_list", type=str, default=None)
    args = parser.parse_args()
    device_list = None
    if args.device_list is not None:
        device_list = args.device_list.split(",")
    comparison = Comparison()
    if args.infer_device_id:
        image_paths = []
        devices = []
        for device_name in os.listdir(args.input_path):
            folder_images = os.path.join(args.input_path, device_name)
            for image in os.listdir(folder_images):
                image_paths.append(os.path.join(folder_images, image))
                devices.append(device_name)
        final_scores, fakeness_score, unique_devices = comparison.device_comparison(image_paths, device_list=device_list, gt=devices)
        top_k_preds = np.argsort(final_scores.T, axis=1)[:, -5:][:, ::-1]
        top_k_devices=  []
        for i in range(top_k_preds.shape[0]):
            top_k_devices.append([])
            for j in range(top_k_preds.shape[1]):
                top_k_devices[i].append(unique_devices[top_k_preds[i][j]])
        top_k_scores = np.sort(final_scores.T, axis=1)[:, -5:][:, ::-1]
        answer = {}
        for i, image_path in enumerate(image_paths):
            current_scores = top_k_scores[i]
            current_devices = top_k_devices[i]
            answer[image_path] = {'Top 5 most similar devices':[{current_devices[j]: str(current_scores[j])} for j in range(len(current_scores))]}
        with open("result.json", "w") as f:
            json.dump(answer, f)
    else:
        device_name = os.path.dirname(args.input_path)
        print(f"Registering device {device_name}")
        registration.register_device(args.input_path, device_name, persist=True)
    

if __name__ == "__main__":
    main()
