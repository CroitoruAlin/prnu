import os
import time
import json
import yaml
import gc
import numpy as np
from datasets import load_from_disk, Features, Value, Array2D
from einops import rearrange
import torch.nn.functional as F
from torch.utils.data import DataLoader
from processors.dataset import DeviceRegistrationDataset
from basicsr.models.archs.restormer_arch import Restormer
from basicsr.models.network_unet import UNetRes
import torch
from tqdm import tqdm
from prnu import extract_single_drunet, extract_single, aligned_cc, aligned_cc_torch, extract_single_classic
from train import EmbeddingModel
import prnu
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def filter_dataset_by_view(dataset, view):
    views = list(dataset['view'])
    indices_select = [i for i, v in enumerate(views) if view in v]
    return dataset.select(indices_select)


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


def filter_dataset_by_list_of_devices_image_ds(dataset, device_list):
    cols = set(dataset.column_names)
    devs = np.asarray(dataset["device"])
    mask = np.isin(devs, np.asarray(device_list))
    idx = np.nonzero(mask)[0].tolist()
    return dataset.select(idx)


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


class Comparison:

    def __init__(self):
        with open("configs/config.json", "r") as f:
            self.config = json.load(f)
        self._create_denoiser()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.registered_devices = load_from_disk(
            self.config["output_prnu_fingerprint"],
            keep_in_memory=False
        )
        keep_cols = [c for c in ["device_id", "resolutions", "prnu"] if c in self.registered_devices.column_names]
        drop_cols = [c for c in self.registered_devices.column_names if c not in keep_cols]
        if drop_cols:
            self.registered_devices = self.registered_devices.remove_columns(drop_cols)

        
        unique_res = self.config['resolutions']#
        self.prnus_per_resolution = {r: filter_dataset_by_resolution(self.registered_devices, r) for r in unique_res}

        self.prnu_cache = {}
        cache_root = os.path.join(self.config["output_prnu_fingerprint"], "_prnu_cache")
        for r, ds_r in self.prnus_per_resolution.items():
            
            first = np.asarray(ds_r[0]["prnu"])
            # print(np.asarray(first).shape)
            # exit()
            flat_shape = int(np.prod(first.shape)) if hasattr(first, "shape") else None
            t0 = time.time()
            prnus, devices = _stack_prnu_fast(ds_r, flat_shape=flat_shape)
            self.prnu_cache[r] = prnus
            t1 = time.time()
        self.devices = list(devices)

    def _create_denoiser(self):
        if self.config.get('denoiser_type') == 'restormer':
            network_config = yaml.load(open("configs/GaussianColorDenoising_Restormer.yml", mode='r'), Loader=Loader)
            network_config['network_g'].pop('type', None)
            self.model = Restormer(**network_config['network_g'])
            weights_path = self.config['denoiser_path']
            self.model.load_state_dict(torch.load(weights_path)['params'])
            self.comparison_models = {}
            ckpts_path = os.path.dirname(self.config['denoiser_path'])
            for resolution in self.config['resolutions']:
                self.comparison_models[resolution] = EmbeddingModel(resolution)
                self.comparison_models[resolution](torch.zeros(1, 1, resolution, resolution))
                self.comparison_models[resolution].load_state_dict(torch.load(os.path.join(ckpts_path, f"model_{resolution}.pt")))
                self.comparison_models[resolution].eval()
        elif self.config.get("denoiser_type") == 'drunet':
            self.model = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                                 downsample_mode="strideconv", upsample_mode="convtranspose")
            self.model.load_state_dict(torch.load(self.config['denoiser_path']), strict=True)
            self.model.eval()
            for _, v in self.model.named_parameters():
                v.requires_grad = False
        else:
            self.model = None

    def device_comparison(self, image_paths, device_list=None, gt=None):
        if self.model is not None:
            self.model.to(self.device)
        device_dataset = {}
        for r, ds_r in self.prnus_per_resolution.items():
            device_dataset[r] = ds_r if device_list is None else filter_dataset_by_list_of_devices(ds_r, device_list)

        final_scores = []

        
        for r in self.config['resolutions']:
            ds_r = device_dataset[r]
            if self.config.get('denoiser_type') == 'restormer':
                self.comparison_models[r].to(self.device)
            if device_list is None:
                prnus = self.prnu_cache[r]
            else:

                base_ids = np.asarray(self.prnus_per_resolution[r]["device_id"])
                sub_ids = np.asarray(ds_r["device_id"])
                
                inv = {d: i for i, d in enumerate(base_ids)}
                rows = np.fromiter((inv[d] for d in sub_ids), dtype=np.int64, count=len(sub_ids))
                print(len(rows))
                prnus = self.prnu_cache[r][rows]

            print(f"[info] resolution {r}: PRNUs shape {prnus.shape}")

            t0 = time.time()
            dataset = DeviceRegistrationDataset(image_paths, gt if gt is not None else ['test']*len(image_paths), r)
            

            dataloader = DataLoader(dataset, num_workers=self.config['num_workers'],
                                    batch_size=self.config['batch_size'], shuffle=False)
            print("Data setup time:", time.time() - t0)

            resolution_scores = []
            all_query_devices = []
            prnus_torch = torch.from_numpy(prnus).unsqueeze(1).float()
            for images, gt_device in tqdm(dataloader):
                if self.config.get('denoiser_type') == 'drunet':
                    t_align0 = time.time()
                    query_noise = extract_single_drunet(images, self.model, 50, 50)
                    t_align1 = time.time()
                    # print("Extract single", t_align1-t_align0)
                    t_align0 = time.time()
                    scores = aligned_cc_torch(query_noise, prnus)
                    t_align1 = time.time()
                    # print("Align cc", t_align1-t_align0)
                    resolution_scores.append(scores['ncc'].T)
                elif self.config.get("denoiser_type") == "restormer":
                    images = images/255.
                    query_noise = extract_single(images, self.model, 50, 50)
                    queries = torch.from_numpy(query_noise).unsqueeze(1).float()
                    
                    similarities = torch.einsum("qchw,pchw->qpchw", queries, prnus_torch)
                    size_q, size_p = similarities.shape[0], similarities.shape[1]
                    similarities = rearrange(similarities, "q p c h w-> (q p) c h w").to(self.device)
                    batch_size = self.config['batch_size_comparison']
                    scores = []
                    with torch.no_grad():
                        for i in range(0, size_q*size_p, batch_size):
                            batch = similarities[i:i+batch_size]
                            scores.extend(self.comparison_models[r](batch).cpu().numpy().squeeze())
                    model_scores = np.array(scores).squeeze()
                    model_scores = rearrange(model_scores, '(q p)->q p', q=size_q, p=size_p).T

                    raw_signal_scores = aligned_cc_torch(query_noise, prnus)['ncc'].T
                    # print(model_scores.shape, raw_signal_scores.shape)
                    # print((raw_signal_scores + model_scores).shape)
                    resolution_scores.append(raw_signal_scores + model_scores)
                else:
                    queries = []
                    for image in images.cpu().numpy():
                        residual = extract_single_classic(image)
                        queries.append(residual)
                    queries = np.array(queries)
                    scores = aligned_cc_torch(queries, prnus)
                    resolution_scores.append(scores['ncc'].T)
                all_query_devices.extend(gt_device)

                

            resolution_scores = np.concatenate(resolution_scores, axis=1)
            final_scores.append(resolution_scores)

            
            del prnus
            gc.collect()

        
        weights = np.asarray(self.config['resolutions'], dtype=np.float32)
        weights = weights / weights.max()
        final_scores = np.stack(final_scores, axis=0)
        final_scores = np.sum(final_scores * weights[:, None, None], axis=0)
        # print(final_scores.shape)
        if gt is not None:
            all_query_devices = np.array(all_query_devices)
            unique_devices = self.devices
            gt_bin = prnu.gt(unique_devices, all_query_devices)
            print(final_scores.shape, gt_bin.shape)
            stats_cc = prnu.stats(final_scores, gt_bin)
            print('AUC on CC {:.5f}'.format(stats_cc['auc']))
            print("Top 1 accuracy", stats_cc["top-1-acc"])
            print("Top 5 accuracy", stats_cc["top-5-acc"])
            print('EER {:.5f}'.format(stats_cc['eer']))
        if self.model is not None:
            self.model.to("cpu")
        return final_scores


if __name__ == "__main__":
    comparison = Comparison()
    t0 = time.time()
    test_devices = np.load("test_devices.npy")[:2]
    image_dataset = load_from_disk("../datasets/PRNU/", keep_in_memory=False)
    image_dataset = filter_dataset_by_view(image_dataset, "view_2")
    image_dataset = filter_dataset_by_list_of_devices_image_ds(image_dataset, list(test_devices))
    # print(len(image_dataset))
    # exit()
    gt_devices = list(image_dataset['device'])
    image_paths = [os.path.join("../datasets/PRNU/", image_path) for image_path in list(image_dataset['image_path'])]
    scores = comparison.device_comparison(image_paths, gt=gt_devices)
    # print(scores.shape, scores)
    print("Total time:", time.time() - t0)

#[[-0.03390784 -0.00919679 -0.05382158 -0.03892823 -0.00190165]
# [ 0.03498485 -0.03103287 -0.05388374 -0.00126721 -0.04313271]
# [ 0.00804291 -0.033762   -0.00577763 -0.01076424 -0.04807157]
# [-0.02588321 -0.03547777  0.00472724  0.01777871 -0.05648795]
# [-0.03781606 -0.02342686  0.03828652 -0.01224873 -0.05234518]]
