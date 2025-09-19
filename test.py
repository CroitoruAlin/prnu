from datasets import load_from_disk
import argparse
import prnu
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
from train import EmbeddingModel
def filter_ds(ds, used_device):
    devices = list(ds['device_id'])
    indices_select = [i for i,d in enumerate(devices) if d in used_device ]
    return ds.select(indices_select)
def filter_ds_resolution(ds, resolution):
    resolutions = list(ds['resolutions'])
    indices_select = [i for i,r in enumerate(resolutions) if r==resolution ]
    return ds.select(indices_select)

class PRNUStatsDataset(Dataset):

    def __init__(self, path, resolution, prnu=True, used_devices=None):
        if used_devices is None:
            self.ds = filter_ds_resolution(load_from_disk(path, keep_in_memory=False), resolution).with_format("numpy")
        else:
            self.ds = filter_ds( filter_ds_resolution(load_from_disk(path, keep_in_memory=False), resolution).with_format("numpy"), used_devices)
        self.prnu = prnu

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        entry = self.ds[idx]
        if self.prnu:
            return entry["prnu"], entry['device_id']
        else:
            return entry['query'], entry['device_id'] 

def compute_prnus_restormer(prnu_dl, resolution, unique_devices):
    prnus=[]
    all_prnus_devices=[]
    noises = []

    for noise_estimated, devices in tqdm(prnu_dl):
        noises.extend(noise_estimated)
    prnu = np.mean(np.array(noises), axis=0)
    return prnu
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PRNU Computation')
    parser.add_argument("--resolutions", type=str, default="1024,1400")
    parser.add_argument("--prnu_signals_path", type=str)
    parser.add_argument("--query_path", type=str)
    parser.add_argument("--ckpt_paths", type=str)
    args = parser.parse_args()
    unique_resolutions = [int(res) for res in args.resolutions.split(",")]
    ckpt_paths = args.ckpt_paths.split(",")
    cc_matrices = []
    weights = []
    for i, r in enumerate(unique_resolutions):
        model = EmbeddingModel(r)
        model(torch.zeros((1, 1, r, r)))
        model.load_state_dict(torch.load(ckpt_paths[i]))
        model.to("cuda")
        model.eval()
        weights.append(r)
        query_ds = PRNUStatsDataset(args.query_path, r, False)
        query_dl = DataLoader(query_ds, batch_size=8, num_workers=8)
        unique_devices = set(query_ds.ds['device_id'])
        prnus = []
        resolution = r
        for device in unique_devices:
                prnu_ds = PRNUStatsDataset(args.prnu_signals_path, resolution, True, used_devices=[device])
                prnu_dl = DataLoader(prnu_ds, batch_size=8, num_workers=8)
                prnus.append(compute_prnus_restormer(prnu_dl, resolution, unique_devices))
        prnus = torch.from_numpy(np.array(prnus)).to("cuda").unsqueeze(1)
        cc_aligned_rot = []
        batch_size = 8
        all_query_devices = []
        for residuals, devices in tqdm(query_dl):
            queries = residuals.unsqueeze(1).to("cuda")
            similarities = torch.einsum("qchw,pchw->qpchw", queries, prnus)
            size_q, size_p = similarities.shape[0], similarities.shape[1]
            similarities = rearrange(similarities, "q p c h w-> (q p) c h w")
            all_query_devices.extend(devices)
            scores = []
            with torch.no_grad():
                for i in range(0, size_q*size_p, batch_size):
                    batch = similarities[i:i+batch_size]
                    scores.extend(batch.sum((1,2,3)).cpu().numpy() + model(batch).cpu().numpy().squeeze())
            scores = np.array(scores).squeeze()
            scores = rearrange(scores, '(q p)->q p', q=size_q, p=size_p)
            
            cc_aligned_rot.append(scores.T)
        cc_aligned_rot = np.concatenate(cc_aligned_rot, axis=1)
        cc_matrices.append(cc_aligned_rot)
    cc_matrices = np.stack(cc_matrices, axis=0)
    print(f'CC matrices: {cc_matrices.shape}')
    weights = np.array(weights).astype(np.float32)
    weights = weights/np.max(weights)
    cc_matrices = cc_matrices * weights[:, None, None]
    cc_aligned_rot = np.sum(cc_matrices, axis=0)
    all_query_devices = np.array(all_query_devices)
    gt = prnu.gt(list(unique_devices), all_query_devices)
    print(cc_aligned_rot.shape)
    print('Computing statistics cross correlation')
    stats_cc = prnu.stats(cc_aligned_rot, gt)
    print('AUC on CC {:.2f}'.format(stats_cc['auc']))
    print("Top 1 accuracy", stats_cc["top-1-acc"])
    print("Top 5 accuracy", stats_cc["top-5-acc"])
    print("EER", stats_cc["eer"])