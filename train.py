import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_from_disk
import random
import torchvision
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import wandb
import timm

import argparse
import einops
import torch.nn as nn
import os


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, args, transform=None):
        anchors = load_from_disk(args.prnu_signals_path, keep_in_memory=False).with_format("numpy")
        unique_devices = np.unique(anchors['device_id'])
        if not os.path.exists("train_devices.npy"):
            permutation = np.random.permutation(len(unique_devices))
            train_devices = unique_devices[permutation[:50]]
            test_devices = unique_devices[permutation[50:]]
            print(train_devices)
            np.save("train_devices.npy", train_devices)
            np.save("test_devices.npy", test_devices)
        else:
            train_devices = np.load("train_devices.npy")
            test_devices = np.load("test_devices.npy")
        self.anchors = {}
        for device in tqdm(train_devices):
            ds = anchors.filter(lambda sample: sample['device_id']==device and sample['resolutions'] == args.resolution).with_format("numpy")['prnu']
            list_prnus = []
            for prnu in ds:

                list_prnus.append(prnu)
            self.anchors[device]=np.mean(np.array(list_prnus), axis=0)
        self.queries = load_from_disk(args.query_training_path, keep_in_memory=False).with_format("numpy").filter(lambda sample: sample['device_id'] in train_devices and sample['resolutions'] == args.resolution)
        self.negative_size = 1

    def __len__(self):
        return len(self.queries)
    
    def normalize(self, x):
        norm = np.linalg.norm(x.ravel(), ord=2)
        x_norm = x / (norm + 1e-8)
        return x_norm
    def __getitem__(self, idx):
        positive = self.queries[idx]["query"].astype(np.float32)
        device_id = self.queries[idx]['device_id']
        negative_device_id = device_id
        negatives = np.zeros((self.negative_size, positive.shape[0], positive.shape[1]), dtype=np.float32)
        for i in range(self.negative_size):
            negative_device_id = device_id 
            while negative_device_id == device_id:
                entry = random.choice(self.queries)
                negative_device_id = entry['device_id']
            negative = entry['query'].astype(np.float32)
            negatives[i]=negative
        anchor = self.anchors[device_id]
        
        return anchor*positive, anchor*negatives[0], 1, 0

class EmbeddingModel(nn.Module):
    def __init__(self, resolution, embed_dim=256):
        super().__init__()

        self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True)
        old_weight = self.backbone.conv1.weight.data
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=self.backbone.conv1.bias is not None
        )
        with torch.no_grad():
            self.backbone.conv1.weight[:] = old_weight.mean(dim=1, keepdim=True)
        feat_info = self.backbone.feature_info
        self.index=-1
        if int(resolution)==1400:
            self.index=-2
        c_last = feat_info.channels()[self.index] 
        self.classifier = None
        self.c_last = c_last
        self.pool = nn.AdaptiveAvgPool2d(((1, 1)))

    def _build_head(self, h, w):
        in_feats = self.c_last * h * w
        self.classifier = nn.Linear(in_feats, 1)
        self.classifier.to(next(self.backbone.parameters()).device)

    def forward(self, x):
        feats = self.backbone(x)[self.index]         # (B, C, H, W)
        B, C, H, W = feats.shape
        if self.index!=-1:
            H = int(H/(2**abs(self.index)))
            W = int(W/(2**abs(self.index)))
            feats = F.adaptive_avg_pool2d(feats, (H, W))
        
        if self.classifier is None:
            self._build_head(H, W)
        input_sum = torch.sum(x, dim=(1,2,3)).unsqueeze(1)
        z = feats.reshape(B, C * H * W)     
        
        logits = self.classifier(z)   
        
        return logits


import gc
def train(model, dataloader, device, args, margin=1.0, epochs=10):
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=1e-3, weight_decay=1e-4)
    print("trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    for n,p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            pass
    for epoch in range(0, epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)
        for step, (p, n, labels_pos, labels_neg) in enumerate(progress_bar):
            p = p.float().to("cuda").unsqueeze(1)
            n = n.float().to("cuda").unsqueeze(1)
            labels_pos, labels_neg = labels_pos.to("cuda"), labels_neg.to("cuda")
            labels = torch.concat((labels_pos, labels_neg), dim=0).unsqueeze(1).float()
            samples = torch.concat((p, n), dim=0)
            logits = model(samples)
            loss = criterion(logits, labels)
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

            avg_loss = running_loss / (step + 1)
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
            wandb.log({"Training Loss": avg_loss})
            
        os.makedirs("trained_models", exist_ok=True)
        torch.save(model.state_dict(), f"trained_models/resnet50_{args.resolution}_{epoch}.pt")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PRNU Computation')
    parser.add_argument("--resolution", type=int, default=1400)
    parser.add_argument("--prnu_signals_path", type=str)
    parser.add_argument("--query_training_path", type=str)
    args = parser.parse_args()
    wandb.init(project="PRNU Contrastive Learning Model",
    config=vars(args))
    ds = TripletDataset(args)
    model = EmbeddingModel(args.resolution)
    model(torch.zeros((1, 1, args.resolution, args.resolution)))
    
    batch_size=4
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {pytorch_total_params}")
    
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8)
    train(model, train_loader, "cuda", args, epochs=50)
    
