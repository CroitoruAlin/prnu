from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision.transforms import Pad
import prnu
from tqdm import tqdm
from datasets import load_from_disk


class DeviceRegistrationDataset(Dataset):

    def __init__(self, image_paths, devices, target_resolution):
        self.resolution = target_resolution
        self.image_paths = image_paths
        self.devices = devices 
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        device = self.devices[idx]
        image = Image.open(image_path)
        width, height = image.size
        padding_size = (0, 0, 0, 0)
        if height<width:
            border_size = width-height
            padding_size =(0, border_size//2, 0, border_size//2)
        elif height>width:
            border_size = height-width
            padding_size = (border_size//2, 0, border_size//2, 0)
        padding = Pad(padding_size,padding_mode="reflect")
        image = padding(image)
        image = image.resize((self.resolution, self.resolution))
        image = np.array(image).astype(np.uint8)

        return image, device


def filter_ds(ds, used_device):
    devices = list(ds['device_id'])
    indices_select = [i for i,d in enumerate(devices) if d==used_device ]
    return ds.select(indices_select)