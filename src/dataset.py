from torch.utils.data import Dataset
import h5py
import cv2
import torch
from PIL import Image
from collections.abc import Sequence
from typing import Callable
from torchvision import transforms as T
from utils import _random_crop, image_transforms, label_transforms


class RUGDH5Dataset(Dataset):
    def __init__(
            self, 
            h5_path: str, 
            image_transforms: Callable = image_transforms,
            label_transforms: Callable = None,    
        ):
        self.h5_path = h5_path
        self.image_transforms = image_transforms
        self.label_transforms = label_transforms
        
        # 【关键点1】：这里只存路径，不打开文件
        self.h5_file = None 
        self.images_ds = None
        self.labels_ds = None
        
        # 获取数据集长度（这一步只需快速读取一次即可关闭）
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['images'])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 【关键点2】：懒加载。
        # 只有在 worker 进程首次调用 getitem 时才打开文件。
        # 每个 worker 进程会拥有自己独立的文件句柄。
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.images_ds = self.h5_file['images']
            self.labels_ds = self.h5_file['labels']

        # 1. 读取二进制流
        img_bytes = self.images_ds[index]
        lbl_bytes = self.labels_ds[index]

        # 2. 解码 (Decode)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR_RGB) # BGR
        
        # Label 通常是单通道图像
        label = cv2.imdecode(lbl_bytes, cv2.IMREAD_UNCHANGED)

        # 3. 应用变换
        if self.image_transforms:
            image = self.image_transforms(image)
        if self.label_transforms:
            label = self.label_transforms(label)

        image, label = _random_crop(image, label)
        return image, label
    
if __name__ == '__main__':
    dataset = RUGDH5Dataset(
        './data/test.h5',
        image_transforms=image_transforms,
    )

    print(f"Dataset length: {len(dataset)}")

    image, label = dataset[0]

    # cv2.imwrite('sample_image.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # cv2.imwrite('sample_label.png', cv2.cvtColor(label, cv2.COLOR_RGB2BGR))