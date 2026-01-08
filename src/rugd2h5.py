import h5py
import numpy as np
import os
import tyro
import torch
import cv2
from tqdm import tqdm
from utils import map_label_colors, is_label_in_image


def _get_rugd_files_path(root_dir: str):
    """
    Get the paths of all RUGD images and their corresponding labels.
    Args:
        root_dir (str): The root directory of RUGD dataset.
    Returns:
        List of tuples containing image paths and label paths.
    """
    rugd_files_path = []
    
    labels_dir = os.path.join(root_dir, "RUGD_annotations")
    images_dir = os.path.join(root_dir, "RUGD_frames-with-annotations")

    if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
        raise FileNotFoundError("Required directories do not exist in the specified root directory.")

    labels_subdir = set(os.listdir(labels_dir))
    images_subdir = set(os.listdir(images_dir))

    available_subdir = labels_subdir.intersection(images_subdir)

    for subdir in available_subdir:
        labels_subdir_path = os.path.join(labels_dir, subdir)
        images_subdir_path = os.path.join(images_dir, subdir)

        sub_labels = sorted([ file_path for file_path in os.listdir(labels_subdir_path) if file_path.endswith('.png') ])
        sub_images = sorted([ file_path for file_path in os.listdir(images_subdir_path) if file_path.endswith('.png') ])

        if len(sub_labels) != len(sub_images):
            raise ValueError(f"Mismatch in number of label and image files in subdirectory '{subdir}'.")
        
        for label_file, image_file in zip(sub_labels, sub_images):
            label_path = os.path.join(labels_subdir_path, label_file)
            image_path = os.path.join(images_subdir_path, image_file)

            rugd_files_path.append((image_path, label_path))
    return rugd_files_path

def _save_images_to_h5(rugd_files_path, save_path):
    """
    Save images and labels to an HDF5 file.
    Args:
        rugd_files_path (list): List of tuples containing image paths and label paths.
        save_path (str): The path where the HDF5 file will be saved.
    """
    data_len = len(rugd_files_path)
    with h5py.File(save_path, 'w') as f:
        dt_uint8 = h5py.special_dtype(vlen=np.dtype('uint8'))
        f.create_dataset('images', (data_len, ), dtype=dt_uint8)
        f.create_dataset('labels', (data_len, ), dtype=dt_uint8)

        for i, (image_path, label_path) in enumerate(rugd_files_path):
            with open(image_path, 'rb') as img_file:
                f['images'][i] = np.frombuffer(img_file.read(), dtype=np.uint8)
            with open(label_path, 'rb') as lbl_file:
                labels_rgb = np.frombuffer(lbl_file.read(), dtype=np.uint8)
                labels_rgb = cv2.imdecode(labels_rgb, cv2.IMREAD_COLOR_RGB)
                labels = map_label_colors(torch.from_numpy(labels_rgb))[..., 0].numpy().astype(np.uint8)
                labels_binary = cv2.imencode('.png', labels)[1]
                f['labels'][i] = labels_binary
            
            # Logging progress every 100 files
            if i % 100 == 0:
                print(f"Processed {i}/{data_len} files...")


def _split_dataset(rugd_files_path, unseen_labels: set[int]):
    """
    Split the dataset into training and other sets based on unseen labels.
    Args:
        rugd_files_path (list): List of tuples containing image paths and label paths.
        unseen_labels (set[int]): Set of labels to be considered unseen.
    Returns:
        Tuple containing training files path and other files path.
    """
    train_files_path, other_files_path = [], []

    tbar = tqdm(rugd_files_path, total=len(rugd_files_path), desc="Splitting dataset", dynamic_ncols=True, leave=False)
    for image_path, label_path in tbar:
        labels_rgb = cv2.imread(label_path, cv2.IMREAD_COLOR_RGB)
        labels = map_label_colors(torch.from_numpy(labels_rgb))[..., 0]
        if is_label_in_image(labels, unseen_labels):
            other_files_path.append((image_path, label_path))
        else:
            train_files_path.append((image_path, label_path))
    
    # Split other_files_path into validation and test sets (e.g., 50-50 split)
    val_files_path = other_files_path[:len(other_files_path)//2]
    test_files_path = other_files_path[len(other_files_path)//2:]
    return train_files_path, val_files_path, test_files_path


def create_rugd_h5(root_dir: str, save_path: str, split: bool = True):
    """
    Create an HDF5 file containing the paths to all RUGD images in the specified directory.

    Args:
        root_dir (str): The root directory of RUGD dataset.
        save_path (str): The path where the HDF5 file will be saved.
        split (bool): Whether to split the dataset into different sets.
    """
    rugd_files_path = _get_rugd_files_path(root_dir)

    if not split:
        _save_images_to_h5(rugd_files_path, save_path)
    else:
        # TODO: Implement dataset splitting and saving logic
        unseen_labels = set([20, 21, 22, 23, 24])
        train_files_path, val_files_path, test_files_path = _split_dataset(rugd_files_path, unseen_labels)
        _save_images_to_h5(train_files_path, save_path.replace('.h5', '_train.h5'))
        _save_images_to_h5(val_files_path, save_path.replace('.h5', '_val.h5'))
        _save_images_to_h5(test_files_path, save_path.replace('.h5', '_test.h5'))
        print(f"Dataset split into train({len(train_files_path)}), val({len(val_files_path)}), and test({len(test_files_path)}) sets and saved accordingly.")

def main():
    tyro.cli(create_rugd_h5)

if __name__ == "__main__":
    main()

