import albumentations as A
import os
import math
import h5py
import torch
import random
import numpy as np
import pandas as pd
import blobfile as bf
import itertools

from os import path as osp
from PIL import Image
# from mpi4py import MPI
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from glob import glob


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# ============================================================================
# CelebA dataloader
# ============================================================================


def load_data_celeba(
    *,
    data_dir,
    batch_size,
    image_size,
    partition='train',
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = CelebADataset(
        image_size,
        data_dir,
        partition,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        class_cond=class_cond,
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=5, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=5, drop_last=True
        )
    while True:
        yield from loader


class CelebADataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
    ):
        partition_df = pd.read_csv(osp.join(data_dir, 'list_eval_partition.csv'))
        self.data_dir = data_dir
        data = pd.read_csv(osp.join(data_dir, 'list_attr_celeba.csv'))

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[partition_df['partition'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])

        self.query = query_label
        self.class_cond = class_cond

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        labels = sample[2:].to_numpy()
        if self.query != -1:
            labels = int(labels[self.query])
        else:
            labels = torch.from_numpy(labels.astype('float32'))
        img_file = sample['image_id']

        with open(osp.join(self.data_dir, 'img_align_celeba', img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)

        if self.query != -1:
            return img, labels, labels

        if self.class_cond:
            return img, {'y': labels}
        else:
            return img, {}


class CelebAMiniVal(CelebADataset):
    def __init__(
        self,
        image_size,
        data_dir,
        csv_file='utils/minival.csv',
        partition=None,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
    ):
        self.data = pd.read_csv(csv_file).iloc[:, 1:]
        self.data = self.data[shard::num_shards]
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x,
        ])
        self.data_dir = data_dir
        self.class_cond = class_cond
        self.query = query_label


class ShortcutCelebADataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        normalize=True,
        query_label=31,
        task_label=39,
        shortcut_label_name='Smiling',
        task_label_name='Young',
        percentage=0.5,
        n_samples=1000,
        seed=4,
    ):
        partition_df = pd.read_csv(osp.join(data_dir, 'list_eval_partition.csv'))
        self.data_dir = data_dir
        data = pd.read_csv(osp.join(data_dir, 'list_attr_celeba.csv'))
        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')
        self.data = data[partition_df['partition'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)
        shortcut_samples = int(n_samples * percentage)
        noshortcut_samples = n_samples - shortcut_samples


        shortcut_positive = self.data[(self.data[shortcut_label_name] == 1) & (self.data[task_label_name] == 1)].sample(n=shortcut_samples, random_state=seed, replace=False)
        shortcut_negative = self.data[(self.data[shortcut_label_name] == 1) & (self.data[task_label_name] == 0)].sample(n=noshortcut_samples, random_state=seed, replace=False)
        noshortcut_positive = self.data[(self.data[shortcut_label_name] == 0) & (self.data[task_label_name] == 1)].sample(n=noshortcut_samples, random_state=seed, replace=False)
        noshortcut_negative = self.data[(self.data[shortcut_label_name] == 0) & (self.data[task_label_name] == 0)].sample(n=shortcut_samples, random_state=seed, replace=False)
        subset_dataset = pd.concat([shortcut_positive, shortcut_negative, noshortcut_positive, noshortcut_negative])
        print()
        print('Shortcut CelebA dataset')
        print(f'Query label ID {query_label}')
        print(f'Shortcut label {shortcut_label_name}')
        print(f'Task label {task_label_name}')
        print(f'Number of shortcut/positive task label samples: {len(shortcut_positive)}')
        print(f'Number of shortcut/negative task label samples: {len(shortcut_negative)}')
        print(f'Number of nonshortcut/positive task label samples: {len(noshortcut_positive)}')
        print(f'Number of nonshortcut/negative task label samples: {len(noshortcut_negative)}')
        print()
        self.data = subset_dataset
        self.data = self.data.sort_index()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])
        self.query = query_label
        self.task = task_label
        self.class_cond = class_cond
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        labels_ = sample[2:].to_numpy()
        if self.query != -1:
            labels = int(labels_[self.query])
            task_labels = int(labels_[self.task])
        else:
            labels = torch.from_numpy(labels.astype('float32'))
        img_file = sample['image_id']
        with open(osp.join(self.data_dir, 'img_align_celeba', img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)
        if self.query != -1:
            return img, labels, task_labels
        if self.class_cond:
            return img, {'y': labels}
        else:
            return img, {}


class ShortcutCFDataset(CelebADataset):
    def __init__(
        self,
        path='/scratch/ppar/output', 
        exp_name='fastdime'):

        self.images = []
        self.path = path
        self.exp_name = exp_name

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])])

        for CL, CF in itertools.product(['CC'], ['CCF', 'ICF']):
            self.images += [(CL, CF, I) for I in os.listdir(osp.join(path, 'Results', self.exp_name, CL, CF, 'CF'))]
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        CL, CF, I = self.images[idx]

        with open(osp.join(self.path, 'Results', self.exp_name, CL, CF, 'Info', I.split('.')[0] + '.txt'), 'r') as f:
            for line in f:
                if line.startswith("pred:"):
                    pred = int(line.split(":")[1].strip())
                elif line.startswith("target:"):
                    target = int(line.split(":")[1].strip())
                elif line.startswith("cf pred:"):
                    cf_pred = int(line.split(":")[1].strip())
                elif line.startswith("task_label:"):
                    task_label = int(line.split(":")[1].strip())
                elif line.startswith("label:"):
                    shortcut_label = int(line.split(":")[1].strip())

        cl_path = osp.join(self.path, 'Original', 'Correct' if CL == 'CC' else 'Incorrect', I)
        cf_path = osp.join(self.path, 'Results', self.exp_name, CL, CF, 'CF', I)
        cf = self.load_img(cf_path)
        cl = self.load_img(cl_path)

        return cl, cf, task_label, shortcut_label

    def load_img(self, path):
        img = Image.open(os.path.join(path))
        img = np.array(img, dtype=np.uint8)
        return self.transform(img)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    dataset = ShortcutCelebADataset(image_size=128,
        data_dir='/scratch/ppar/data/img_align_celeba/',
        partition='val',
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=31,
        normalize=True,)
    print(len(dataset))
    img, labels, task_labels = next(iter(dataset))
    print(img.shape, img.min(), img.max())
    plt.imshow((img.permute(1,2,0).numpy()+1)/2)
    plt.imsave('assets/example_shortcut.png', (img.permute(1,2,0).numpy()+1)/2 )
