# CS 551 - Deep Learning
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip

import glob
import os
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CustomDataSet(Dataset):
    """Load images under folders"""
    def __init__(self, main_dir, ext='*.png', transform=None):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = glob.glob(os.path.join(main_dir, ext))
        self.total_imgs = all_imgs
        print(os.path.join(main_dir, ext))
        print(len(self))

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, 0.


def get_data_loader(data_path, opts):
    """Creates data loaders.
    """
    basic_transform = transforms.Compose([
        transforms.Resize(opts.image_size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if opts.data_aug == 'basic':
        transform = basic_transform
    elif opts.data_aug == 'deluxe':
        load_size = int(1.1 * opts.image_size)
        osize = [load_size, load_size]
        transform = transforms.Compose([
            transforms.Resize(osize, Image.BICUBIC),
            transforms.RandomCrop(opts.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(
                degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        pass

    dataset = CustomDataSet(os.path.join('data/', data_path), opts.ext, transform)
    dloader = DataLoader(dataset=dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)

    return dloader
