from torch.utils.data import Dataset, WeightedRandomSampler
import torch
import numpy as np 


def get_weighted_sampler(labels, num_samples):

    class_sample_count = np.array(
        [len(np.where(labels == label_class)[0]) for label_class in np.unique(labels)])
    
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)

    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), num_samples=num_samples)
    return sampler

class ImageDataset(Dataset):
    def __init__(self, imgs, labels, transform=None, target_transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return torch.tensor(image).to(torch.float32), torch.tensor(label).to(torch.float32)