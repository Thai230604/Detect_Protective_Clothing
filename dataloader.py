# dataloader.py
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import ProtectiveClothingDataset

def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataloaders(data_dir, batch_size, num_workers=4):  # Thêm num_workers
    # Transform
    transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
    ])

    # Tạo dataset
    train_dataset = ProtectiveClothingDataset(
        root_dir=data_dir,
        split='train',
        transform=transform
    )

    val_dataset = ProtectiveClothingDataset(
        root_dir=data_dir,
        split='valid',
        transform=transform
    )

    # Tạo DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,  # Thêm num_workers
        pin_memory=True if torch.cuda.is_available() else False  # Tăng tốc nếu dùng GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,  # Thêm num_workers
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader