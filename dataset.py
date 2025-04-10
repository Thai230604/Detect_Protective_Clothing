# dataset.py
import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset

class ProtectiveClothingDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = ['background', 'Vest', 'Helmet', 'Mask', 'Goggles']  # Thêm 'Goggles'
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Kiểm tra thư mục tồn tại
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Directory not found: {self.root_dir}. Please check the --data_dir argument or dataset path.")

        self.image_files = [f for f in os.listdir(self.root_dir) if f.endswith('.jpg')]
        if not self.image_files:
            raise FileNotFoundError(f"No .jpg files found in {self.root_dir}. Please check the dataset structure.")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        xml_path = os.path.join(self.root_dir, img_name.replace('.jpg', '.xml'))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            label = self.class_to_idx[name]
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels
        }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target