# Detect_Protective_Clothing

# Access this link and download Protective_Clothing_pascalVOC_dataset.zip
https://drive.google.com/drive/folders/1Y9Da9FfnEtcxibKT7fTo4zYVM90fYHX_

# Demo

![Demo](./demo/1.png)

![Demo](./demo/3.png)

![Demo](./demo/2.png)

# Protective Clothing Detection with Faster R-CNN
```bash
This project implements a protective clothing detection system using Faster R-CNN with MobileNetV3 backbone. It detects four types of protective equipment: Vests, Helmets, Masks, and Goggles.
```
## Features
```bash
- Object detection for protective clothing items
- Training and evaluation scripts
- Model testing with visualization
- Support for PascalVOC format dataset
- TensorBoard logging for training metrics
```
# Dataset Structure
```bash
The dataset should be organized in PascalVOC format with the following structure:

Protective_Clothing_pascalVOC_dataset/
├── train/
│ ├── *.jpg
│ ├── *.xml
├── valid/
│ ├── *.jpg
│ ├── *.xml
├── test/
├── *.jpg
├── *.xml
```

## Installation
```bash
1. Clone this repository
2. Install requirements:


pip install -r requirements.txt
```

## Training
```bash
python train.py --num_epochs 20 --batch_size 8 --data_dir path/to/dataset
```
## Testing
```bash
python test.py --model_path models/best_model_mobilenet.pth --data_dir path/to/dataset --path image_name.jpg
```

## Explain argument Training:
```bash
--num_epochs: Number of training epochs (default: 10)

--batch_size: Batch size (default: 4)

--lr: Learning rate (default: 0.005)

--data_dir: Path to dataset directory (default: 'Protective_Clothing_pascalVOC_dataset')

--resume: Path to checkpoint to resume training (default: 'checkpoint.pth')

--log_dir: Directory for TensorBoard logs (default: 'runs/protective_clothing_experiment')

--save_dir: Directory to save models (default: 'models')

--num_workers: Number of DataLoader workers (default: 4)
```
## Explain argument Testing:
```bash
--batch_size: Batch size for testing (default: 4)

--data_dir: Path to dataset directory (default: 'Protective_Clothing_pascalVOC_dataset')

--path: Image filename to test (default: '00350_jpg.rf.858791c514ed5868dc98e1bae1ee2711.jpg')

--model_path: Path to trained model (default: 'models/best_model_mobilenet.pth')

--num_workers: Number of DataLoader workers (default: 4)
```
