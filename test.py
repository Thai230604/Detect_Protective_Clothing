# test.py
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import argparse
from dataloader import get_dataloaders
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image
import torchvision.transforms as transforms
import cv2
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Test Faster R-CNN on Protective Clothing Dataset")
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--data_dir', type=str, default='Protective_Clothing_pascalVOC_dataset', help='Path to dataset directory')
    parser.add_argument('--path', type=str, default='00350_jpg.rf.858791c514ed5868dc98e1bae1ee2711.jpg', help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, default='models/best_model_mobilenet.pth', help='Path to the trained model')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Chọn device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    num_classes = 5  # 4 lớp (Vest, Helmet, Mask, Goggles) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    path = os.path.join(args.data_dir, 'test', args.path)

    o_image = Image.open(path).convert('RGB')
    width, height = o_image.size
    transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
    ])
    image = transform(o_image)
    image = image.to(device)
    image = [image]
    model.eval()
    with torch.no_grad():
        predict = model(image)

    cv_image = cv2.imread(path)
    cate = ['background', 'Vest', 'Helmet', 'Mask', 'Goggles']
    for boxes, labels, scores in zip(predict[0]['boxes'],predict[0]['labels'],predict[0]['scores']):
        if scores > 0.75:
            x1 = int(boxes[0]/600 * width)
            y1 = int(boxes[1]/600 * height)
            x2 = int(boxes[2]/600 * width)
            y2 = int(boxes[3]/600 * height)
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), color=(0,255,0), thickness=2)
            cv2.putText(cv_image, cate[labels], (x1, y2),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    print(predict[0]['scores'])
    cv2.imshow("lalala", cv_image)
    cv2.waitKey(0) # 0==wait forever
             


if __name__ == "__main__":
    main()