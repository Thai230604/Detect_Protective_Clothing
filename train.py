# train.py
import torch
import os
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataloader import get_dataloaders
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on Protective Clothing Dataset")
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='Protective_Clothing_pascalVOC_dataset', help='Path to dataset directory')
    parser.add_argument('--resume', type=str, default='checkpoint.pth', help='Path to checkpoint to resume training')
    parser.add_argument('--log_dir', type=str, default='runs/protective_clothing_experiment', help='Directory to save TensorBoard logs')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models and checkpoints')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    # Bỏ --gpu_id vì không cần kiểm tra GPU ID nữa
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Chọn device (đơn giản hóa)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tạo thư mục save_dir nếu chưa tồn tại
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Tạo DataLoader
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size, num_workers=args.num_workers)

    # Tạo model
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    num_classes = 5  # 4 lớp (Vest, Helmet, Mask, Goggles) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Chuyển model sang device
    model.to(device)

    # Optimizer và scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Khởi tạo TensorBoard writer
    writer = SummaryWriter(args.log_dir)

    # Khởi tạo biến cho checkpoint
    start_epoch = 0
    best_map = -1

    # Tải checkpoint nếu có
    checkpoint_path = os.path.join(args.save_dir, args.resume)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['start_epoch']
        best_map = checkpoint.get('best_map', -1)
        print(f"Resumed training from epoch {start_epoch}, best mAP: {best_map:.4f}")

    # Huấn luyện
    for epoch in range(start_epoch, args.num_epochs):
        # Huấn luyện
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]", leave=False)
        for i, (images, targets) in enumerate(train_loop):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_loop.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch+1, args.num_epochs, train_loss / (i + 1)))
        
        train_loss /= len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Validation (chỉ tính mAP, không tính loss)
        model.eval()
        metric = MeanAveragePrecision(iou_type="bbox")
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for j, (images, targets) in enumerate(val_loop):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                predictions = model(images)
                metric.update(predictions, targets)
        
        map_result = metric.compute()
        current_map = map_result["map"].item()
        writer.add_scalar('Val/mAP', current_map, epoch)
        val_loop.set_description("Epoch {}/{}. mAP {:0.4f}".format(epoch+1, args.num_epochs, current_map))
        
        # Cập nhật learning rate
        scheduler.step()
        
        # Lưu checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "start_epoch": epoch + 1,
            "best_map": best_map
        }
        torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pth'))
        print(f"Saved checkpoint at epoch {epoch+1}")
        
        # Lưu model nếu mAP tốt hơn
        if current_map > best_map:
            best_map = current_map
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model_mobilenet.pth'))
            print(f"Saved best model with mAP: {best_map:.4f}")
        
        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val mAP: {current_map:.4f}")

    # Lưu model cuối cùng
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model_mobilenet.pth'))

    # Đóng TensorBoard writer
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    main()