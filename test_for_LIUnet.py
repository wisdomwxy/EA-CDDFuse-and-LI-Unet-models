import os
import shutil
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from nets.LIunet import Unet
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils_metrics import compute_mIoU, per_Accuracy
import cv2

dataset_path = r'D:\pyc_workspace\unetPPA\dataset_Liunet'
model_path = r'logs\87.19unet+resnet+MSFF+CLFE.pth'
num_classes = 4
backbone = "resnet50"
input_shape = [512, 512]
cuda = True
name_classes = ["_background_", "worker", "coal", "other"]
miou_out_path = ".temp_miou_out"

def predict_image(model, image, input_shape, cuda):
    image = cvtColor(image)
    orininal_h = np.array(image).shape[0]
    orininal_w = np.array(image).shape[1]
    image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    with torch.no_grad():
        images = torch.from_numpy(image_data)
        if cuda:
            images = images.cuda()
        
        pr = model(images)[0]
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        pr = pr[int((input_shape[0] - nh) // 2):int((input_shape[0] - nh) // 2 + nh),
                int((input_shape[1] - nw) // 2):int((input_shape[1] - nw) // 2 + nw)]
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)
    
    return Image.fromarray(np.uint8(pr))

def main():
    images_dir = os.path.join(dataset_path, "JPEGImages")
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    image_ids = [os.path.splitext(f)[0] for f in sorted(image_files)]
    print(f"Found {len(image_ids)} validation images")
    
    device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
    model = Unet(num_classes=num_classes, backbone=backbone)
    
    print('Loading weights...')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.to(device)
    model.eval()
    print('Weights loaded successfully!')
    
    pred_dir = os.path.join(miou_out_path, 'detection-results')
    os.makedirs(pred_dir, exist_ok=True)
    
    print("\nGenerating predictions...")
    for image_id in tqdm(image_ids):
        image_path = os.path.join(dataset_path, "JPEGImages", image_id + ".jpg")
        image = Image.open(image_path)
        pred_image = predict_image(model, image, input_shape, cuda)
        pred_image.save(os.path.join(pred_dir, image_id + ".png"))
    
    print("\nComputing mIoU...")
    gt_dir = os.path.join(dataset_path, "SegmentationClass")
    hist, IoUs, PA_Recall, Precision, F1_Score = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
    
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    for i in range(num_classes):
        class_name = name_classes[i]
        print(f"{class_name:<15} IoU:{IoUs[i]*100:>6.2f}% Recall:{PA_Recall[i]*100:>6.2f}% Precision:{Precision[i]*100:>6.2f}% F1:{F1_Score[i]*100:>6.2f}%")
    
    print("-"*60)
    print(f"{'Average':<15} IoU:{np.nanmean(IoUs)*100:>6.2f}% Recall:{np.nanmean(PA_Recall)*100:>6.2f}% Precision:{np.nanmean(Precision)*100:>6.2f}% F1:{np.nanmean(F1_Score)*100:>6.2f}%")
    print(f"Overall Accuracy: {per_Accuracy(hist)*100:.2f}%")
    print("="*60)
    
    shutil.rmtree(miou_out_path)
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()

