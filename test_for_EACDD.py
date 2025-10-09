from net_eacdd import Restormer_Encoder, Restormer_Decoder
import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save, image_read_cv2
import warnings
import logging
import cv2
import time


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ckpt_path = r"models/CDDFuse_best_12-18-11-37.pth"  # Replace it with your model path
#Select the test dataset as "scatter"
for dataset_name in ["scatter"]:
    print("\n"*2+"="*80)
    model_name = "CDDFuse_struction"  
    print("The test result of "+dataset_name+' :')
    test_folder = os.path.join('test_img', dataset_name) 
    test_out_folder = os.path.join('test_result', dataset_name)
    os.makedirs(test_out_folder, exist_ok=True)

    # Initialize the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #Encoder = nn.DataParallel(Restormer_Encoder(efficient_attention=True)).to(device)
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)

    # Load the model weights
    checkpoint = torch.load(ckpt_path)
    Encoder.load_state_dict(checkpoint['DIDF_Encoder'])
    Decoder.load_state_dict(checkpoint['DIDF_Decoder'])
    
    Encoder.eval()
    Decoder.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder, "ir")):
          
            data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[
                          np.newaxis, np.newaxis, ...] / 255.0
            data_VIS = cv2.split(image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='YCrCb'))[0][
                           np.newaxis, np.newaxis, ...] / 255.0
            data_VIS_BGR = cv2.imread(os.path.join(test_folder, "vi", img_name))
            _, data_VIS_Cr, data_VIS_Cb = cv2.split(cv2.cvtColor(data_VIS_BGR, cv2.COLOR_BGR2YCrCb))
            
            data_IR = torch.FloatTensor(data_IR).to(device)
            data_VIS = torch.FloatTensor(data_VIS).to(device)

            #torch.cuda.synchronize()

            feature_V_B, feature_V_D, _ = Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = Encoder(data_IR)
            
            data_Fuse, _ = Decoder(data_VIS, 
                                 feature_V_B + feature_I_B,  
                                 feature_V_D + feature_I_D) 
            
            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))

            #torch.cuda.synchronize()
            
            fi = np.squeeze((data_Fuse * 255.0).cpu().numpy())
            fi = fi.astype(np.uint8)
            ycrcb_fi = np.dstack((fi, data_VIS_Cr, data_VIS_Cb))
            rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
            img_save(rgb_fi, img_name.split(sep='.')[0], test_out_folder)


    eval_folder = test_out_folder  
    ori_img_folder = test_folder

    metric_result = np.zeros((8))
    for img_name in os.listdir(os.path.join(ori_img_folder, "ir")):
        ir = image_read_cv2(os.path.join(ori_img_folder, "ir", img_name), 'GRAY')
        vi = image_read_cv2(os.path.join(ori_img_folder, "vi", img_name), 'GRAY')
        fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
        metric_result += np.array([
            Evaluator.EN(fi), Evaluator.SD(fi),
            Evaluator.SF(fi), Evaluator.MI(fi, ir, vi),
            Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi),
            Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)
        ])

    metric_result /= len(os.listdir(os.path.join(ori_img_folder, "ir")))
    print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
    print(model_name+'\t'+str(np.round(metric_result[0], 2))+'\t'
          +str(np.round(metric_result[1], 2))+'\t'
          +str(np.round(metric_result[2], 2))+'\t'
          +str(np.round(metric_result[3], 2))+'\t'
          +str(np.round(metric_result[4], 2))+'\t'
          +str(np.round(metric_result[5], 2))+'\t'
          +str(np.round(metric_result[6], 2))+'\t'
          +str(np.round(metric_result[7], 2))
    )
    print("="*80)
