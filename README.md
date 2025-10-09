# EA-CDDFuse-and-LI-Unet-models
The relevant reproduction codes of the multimodal image fusion model EA-CDDFuse and the semantic segmentation model LI-Unet. Among them, the EA-CDDFuse model accepts visible light images and near-infrared images to generate fused images. Then, the LI-Unet model performs semantic segmentation on the parts of interest of the fused image. All the codes will be gradually made public after the paper is published.

## Network Introduction
Aiming at the problem of low obstacle detection accuracy of unmanned electric locomotives in coal mines in low-illumination environments, an obstacle detection method based on the fusion of near-infrared and visible light is proposed. The front end of this method fuses near-infrared and visible light images through the EA-CDDFuse model to generate a fused image containing rich texture details and complete semantic information, effectively solving the defect of visible light single-modal perception in low-illumination scenes. The back end conducts cross-scale feature analysis on the fused image based on the LI-UNet model, and achieves precise semantic segmentation of obstacles through multi-module collaborative optimization.

This is the architecture of the EA-CDDFuse multimodal image fusion network.

![](./figures/EACDDFuse.png?msec=1759325406919)

This is the architecture of the LI-Unet semantic segmentation network.

![](./figures/LIUnet.png?msec=1759325406919)

## Visual display
EA-CDDFuse is compared with different multimodal image fusion models.  
Contrast model including SDNet, U2Fusion, DIDFuse CDDFuse, DATFuse, CreossFuse, SwinFuse.

![](./figures/EACDDFuse_compare.png?msec=1759325406919)

LI-Unet is compared with different semantic segmentation models.   
Contrast model including LR - ASPP, FCN SegFormer, PSPNet, DeepLabV3 +, DDRNet, techches - Unet, Unet.

![](./figures/LIunet_compare.png?msec=1759325406919)

## Start quickly
**python≥3.8 torch≥1.11**   
Installation dependency environment  
```bash
pip install -r requirements.txt
```
Run the following code in the project directory to test the fusion effect of EA-CDDFuse:

```bash
python test_for_EACDD.py
```

Run the following code in the project directory to test the fusion splitting effect of LI-Unet:

```bash
python test_for_LIUnet.py
```
Conduct real-time performance tests on the end-to-end model：

```bash
python test_fps.py
```
