# DOSS: Addressing Content Misalignment Issue by Comparing Order Statistics of Deep Features
----------------------------
This is the repository of paper [Full-Reference Image Quality Assessment: Addressing Content Misalignment Issue by Comparing Order Statistics of Deep Features](https:). Related Quality Assessment results are in `results' folder.

-----------------------------
## Updating log:
2023/6/30: Uploading the DOSS and supporting projection kernels. 

[Download Projection kernels](https://drive.google.com/file/d/1uBUMpy5NrhH4kLpWotsW4CvcrM0dfYgp/view?usp=sharing)

-----------------------------
## Requirements:
numpy==1.18.5

Pillow==8.2.0

torch==1.10.1

torchvision==0.14.1

------------------------------

## Useage:

>python DOSS.py --ref images/26-0.png --dist images/26-4.png --proj ./Projection_kernel.pth
------------------------------

## Citations:
