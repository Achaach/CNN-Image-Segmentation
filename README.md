# CNN-Image-Segmentation

## Introduction
Image segmentation is the process of partitioning a digital image into multiple segments (sets of pixels, also known as image objects). In image recognition system, segmentation helps to extract the object of interest from an image which is further used for processing like recognition and description. Image segmentation is the practice for classifying the image pixels.

We are going to focus on semantic image segmentation with a state-of-the-art approach: deep learning.

## Basic objectives:

- Basic segmentation
    - Understanding the goal of image segmentation.
    - Understanding sigmoid and softmax functions.
    - Applying segmentation masks to images.

- Architechture of segmentation model
    - Fully Convolutional Networks and its backbones (VGG-19 and ResNet-50)
    - Looking into feature maps generated at different parts of the network.

- Model performance
    - IoU.
    - Comparing performance for two different models
    


## Results

### Mask:

<img width="1021" alt="mask" src="https://user-images.githubusercontent.com/90078254/219511463-37a602de-a93b-46f6-bd8c-94c61436b5a1.png">

### IoU score for VGG-19 and ResNet- 50:
Vgg19 IoU score is: [0.9223450193954476, 0.9311584670700693, 0.9414837244511733]

ResNet50 IoU score is: [0.8864403129710717, 0.909509735744089, 0.9392925066565234]

VGG-19 has a better performance on these testing images with a bigger IoU score.

###  IoU score for PSPNet-ResNet50 and FPN-ResNet50:

PSPNet-ResNet50 IoU score is: [0.8106170240757795, 0.9054329371816638, 0.9005265815480055]

FPN-ResNet50 IoU score is: [0.8864403129710717, 0.909509735744089, 0.9392925066565234]
