import segmentation_models_pytorch as smp
# import albumentations as albu
import torch
import utils as utils
import torchinfo as torchinfo
import os
from torchvision.io import read_image
import numpy as np


DEVICE = 'cpu'


def sigmoid(x):
    '''
    Args:
        x: N x 1 torch.FloatTensor
        
    Returns:
        y: N x 1 torch.FloatTensor
    '''
    y = None
    y = 1/(1+torch.exp(-x))
    return y


def softmax(x):
    '''
    Args:
        x: N x 1 torch.FloatTensor
        
    Returns:
        y: N x 1 torch.FloatTensor
    '''
    y = None
    y = torch.exp(x) / sum(torch.exp(x))
    return y

def apply_mask_to_image(image, mask):
    image = torch.from_numpy(image).type(torch.int64)
    mask = torch.from_numpy(mask).type(torch.int64)
    colors = {"red": torch.Tensor([255,0,0])}
    colored_mask = None
    final_seg_img = None

    m, n, c = mask.shape
    red = colors['red']
    colored_mask = torch.zeros(m, n, 3)
    
    reshape_mask = torch.cat((mask, mask, mask), 2)
    for i in range(360):
        for j in range(480):
            colored_mask[i][j] = torch.mul(mask[i][j], red)
            
    R = colored_mask[:,:,0]
    G = colored_mask[:,:,1]
    B = colored_mask[:,:,2]
    
    final_seg_img = R + G + B
    final_seg_img = final_seg_img[:, :, None]


    return final_seg_img

def load_FPN_resnet50():

    resnet50_model = smp.FPN(
        encoder_name='resnet50', 
        classes=1, 
        activation='sigmoid'
    )
    resnet50_model.load_state_dict(torch.load('./models/resnet50_best_model_weights.pt', map_location=torch.device('cpu')))

    return resnet50_model

def IoU(predict: torch.Tensor, target: torch.Tensor):

    denom = 0
    nume = 0
    row, col = predict.shape

    for i in range(row):
        for j in range(col):
            if predict[i][j] == target[i][j] and predict[i][j] > 0:
                nume += 1
                
            if predict[i][j] > 0 or target[i][j] > 0:
                denom += 1
                
    IoU = nume / denom
    
    return IoU

def applyIoU(model: smp.fpn.model, dataset: utils.Dataset):
    
    
    IoU_score = []
    
    for i in range(len(dataset)):
        image, true_mask = dataset[i]

        x_tensor = torch.from_numpy(image).to('cpu').unsqueeze(0)
        pred_mask = model.predict(x_tensor)
        pred_mask = (pred_mask.squeeze().cpu().numpy().round())
        true_mask = torch.tensor(true_mask[0,:,:])
        img_iou = IoU(pred_mask, true_mask)
        IoU_score.append(img_iou)
    
    assert type(true_mask) == torch.Tensor
    
    return IoU_score


def compare_psp_fpn(test_dataset):
 
    fpn = load_FPN_resnet50()
    psp = smp.PSPNet(
            encoder_name='resnet50', 
            classes=1, 
            activation='sigmoid')
    psp.load_state_dict(torch.load('./models/pspnet_resnet50_best_model_weights.pt', map_location=torch.device('cpu')))
    
    psp_iou = applyIoU(psp, test_dataset)
    fpn_iou = applyIoU(fpn, test_dataset)
    
    return psp_iou, fpn_iou

def load_model(decoder_weights_path=None):
    ENCODER = 'resnet50' 
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['car']
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )

    model.load_state_dict(torch.load('models/{}_best_model_weights.pt'.format(ENCODER), map_location=torch.device('cpu')))
    #model.load_state_dict(torch.load(decoder_weights_path, map_location=torch.device('cpu')))
    return model

def print_model_summary(model, channels=3, H=384, W=480):
    print(torchinfo.summary(model, input_size=(1, channels, H, W)))
    

def create_vis_dataset():
    # should paths and classes be defined here or in notebook?
    x_vis_dir = "./data/CamVid/test/"
    y_vis_dir = "./data/CamVid/testannot/"
    classes = ["car"]
    return utils.Dataset(x_vis_dir, y_vis_dir, classes)

def create_test_dataset():

    x_test_dir = "./data/CamVid/test/"
    y_test_dir = "./data/CamVid/testannot/"
    classes = ["car"]
    ENCODER = 'resnet50' 
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    test_dataset = utils.Dataset(
        x_test_dir, 
        y_test_dir, 
        augmentation=utils.get_validation_augmentation(), 
        preprocessing=utils.get_preprocessing(preprocessing_fn),
        classes=classes,
    )

    return test_dataset

def test_model(model, test_dataset, vis_dataset):

    for i in range(len(test_dataset)):

        image_vis = vis_dataset[i][0].astype('uint8')
        image, gt_mask = test_dataset[i]
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        print(x_tensor.shape)
        pr_mask = model.predict(x_tensor)
        print(f"Shape of predicted mask = {pr_mask.shape}")
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
        utils.visualize(
            image=image_vis, 
            ground_truth_mask=gt_mask, 
            predicted_mask=pr_mask
        )
    return image, pr_mask
