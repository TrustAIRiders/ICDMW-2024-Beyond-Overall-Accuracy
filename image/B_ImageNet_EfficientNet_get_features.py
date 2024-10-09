#!/usr/bin/env python
# coding: utf-8

import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

os.environ["CUDA_VISIBLE_DEVICES"]=str(2) 

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
import random
import numpy as np
import pandas as pd

import torch
import torchvision
from torchvision.models import vit_l_16, ViT_L_16_Weights
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from torchvision.models.feature_extraction import create_feature_extractor

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from pathlib import Path
from imagenet_kaggle_dataset import ImageNetKaggle


#


np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic  = True
torch.set_num_threads(1)

torch.set_float32_matmul_precision("high")


#


device = torch.device("cuda:0")

DATASET_PATH = "/datasets/"
DATASET_IMAGE_NET_2012_PATH = "{}/{}".format(DATASET_PATH, "ImageNet2012")

CONFIG = {
    "batch_size": 128,
    "num_workers": 1
}

with open(DATASET_IMAGE_NET_2012_PATH+"/LOC_synset_mapping.txt") as file:
    labels = [" ".join(line.split(" ")[1:]).replace("\n", "") for line in file]


#

def save_features(save_path, loader, model, fun_get_classifier, fun_get_features):
    all_out = []
    data_id = 0
    with torch.no_grad():
        for i_batch, (data, targets) in enumerate(loader):                
            sys.stdout.write("{}/{}\r".format(i_batch, len(loader)))
            sys.stdout.flush()

            data = data.to(device)
            outputs = fun_get_classifier(model, data)
            features = fun_get_features(model, data)
            
            for i in range(len(data)):
                out = {}
                out["id"] = data_id
                out["original_label"] = targets[i].item()
                out["features"] = np.array(features[i].detach().cpu())
                out["classifier"] = np.array(outputs[i].detach().cpu())
                all_out.append(out)
                data_id += 1
                
    df = pd.DataFrame(all_out)
    
    # Save pickle
    directory = os.path.dirname(path)
    Path(directory).mkdir(parents=True, exist_ok=True)

    np.set_printoptions(suppress=True, threshold=np.inf, precision=8, floatmode="maxprec_equal")
    df = df.rename(index={0: "id", 1: "original_label", 2: "features", 3: "classifier"})
    df.to_pickle(save_path)


#

def prepare__EfficientNet_B4(seed):
    if seed == 0:
        checkpoint_path = "./Experiments/r2k8qcm0/checkpoints/epoch=199-step=1001000.ckpt"
        
    if seed == 1:
        checkpoint_path = "./Experiments/d4g55mqt/checkpoints/epoch=199-step=1001000.ckpt"
        
    if seed == 2:
        checkpoint_path = "./Experiments/pvgmfoc9/checkpoints/epoch=199-step=1001000.ckpt"       
    
    state_dict = torch.load(checkpoint_path, map_location=device)["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace('model._orig_mod.', '')] = state_dict.pop(key)
        
        
    model = torchvision.models.efficientnet_b4()
    #model = torch.compile(model)
    model.load_state_dict(state_dict) 
    model = model.to(device)
    model = model.eval()
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)  
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(384),
        torchvision.transforms.CenterCrop(380),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    def fun_get_classifier(model, data):
        return model(data)    
    
    def fun_get_features(model, data):
        x = data
        sub_model = create_feature_extractor(model, return_nodes=["avgpool"])
        return sub_model(data)["avgpool"].view(data.shape[0], -1)
        

    return ("EfficientNet_B4_long_seed_{}".format(seed), model, preprocess, fun_get_classifier, fun_get_features)

def prepare__EfficientNet_B4_seed_0():
    return prepare__EfficientNet_B4(seed=0)

def prepare__EfficientNet_B4_seed_1():
    return prepare__EfficientNet_B4(seed=1)

def prepare__EfficientNet_B4_seed_2():
    return prepare__EfficientNet_B4(seed=2)

#

def prepare__ImageNetTrain(preprocess):
    loader_name = "train"
    _set = ImageNetKaggle(DATASET_IMAGE_NET_2012_PATH, "train", transform=preprocess)
    loader = torch.utils.data.DataLoader(_set, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)         
    return loader_name, loader

def prepare__ImageNetTest(preprocess):
    loader_name = "test"
    _set = ImageNetKaggle(DATASET_IMAGE_NET_2012_PATH, "val", transform=preprocess)
    loader = torch.utils.data.DataLoader(_set, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)         
    return loader_name, loader

#

fun_models = [
    prepare__EfficientNet_B4_seed_0,
    prepare__EfficientNet_B4_seed_1,
    prepare__EfficientNet_B4_seed_2
]

fun_loaders = [
    prepare__ImageNetTrain,
    prepare__ImageNetTest,
]

for fun_model in fun_models:
    model_name, model, preprocess, fun_get_classifier, fun_get_features = fun_model()

    print(model_name)
    for fun_loader in fun_loaders: 
        loader_name, loader = fun_loader(preprocess)

        path = "./features/{}/".format(model_name)
        save_path = "{}/{}.pickle".format(path, loader_name)
        if not os.path.isfile(save_path):
            save_features(save_path, loader, model, fun_get_classifier, fun_get_features)
            print(">>> Saved: {}".format(save_path))
            torch.cuda.empty_cache()
        else:
            print(">>> File exists: {}".format(save_path))