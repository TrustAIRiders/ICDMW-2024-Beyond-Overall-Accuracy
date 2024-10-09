#!/usr/bin/env python
# coding: utf-8

import sys
_cuda_device = str(sys.argv[1]) if len(sys.argv) > 1 else "0"
_model = str(sys.argv[2]) if len(sys.argv) > 2 else ""

print("argv:", _cuda_device, _model)

#

import os
os.environ['MKL_NUM_THREADS'] = '20'
os.environ['NUMEXPR_NUM_THREADS'] = '20'
os.environ['OMP_NUM_THREADS'] = '20'

os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_device

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


#

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

import clip
import open_clip

from pathlib import Path


#


np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic  = True
torch.set_num_threads(1)

torch.set_float32_matmul_precision("high")


# %%


device = torch.device("cuda:0")

DATASET_PATH = "/datasets/"
DATASET_IMAGE_NET_2012_PATH = "{}/{}".format(DATASET_PATH, "ImageNet2012")
DATASET_IMAGE_NET_C_PATH = "{}/{}".format(DATASET_PATH, "ImageNet-C")

CONFIG = {
    "batch_size": 1024,
    "num_workers": 1
}

with open(DATASET_IMAGE_NET_2012_PATH+"/LOC_synset_mapping.txt") as file:
    label_mapping, labels = {}, []
    for i, line in enumerate(file):
        label_mapping[line.split(" ")[0]] = i
        labels.append(" ".join(line.split(" ")[1:]).replace("\n", ""))

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
            
            for i in range(len(data)):
                out = {}
                out["id"] = data_id
                out["original_label"] = targets[i].item()
                out["features"] = []
                out["classifier"] = np.array(outputs[i].detach().cpu())
                all_out.append(out)
                data_id += 1
                
    df = pd.DataFrame(all_out)
    
    # Save pickle
    np.set_printoptions(suppress=True, threshold=np.inf, precision=8, floatmode="maxprec_equal")
    df = df.rename(index={0: "id", 1: "original_label", 2: "features", 3: "classifier"})
    df.to_pickle(save_path)

#

def prepare__ResNet152():
    model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
    preprocess = ResNet152_Weights.IMAGENET1K_V2.transforms()
    
    model = model.to(device)

    def fun_get_classifier(model, data):
        return model(data)    
    
    def fun_get_features(model, data):
        x = data
        for name, module in model._modules.items():
            x = module(x)
            if name == 'avgpool':
                x = x.view(x.shape[0], -1)
                return x

    return ("ResNet152", model, preprocess, fun_get_classifier, fun_get_features)

def prepare__ViT_L_16():
    model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
    preprocess = ViT_L_16_Weights.IMAGENET1K_V1.transforms()
    
    model = model.to(device)

    def fun_get_classifier(model, data):
        return model(data)    
    
    def fun_get_features(model, data):
        x = model._process_input(data)
        n = x.shape[0]

        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = model.encoder(x)
        x = x[:, 0]
        return x

    return ("ViT_L_16", model, preprocess, fun_get_classifier, fun_get_features)

def prepare__EfficientNet_V2():
    model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    preprocess = EfficientNet_V2_M_Weights.IMAGENET1K_V1.transforms()
    
    model = model.to(device)
    
    def fun_get_classifier(model, data):
        return model(data)    
    
    def fun_get_features(model, data):
        x = data
        for name, module in model._modules.items():
            x = module(x)
            if name == 'avgpool':
                x = x.view(x.shape[0], -1)
                return x

    return ("EfficientNet_V2", model, preprocess, fun_get_classifier, fun_get_features)

def prepare__ConvNeXt_Large():
    model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
    preprocess = ConvNeXt_Large_Weights.IMAGENET1K_V1.transforms()
    
    model = model.to(device)
    
    def fun_get_classifier(model, data):
        return model(data)    
    
    def fun_get_features(model, data):
        x = data
        for name, module in model._modules.items():
            x = module(x)
            if name == 'avgpool':
                x = x.view(x.shape[0], -1)
                return x

    return ("ConvNeXt_Large", model, preprocess, fun_get_classifier, fun_get_features)


# %%


class ImageNetCDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, corruption, severity, transform=None, label_mapping=label_mapping):
        self.root_dir = root_dir
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        self.label_mapping = label_mapping
        self.image_paths = []
        self.labels = []
        self.load_images_and_labels()

    def load_images_and_labels(self):
        corruption_path = os.path.join(self.root_dir, self.corruption, str(self.severity))
        for class_folder in os.listdir(corruption_path):
            class_path = os.path.join(corruption_path, class_folder)
            for image_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, image_name))
                if self.label_mapping:
                    self.labels.append(self.label_mapping[class_folder])
                else:
                    self.labels.append(class_folder)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def prepare__ImageNetC(corruption, severity, preprocess):
    loader_name = "ImageNetC_{}_{}".format(corruption, severity)
    _set = ImageNetCDataset(DATASET_IMAGE_NET_C_PATH, corruption, severity, transform=preprocess)
    loader = torch.utils.data.DataLoader(_set, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)         
    return loader_name, loader


# %%


fun_models = [
    prepare__ResNet152,
    prepare__ViT_L_16,
    prepare__EfficientNet_V2,
    prepare__ConvNeXt_Large
]

corruption_types = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
    "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
    "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"
]

severity_levels = ["1", "2", "3", "4", "5"]
corruption_severity_tuples = [(corruption, severity) for corruption in corruption_types for severity in severity_levels]

for fun_model in fun_models:
    model_name, model, preprocess, fun_get_classifier, fun_get_features = fun_model()
    if not (_model == "" or _model in model_name):
        continue

    print(model_name)
    for (corruption, severity) in corruption_severity_tuples: 
        loader_name, loader = prepare__ImageNetC(corruption, severity, preprocess)

        path = "./features/ImageNetC/{}/".format(model_name)
        save_path = "{}/{}.pickle".format(path, loader_name)
        if not os.path.isfile(save_path):
            directory = os.path.dirname(path)
            Path(directory).mkdir(parents=True, exist_ok=True)
            open(save_path, "w").close()

            save_features(save_path, loader, model, fun_get_classifier, fun_get_features)
            print(">>> Saved: {}".format(save_path))
            torch.cuda.empty_cache()
        else:
            print(">>> File exists: {}".format(save_path))
            