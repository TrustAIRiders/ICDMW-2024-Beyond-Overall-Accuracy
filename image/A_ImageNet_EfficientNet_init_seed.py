#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['MKL_NUM_THREADS'] = '45'
os.environ['NUMEXPR_NUM_THREADS'] = '45'
os.environ['OMP_NUM_THREADS'] = '45'


# In[1]:


import os
import numpy as np
import torch
import torchvision
import torchmetrics
import wandb

from datetime import datetime 
from imagenet_kaggle_dataset import ImageNetKaggle

import pytorch_lightning
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning import Trainer, LightningDataModule, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# In[ ]:


torch.set_float32_matmul_precision("high")
DATASET_PATH = "/ImageNet2012/"


# In[ ]:


class DataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.num_class = 1000
        self.labels = self.get_labels()
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)  
        
    def get_labels(self):
        with open(DATASET_PATH+"/LOC_synset_mapping.txt") as file:
            labels = ["_".join(line.rstrip().split(",")[0].replace(" ", "_").replace("-", "_").split("_")[:3]) for line in file]

        return labels

    def train_dataloader(self):
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(380),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.TrivialAugmentWide(interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
            torchvision.transforms.RandomErasing(p=0.1)
        ])
        
        
        train_set = ImageNetKaggle(DATASET_PATH, "train", transform=train_transforms)
        return torch.utils.data.DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)

    def val_dataloader(self):
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(384),
            torchvision.transforms.CenterCrop(380),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        
        test_set = ImageNetKaggle(DATASET_PATH, "val", transform=test_transforms)
        return torch.utils.data.DataLoader(test_set, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True) 
    
    def test_dataloader(self):
        return self.val_dataloader()


# In[ ]:


def mixup(x, y):
    ratio = np.random.beta(0.2, 0.2)
    rand_idx = torch.randperm(x.size(0)).to(x.device)

    return (ratio * x) + ((1 - ratio) * x[rand_idx]), y, y[rand_idx], ratio    

class LabelSmoothing(torch.nn.Module):
    def __init__(self, alpha=0.1):
        super(LabelSmoothing, self).__init__()
        self.alpha = alpha
        self.certainty = 1.0 - alpha
        self.criterion = torch.nn.KLDivLoss(reduction='batchmean')

    def forward(self, x, y):
        b, c = x.shape
        label = torch.full((b, c), self.alpha / (c - 1)).to(y.device)
        label = label.scatter(1, y.unsqueeze(1), self.certainty)
        return self.criterion(torch.nn.functional.log_softmax(x, dim=1), label)


# In[ ]:


class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = None
        self.criterion = None
        self.metric_accuracy = None
        self.metric_accuracy_top5 = None
        self.metric_accuracy_per_class = None
        self.metric_accuracy_per_class = None

    def init(self, model, data):
        self.model = model
        self.criterion = LabelSmoothing()
        
        self.metric_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=data.num_class, top_k=1)
        self.metric_accuracy_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=data.num_class, top_k=5)

        self.metric_accuracy_per_class = torchmetrics.ClasswiseWrapper(
            torchmetrics.Accuracy(task="multiclass", num_classes=data.num_class, average=None, top_k=1),
            labels=data.labels
        )
        
        self.metric_accuracy_per_class = torchmetrics.ClasswiseWrapper(
            torchmetrics.Accuracy(task="multiclass", num_classes=data.num_class, average=None, top_k=5),
            labels=data.labels        
        )
        
    def configure_optimizers(self):
        if CONFIG["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=0.0125, momentum=0.9, weight_decay=2e-05)
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0125, pct_start=0.1, total_steps=stepping_batches)
            lr_schedulers = {"scheduler": scheduler, "interval": "step"}
            return [optimizer], [lr_schedulers]              
        
    def forward(self, x):
        return self.model(x)   
    
    def test_val_step(self, batch, name):
        inputs, targets = batch
        
        preds = self.model(inputs)
        loss = self.criterion(preds, targets)
        acc = self.metric_accuracy(preds, targets)
        acc5 = self.metric_accuracy_top5(preds, targets)

        self.log("acc/{}".format(name), acc, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("acc_top5/{}".format(name), acc5, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("loss/{}".format(name), loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)       

        if name == "test":
            acc_per_class = self.metric_accuracy_per_class(preds, targets)
            for key, value in acc_per_class.items():
                key = "_".join(key.split("_")[1:])
                self.log("_final/{}".format(key), value, on_step=False, on_epoch=True, logger=True, sync_dist=True)  

            acc_per_class = self.metric_accuracy_top5_per_class(preds, targets)
            for key, value in acc_per_class.items():
                key = "_".join(key.split("_")[1:])
                self.log("_final_top5/{}".format(key), value, on_step=False, on_epoch=True, logger=True, sync_dist=True)  
                    
        return loss
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        
        x, y1, y2, ratio = mixup(inputs, targets)
        preds = self.model(x)
        loss = self.criterion(preds, y1) * ratio + self.criterion(preds, y2) * (1 - ratio)
        
        self.log("loss/train", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)    
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        return self.test_val_step(batch, "val") 
    
    def test_step(self, batch, batch_idx):
        return self.test_val_step(batch, "test")               
            
def main(CONFIG): 
    seed_everything(CONFIG["seed"], workers=True)
    
    model = torchvision.models.efficientnet_b4()
    model = torch.compile(model)
    
    callbacks = []
    logger = WandbLogger(
        project="Experiments",
        name="{}__seed_{}__{}".format(CONFIG["name"], CONFIG["seed"], datetime.utcnow()),
        config=CONFIG
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    if CONFIG["early_stopping"] > 0:
        early_stopping = EarlyStopping(monitor="loss/train", mode="min", patience=CONFIG["early_stopping"])
        callbacks.append(early_stopping)

    data = DataModule()
    lightning_model = Model()
    lightning_model.init(model, data)
    
    trainer = Trainer(max_epochs=CONFIG["max_epochs"], accelerator="gpu", devices=CONFIG["devices"], logger=logger, callbacks=callbacks, deterministic=True, strategy="auto")
    trainer.fit(lightning_model, datamodule=data)
    trainer.save_checkpoint("./saved_models/{}.ckpt".format(CONFIG["name"]))
    
    torch.distributed.destroy_process_group()
    if trainer.global_rank == 0:
        model = torch.load("./saved_models/{}.ckpt".format(CONFIG["name"]))["model"]
        
        data = DataModule()
        lightning_model = Model()
        lightning_model.init(model, data)
        
        trainer = Trainer(devices=CONFIG["devices"][0])
        trainer.test(lightning_model, datamodule=data)


# In[ ]:


CONFIG = {
        "name": "EfficientNet_B4_ImageNet2012_seed_0",
        "devices": [2, 3],
        "num_workers": 45,
        "seed": 0,
        "batch_size": 128,
        "max_epochs": 75,
        "early_stopping": 15,
        "optimizer": "SGD"
}

#main(CONFIG)


# In[ ]:


CONFIG = {
        "name": "long_EfficientNet_B4_ImageNet2012_seed_1",
        "devices": [2, 3],
        "num_workers": 45,
        "seed": 1,
        "batch_size": 128,
        "max_epochs": 200,
        "early_stopping": 15,
        "optimizer": "SGD"
}

#main(CONFIG)


# In[ ]:


CONFIG = {
        "name": "long_EfficientNet_B4_ImageNet2012_seed_2",
        "devices": [2, 3],
        "num_workers": 45,
        "seed": 2,
        "batch_size": 128,
        "max_epochs": 200,
        "early_stopping": 15,
        "optimizer": "SGD"
}

main(CONFIG)

