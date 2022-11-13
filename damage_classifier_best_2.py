import sys
sys.path.append('/work3/kniud/Voxel_grid/') 
from re import X
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional,neuron,layer
import numpy as np

from torch.utils.data import Dataset
import os
import torchvision.transforms as Tr

from collections import OrderedDict

import pytorch_lightning as pl
import torchmetrics
from torch.utils.data import DataLoader

import h5py
import argparse
import random

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from eventDataSet_voxel_mean_std import v2EDamageDataSet


cfgs = {'A' : [64,'M',128,'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
         'B' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']}


class VGGSNN(nn.Module):
    def __init__(self,num_of_inChannels,num_classes=4,init_weights=True,single_step_neuron : callable = None,**kwargs):
        super(VGGSNN,self).__init__()

        self.out_channels = []
        #self.idx_pool = [i for i,v in enumerate(cfg) if v=='M']
        #if norm_layer is None:
        #    norm_layer = nn.Identity
        bias = False
        """self.features = self.make_layers(num_of_inChannels,cfg=cfg,norm_layer=norm_layer,lifNeuron=single_step_neuron,bias=bias,**kwargs)"""

        representationType = 0
        if representationType == 0:
            single_step_neuron = neuron.ParametricLIFNode
        else:
            single_step_neuron = nn.ReLU


        affine_flag = True
        self.features = nn.Sequential(
            nn.Conv2d(num_of_inChannels, 64, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag),
            single_step_neuron(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag),
            single_step_neuron(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag),
            single_step_neuron(),

            nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag),
            single_step_neuron(),
            #self.pool4 = nn.AvgPool2d(kernel_size=2,stride=2)
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag),
            single_step_neuron(),

            nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag),
            single_step_neuron(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag),
            single_step_neuron(),

            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag),
            single_step_neuron(),
            #self.pool8 = nn.AvgPool2d(kernel_size=2,stride=2)
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag),
            single_step_neuron(),

            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag),
            single_step_neuron(),
            nn.MaxPool2d(kernel_size=2,stride=2))

        self.classifier = nn.Sequential(
                            nn.Conv2d(512,num_classes, kernel_size=1, stride=1, padding=1, bias=bias),
                            nn.BatchNorm2d(num_classes, eps=1e-4, momentum=0.1, affine=affine_flag),
                            single_step_neuron())
            
        self.lastbntt = nn.BatchNorm1d(num_classes, eps=1e-4, momentum=0.1, affine=affine_flag)

        """self.conv1 = nn.Conv2d(num_of_inChannels, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt1 = nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif1 = single_step_neuron(**kwargs)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt2 = nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif2 = single_step_neuron(**kwargs)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3 = nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt3 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif3 = single_step_neuron(**kwargs)

        self.conv4 = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt4 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif4 = single_step_neuron(**kwargs)

        #self.pool4 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv5 = nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt5 = nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif5 = single_step_neuron(**kwargs)
        
        self.conv6 = nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt6 = nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif6 = single_step_neuron(**kwargs)

        self.pool6 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv7 = nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt7 = nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif7 = single_step_neuron(**kwargs)

        self.conv8 = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt8 = nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif8 = single_step_neuron(**kwargs)

        #self.pool8 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.pool8 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv9 = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt9 = nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif9 = single_step_neuron(**kwargs)

        self.conv10 = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt10 = nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif10 = single_step_neuron(**kwargs)

        self.pool10 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv11 = nn.Conv2d(512,num_classes, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt11 = nn.BatchNorm2d(num_classes, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif11 = single_step_neuron(**kwargs)

        self.conv_list = [self.conv1,self.conv2,self.conv3,self.conv4,self.conv5,self.conv6,self.conv7,self.conv8, self.conv9,self.conv10,self.conv11]
        self.bntt_list = [self.bntt1,self.bntt2,self.bntt3,self.bntt4,self.bntt5,self.bntt6,self.bntt7,self.bntt8,self.bntt9,self.bntt10,self.bntt11]
        self.lif_list = [self.lif1,self.lif2,self.lif3,self.lif4,self.lif5,self.lif6,self.lif7,self.lif8,self.lif9,self.lif10,self.lif11]
        self.pool_list = [False,self.pool2,False,self.pool4,False,self.pool6,False,self.pool8,False,self.pool10,False]

        self.lastbntt = nn.BatchNorm1d(num_classes, eps=1e-4, momentum=0.1, affine=affine_flag)"""

        if init_weights:
            self._initialize_weights()
    
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    """def make_layers(self,num_of_inChannels,cfg,norm_layer,lifNeuron,bias,**kwargs):
        layers = []
        channel_in_info = num_of_inChannels
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
                self.out_channels.append(channel_in_info)
            else:
                #this may be unwrapped later        
                layers.append(nn.ConstantPad2d(padding=(1, 1, 1, 1), value=0.0))
                layers.append(nn.Conv2d(channel_in_info, v, kernel_size=(3, 3), stride=(1, 1), bias=False))
                layers.append(norm_layer(v))
                layers.append(lifNeuron(**kwargs))
                #if (v > 64):
                #    layers.append(nn.Dropout(0.1))
                channel_in_info = v
                
        self.out_channels = self.out_channels[2:]
        return nn.Sequential(*layers)"""
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight)
                m.threshold = 1.0
                torch.nn.init.xavier_uniform(m.weight,gain=2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Linear):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform(m.weight,gain=2)
            """elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)"""
    
    def add_hooks(self):
        def get_nz(name):
            def hook(model, input, output):
                self.nz[name] += torch.count_nonzero(output)
                self.numel[name] += output.numel()
            return hook
        
        self.hooks = {}
        for name, module in self.named_modules():
            self.nz[name], self.numel[name] = 0, 0
            self.hooks[name] = module.register_forward_hook(get_nz(name))
                
    def reset_nz_numel(self):
        for name, module in self.named_modules():
            self.nz[name], self.numel[name] = 0, 0
        
    def get_nz_numel(self):
        return self.nz, self.numel


class MultiStepVGGSNN(VGGSNN):
    def __init__(self, num_of_input_channels,representationType,num_classes=4, init_weights=True, timeSteps : int = None,
                 multi_step_neuron: callable = None, **kwargs):

        super().__init__(num_of_input_channels,num_classes,init_weights,multi_step_neuron, **kwargs)
        self.TimeSteps = timeSteps
        self.representationType = representationType
                 
    """def forward(self,x,classify = True):
        x_seq = x
        if x.dim() != 5:
            assert self.T is not None, 'When x.shape is [N, C, H, W], self.T can not be None.'
        
        if classify:
            #x_seq = functional.seq_to_ann_forward(x, self.features[0])
            z_seq_step_by_step = []
            for t in range(self.TimeSteps):
                x = x_seq[t]
                y = self.features(x)
                z = self.classifier(y)
                z_seq_step_by_step.append(z.unsqueeze(0))
            z_seq_step_by_step = torch.cat(z_seq_step_by_step, 0)
            z_seq_step_by_step = z_seq_step_by_step.flatten(start_dim=-2).sum(-1)
            return z_seq_step_by_step"""
    
    def event_cube_forward(self,x):

        if x.dim() != 5:
            assert self.T is not None, 'When x.shape is [N, C, H, W], self.T can not be None.'
        
        z_seq_step_by_step = []

        for t in range(self.TimeSteps):
            out = x[t]
            out = self.features(out)
            out = self.classifier(out)
            z_seq_step_by_step.append(out.unsqueeze(0))

        z_seq_step_by_step = torch.cat(z_seq_step_by_step, 0)
        z_seq_step_by_step = z_seq_step_by_step.flatten(start_dim=-2).sum(-1)
        z_seq_step_by_step = z_seq_step_by_step.permute(1,0,2)
        z_seq_step_by_step = z_seq_step_by_step.sum(dim=1)
        z_seq_step_by_step = self.lastbntt(z_seq_step_by_step)

        return z_seq_step_by_step
    
    def forward(self,x):

        if self.representationType == 0: #event cube
            return self.event_cube_forward(x)
        elif self.representationType == 1: # voxel grid
            if x.dim() != 4:
                assert 'not valid input , for voxel grid [N,Tbins,H,W]'
            out = x
            out = self.features(out)
            out = self.classifier(out)
            out = out.flatten(start_dim=-2).sum(-1)
            #out = self.lastbntt(out)
            return out
        elif self.representationType == 2: #
            if x.dim() != 6:
                assert 'not valid input , for mean stad represent [N,6,H,W]'
            out = x
            out = self.features(out)
            out = self.classifier(out)
            out = out.flatten(start_dim=-2).sum(-1)
            #out = self.lastbntt(out)
            return out

class ClassificationLitModule(pl.LightningModule):
    def __init__(self, model,cfg,representationType = 0, epochs=10, lr=5e-3, num_classes=4):
        super().__init__()
        self.save_hyperparameters()
        self.lr, self.epochs = lr, epochs
        self.num_classes = num_classes
        self.representationType = representationType
        self.cfg = cfg

        self.model = model

        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.train_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none")
        self.val_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none")
        self.test_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none")
        self.train_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.val_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.test_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)
    
    def forward(self, x):
        
        if self.representationType == 0:
            x = x.permute(1,0,2,3,4)

        retVal = self.model(x)
        print("input shape for the classLit Forward ",x.shape)
        print("ClassLit return value shape = ",retVal.shape)
        #retVal = self.model(x).sum(dim=1)
        #retVal = retVal.permute(1,0,2)
        #retVal = retVal.sum(dim=1)
        return retVal
    
    def step(self, batch, batch_idx, mode):
        events, target = batch
        #print("batch_shape  ",batch[0].shape)
        #print("batch target shape  ",batch[1].shape)
        print("printing target ",target)
        outputs = self(events)
        #print("eventssssssss = ", events.shape)
        #print("outputsttttttttt ",outputs.shape)
        
        loss = nn.functional.cross_entropy(outputs, target)
        print("loss is ",loss)
        
        # Measure sparsity if testing
        #if mode=="test":
        #    self.process_nz(self.model.get_nz_numel())

        # Metrics computation
        sm_outputs = outputs.softmax(dim=-1)
        print("printing softmax outputs =",sm_outputs)
        
        """selected_out = sm_outputs[target == 4]
        print("selected output ..",selected_out)
        if selected_out.numel() != 0:
            t_list = [0,1,2,3]
            for i in t_list:
                loss += 2*torch.max(torch.tensor(0,device="cuda:0"),torch.sum(selected_out[:,i] - selected_out[:,4]))"""

        
        acc, acc_by_class = getattr(self, f"{mode}_acc"), getattr(self, f"{mode}_acc_by_class")
        confmat = getattr(self, f'{mode}_confmat')

        #print("sm outputs ",sm_outputs)
        acc(sm_outputs, target)
        acc_by_class(sm_outputs, target)
        confmat(sm_outputs, target)

        if mode != "test":
            self.log(f'{mode}_loss', loss, on_epoch=True, prog_bar=(mode == "train"))
        if mode == "test":
            mode_acc, mode_acc_by_class = getattr(self, f"{mode}_acc"), getattr(self, f"{mode}_acc_by_class")
            acc, acc_by_class = mode_acc.compute(), mode_acc_by_class.compute()
            for i,acc_i in enumerate(acc_by_class):
                self.log(f'{mode}_acc_{i}', acc_i)
                self.log(f'{mode}_acc', acc)
            
            print(f"{mode} accuracy: {100*acc:.2f}%")
            print(f"spalling {100*acc_by_class[0]:.2f}% - healthy {100*acc_by_class[1]:.2f}% - crack {100*acc_by_class[2]:.2f}% - corrosion {100*acc_by_class[3]:.2f}%")
            mode_acc.reset()
            mode_acc_by_class.reset()
        
        functional.reset_net(self.model)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="train")
        
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="val")
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="test")
    
    def on_mode_epoch_end(self, mode):
        print()
        mode_acc, mode_acc_by_class = getattr(self, f"{mode}_acc"), getattr(self, f"{mode}_acc_by_class")
        acc, acc_by_class = mode_acc.compute(), mode_acc_by_class.compute()
        for i,acc_i in enumerate(acc_by_class):
            self.log(f'{mode}_acc_{i}', acc_i)
        
        self.log(f'{mode}_acc', acc)
        
        print(f"{mode} accuracy: {100*acc:.2f}%")
        print(f"spalling {100*acc_by_class[0]:.2f}% - healthy {100*acc_by_class[1]:.2f}% - crack {100*acc_by_class[2]:.2f}% - corrosion {100*acc_by_class[3]:.2f}%")
        mode_acc.reset()
        mode_acc_by_class.reset()
        print(f"{mode} confusion matrix:")
        self_confmat = getattr(self, f"{mode}_confmat")
        confmat = self_confmat.compute()
        #self.log(f'{mode}_confmat', confmat)
        print(confmat)
        self_confmat.reset()

        if mode=="test":
            print(f"Total sparsity: {self.all_nnz} / {self.all_nnumel} ({100 * self.all_nnz / self.all_nnumel:.2f}%)")
            self.all_nnz, self.all_nnumel = 0, 0

    def process_nz(self, nz_numel):
        nz, numel = nz_numel
        total_nnz, total_nnumel = 0, 0

        for module, nnz in nz.items():
            if "act" in module:
                nnumel = numel[module]
                if nnumel != 0:
                    total_nnz += nnz
                    total_nnumel += nnumel
        if total_nnumel != 0:
            self.all_nnz += total_nnz
            self.all_nnumel += total_nnumel

    def on_train_epoch_end(self):
        return self.on_mode_epoch_end(mode="train")

    def on_test_epoch_end(self):
        return self.on_mode_epoch_end(mode="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate,weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    """def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr = self.cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.cfg.step_size, gamma=0.1, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            "monitor": "val_loss"
        }"""
def collate_fn(batch):     # This collate function handle empty samples returns to the dataloader by by CustomDataset
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-device',default=0,type=int,help='device')
    parser.add_argument('-no_train',action='store_false',help="once this arg is added train will not run",dest='train')
    parser.add_argument('-test',action='store_true',help="once add this arg test will run")
    parser.add_argument('-pretrained',default=None,type=str,help='path to pretrained model')
    parser.add_argument('-data_build_only',action='store_true',help='for building the dataset only ')
    parser.add_argument("--learning_rate", type=float, default=0.004)#    1e-4)
    parser.add_argument("--step_size", type=int, default=15)

    args = parser.parse_args()
    print(args)

    if args.data_build_only:
        return

    callbacks=[]
    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f"ckpt-damage-classifier-vgg/",
        filename=f"damage-classifier-vgg" + "-{epoch:02d}-{train_acc:.4f}",
        save_top_k=3,
        mode='min',
    )
    callbacks.append(ckpt_callback)

    logger = None
    try:
        comet_logger = CometLogger(
        api_key=None,
        project_name=f"classif-damage-classifier-vgg/",
        save_dir="comet_logs",
        log_code=True,
        )
        logger = comet_logger
    except ImportError:
        print("Comet is not installed, Comet logger will not be available.")
    

    trainer = pl.Trainer(
        accelerator='gpu',devices=[0],gradient_clip_val=1., max_epochs=35,
        limit_train_batches=1., limit_val_batches=1.,
        check_val_every_n_epoch=1,
        deterministic=False,
        precision=16,
        callbacks=callbacks,
        logger=logger,
    )


    folder_path_for_event_data  = "/work3/kniud/Voxel_grid/ClassificationSetNew_old_Full/" 
    folder_path_2 = "/work3/kniud/Voxel_grid/tunnel_event_extracted/"
    representationType = 0

    trainCSVFile = folder_path_for_event_data + "/train/damageClassesTrainEventData.csv"
    testCSVFile = folder_path_for_event_data + "/test/damageClassesTestEventData.csv"

    testCSVFile3 = folder_path_2 + "/test/damageClassesTestEventData2.csv"
    image_shape = (346,260)
    numOfTimeSteps1 = 5
    numOfTbins1 = 2

    train_transform = Tr.Compose([Tr.RandomRotation(5),Tr.RandomHorizontalFlip(p=0.5),Tr.RandomVerticalFlip(p=0.5)])

    trainDataSet = v2EDamageDataSet("train",folder_path_for_event_data,trainCSVFile,representationType,image_shape,"v",numOfTimeSteps = numOfTimeSteps1,sampleSize = 50000,numOfTbins = numOfTbins1,transform = train_transform)
    train_dataloader = DataLoader(trainDataSet, batch_size=16, num_workers=4,shuffle=True,drop_last=True)

    testDataSet = v2EDamageDataSet("test",folder_path_for_event_data,testCSVFile,representationType,image_shape,"v",numOfTimeSteps = numOfTimeSteps1,sampleSize = 50000,numOfTbins = numOfTbins1)
    test_dataloader = DataLoader(testDataSet, batch_size=8, num_workers=4,drop_last=True)

    testDataSet3 = v2EDamageDataSet("test",folder_path_2,testCSVFile3,representationType,image_shape,"v",numOfTimeSteps = numOfTimeSteps1,sampleSize = 50000,numOfTbins = numOfTbins1)
    test_dataloader3 = DataLoader(testDataSet3, batch_size=8, num_workers=4,drop_last=True)

    
    number_of_classes = 4

    if (representationType == 0):
        channels_per_timeStep = 4
        ms_neuron = neuron.ParametricLIFNode #step_mode='m',backend='cupy') #accelerate the processing in GPU with cupy
        model = MultiStepVGGSNN(channels_per_timeStep,representationType,num_classes=number_of_classes,init_weights=True,timeSteps=5,multi_step_neuron = ms_neuron)#,step_mode='s')
    else:
        channels = 6 #histo mean std channels 

        if representationType == 1:
            channels = numOfTimeSteps1 * numOfTbins1 #Number of voxel bins
        ms_neuron = nn.ReLU()
        model = MultiStepVGGSNN(channels,representationType,num_classes=number_of_classes,init_weights=True,timeSteps=5,multi_step_neuron = ms_neuron)
    
    module = ClassificationLitModule(model,args,representationType, epochs=35, lr=args.learning_rate,num_classes=number_of_classes)

    if args.pretrained is not None:
        ckpt_path = args.pretrained
        #module = module.load_from_checkpoint(checkpoint_path=ckpt_path,strict=False)
        #checkpoint = torch.load(ckpt_path)
        module.model.load_state_dict(torch.load(ckpt_path))

    if args.train:
        trainer.fit(module,train_dataloader,test_dataloader)
        torch.save(model.state_dict(),"./final_model_voxel_cube_0.005.pth")
    if args.test:
        test_dataloader = DataLoader(testDataSet, batch_size=16, num_workers=4)
        trainer.test(module,test_dataloader)
                
if __name__ == '__main__':
    main()
