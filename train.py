import argparse
import logging
import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import optim
import random
import pickle
from sklearn.preprocessing import normalize
from PIL import Image

from datasets.cifar_dataset import CIFARDataset
from datasets.dataset import StandardDataset
from evaluate import evaluate_val
from models.backbone import encoder32
from models.backbone_resnet import encoder
from models.backbone_wide_resnet import wide_encoder
from penalties import compute_rpl_loss
from utils import count_parameters, setup_logger


    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epochs", type=int,
                        help="number of epochs to train", default=300)

    parser.add_argument("--gap", type=str,
                        help="TRUE iff use global average pooling layer. Otherwise, use linear layer.", default="TRUE")
    parser.add_argument("--lr_scheduler", type=str,
                        help="patience, step.", default="patience")

    parser.add_argument("--latent_size", type=int,
                        help="Dimension of embeddings.", default=256)
    parser.add_argument("--num_rp_per_cls", type=int,
                        help="Number of reciprocal points per class.", default=1)
    parser.add_argument("--lamb", type=float,
                        help="how much to weight the open-set regularization term in objective.", default=0.1)
    parser.add_argument("--gamma", type=float,
                        help="how much to weight the probability assignment.", default=1)

    parser.add_argument("--divide", type=str,
                        help="TRUE or FALSE, as to whether or not to divide loss by latent_size for convergence.",
                        default="TRUE")

    parser.add_argument("--dataset", type=str,
                        help="CIFAR_PLUS, TINY, or IMAGENET, or LT", default="CIFAR_PLUS")

    parser.add_argument("--dataset_folder", type=str,
                        help="name of folder where dataset details live.", default="cifar10_splits/split0/")

    parser.add_argument("--batch_size", type=int,
                        help="size of a batch during training", default=64)
    parser.add_argument("--lr", type=float,
                        help="initial learning rate during training", default=0.01)
    parser.add_argument("--patience", type=int,
                        help="patience of lr scheduler", default=50)
    parser.add_argument("--img_size", type=int,
                        help="desired square image size.", default=32)
    parser.add_argument("--num_workers", type=int,
                        help="number of workers during training", default=4)
    parser.add_argument("--backbone_type", type=str,
                        help="architecture of backbone", default="wide_resnet")
    parser.add_argument("--checkpoint_folder_path", type=str,
                        help="architecture of backbone", default="./")

    parser.add_argument("--logging_folder_path", type=str,
                        help="folder where logfile will be saved", default="./")
    parser.add_argument("--debug", type=str,
                        help="this input is 'DEBUG' when experiment is for debugging", default="False")
    parser.add_argument("--msg", type=str,
                        help="if none, put NONE. else, place message.", default="NONE")

    args = parser.parse_args()


    CKPT_BASE_NAME = args.backbone_type
    LOGFILE_NAME = CKPT_BASE_NAME + '_logfile'
    
    if args.debug == 'DEBUG':
        CKPT_BASE_NAME = 'debug_' + CKPT_BASE_NAME
        LOGFILE_NAME = 'debug_' + LOGFILE_NAME     
        
    if not os.path.exists(args.checkpoint_folder_path + CKPT_BASE_NAME):
        os.mkdir(args.checkpoint_folder_path + CKPT_BASE_NAME)
        os.mkdir(args.checkpoint_folder_path + CKPT_BASE_NAME + '/' + 'backups')
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger = setup_logger('logger', formatter, args.logging_folder_path + LOGFILE_NAME)
    

    if args.dataset == 'CIFAR_PLUS':
        with open(args.dataset_folder + '/train_obj.pkl', 'rb') as f:
            train_obj = pickle.load(f)
        with open(args.dataset_folder + '/test_obj.pkl', 'rb') as f:
            test_obj = pickle.load(f)
        with open(args.dataset_folder + '/meta.pkl', 'rb') as f:
            meta_dict = pickle.load(f)
        with open(args.dataset_folder + '/label_to_idx.pkl', 'rb') as f:
            label_to_idx = pickle.load(f)
        with open(args.dataset_folder + '/idx_to_label.pkl', 'rb') as f:
            idx_to_label = pickle.load(f)


        num_classes = 6

        logging.info("Number of seen classes: " + str(num_classes))

        dataset = CIFARDataset(train_obj, meta_dict, label_to_idx,
                              transforms.Compose([
                                  transforms.ToTensor(),
                                            transforms.RandomResizedCrop(args.img_size),
                                            transforms.RandomHorizontalFlip(),
                                            ]))
        val_dataset = CIFARDataset(train_obj, meta_dict, label_to_idx,
                              transforms.Compose([transforms.ToTensor(),transforms.Resize((args.img_size,args.img_size))]))



    if args.backbone_type == 'OSCRI_encoder':
        model = encoder32(latent_size=args.latent_size, num_classes=num_classes, num_rp_per_cls=args.num_rp_per_cls, gap=args.gap == 'TRUE')
        
    elif args.backbone_type == 'wide_resnet':
        model = wide_encoder(args.latent_size, 40, 4, 0, num_classes=num_classes, num_rp_per_cls=args.num_rp_per_cls)
        
    elif args.backbone_type == 'resnet_50':
        backbone = models.resnet50(pretrained=False)
        VISUAL_FEATURES_DIM = 2048
        model = encoder(backbone, VISUAL_FEATURES_DIM, latent_size=args.latent_size, num_classes=num_classes, num_rp_per_cls=args.num_rp_per_cls, gap=args.gap == 'TRUE')
    
    else:
        raise ValueError(args.backbone_type + ' is not supported.')
    load_model = True
    if load_model:
        model_path = args.checkpoint_folder_path + CKPT_BASE_NAME
        files = os.listdir(model_path)
        model_name = None
        for file in files:
            if file[-3:] == '.pt':
                if model_name is not None:
                    raise ValueError("Multiple possible models.")
                model_name = file[:-3]
        model.load_state_dict(torch.load(model_path + '/' + model_name + '.pt'))
    model.cuda()
    
    num_params = count_parameters(model)
    logger.info("Number of model parameters: " + str(num_params))

    criterion = nn.CrossEntropyLoss(reduction='none')   
    
       
    if args.lr_scheduler == 'step':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
    elif args.lr_scheduler == 'patience':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, verbose=True)
        
    else:
        raise ValueError(args.lr_scheduler + ' is not supported.')
        

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    train_n = len(dataset)
    best_used_running_loss = 100000000
    best_val_acc = 0.
    

    last_lr = False
    last_patience_counter = 0
    for epoch in range(0, args.n_epochs):

        logger.info("EPOCH " + str(epoch))
        running_loss = 0.0
        train_rpl_loss = 0.
        train_std_loss = 0.0
        train_correct = 0.0
        
        actual_lr = None
        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']
            if actual_lr is None:
                actual_lr = curr_lr
            else:
                if curr_lr != actual_lr:
                    raise ValueError("some param groups have different lr")
        logger.info("Learning rate: " + str(actual_lr))
        if actual_lr < 10 ** (-7):
            last_lr = True

        for i, data in enumerate(train_loader, 0):
            
            if args.debug == 'DEBUG':
                print('\nbatch ' + str(i))
            
            # get the inputs & combine positive and negatives together
            img = data['image']
            img = img.cuda()
            
            labels = data['label']
            labels = labels.cuda()
            
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model.forward(img)

            
            # Compute RPL loss
            loss, open_loss, closed_loss, logits = compute_rpl_loss(model, outputs, labels, criterion, args.lamb, args.gamma, args.divide == 'TRUE')
            train_rpl_loss += loss.item()
            
            loss.backward()
            
            optimizer.step()

            # update loss for this epoch
            running_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            max_probs, max_indices = torch.max(probs, 1)
                      
            train_correct += torch.sum(max_indices == labels).item()
        
            if args.debug == 'DEBUG':
                print("batch loss: " + str(loss.item()))
                print("rpl loss: " + str((closed_loss + open_loss).item()))
                print("number correct: " + str(torch.sum(max_indices == labels).item()))

        train_acc = train_correct/train_n
        logger.info("Training Accuracy is: " + str(train_acc))
        logger.info("Average overall training loss in epoch is: " + str(running_loss/train_n))

        model.eval()
        used_running_loss, used_val_acc = evaluate_val(model, criterion, val_loader, args.gamma, args.lamb, args.divide, logger)
        
        # Adjust learning rate
        if args.lr_scheduler == 'patience':
            scheduler.step(used_running_loss)
        elif args.lr_scheduler == 'step':
            scheduler.step()
        else:
            raise ValueError('scheduler did not update.')

        # case where only acc is top
        if used_val_acc > best_val_acc:
            
            curr_files = os.listdir(args.checkpoint_folder_path + CKPT_BASE_NAME + '/')
            models_to_move = []
            for file in curr_files:
                if file[-3:] == '.pt':
                    models_to_move.append(file)
            for mover in models_to_move:
                os.replace(args.checkpoint_folder_path + CKPT_BASE_NAME + '/' + mover, args.checkpoint_folder_path + CKPT_BASE_NAME + '/backups/' + mover)
            
            torch.save(model.state_dict(), args.checkpoint_folder_path + CKPT_BASE_NAME + '/' + str(epoch) + '.pt')
            
        elif args.lr_scheduler == 'patience' and last_lr:
            last_patience_counter += 1
            if last_patience_counter == 5:
                break
            
        if used_running_loss < best_used_running_loss:
            best_used_running_loss = used_running_loss
        if used_val_acc > best_val_acc:
            best_val_acc = used_val_acc
            if args.lr_scheduler == 'patience' and last_lr:
                last_patience_counter = 0
        model.train()
        
