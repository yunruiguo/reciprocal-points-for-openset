import argparse
import os
import shutil

import torch
import pickle

import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.cifar_dataset import CIFARDataset
from datasets.dataset import StandardDataset
from datasets.open_dataset import OpenDataset
from models.backbone import encoder32
from models.backbone_wide_resnet import wide_encoder
from evaluate import collect_rpl_max, seenval_baseline_thresh, unseenval_baseline_thresh

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_type", type=str,
                        help="e.g. VAL or TEST", default="TEST")

    parser.add_argument("--gap", type=str,
                        help="TRUE iff use global average pooling layer. Otherwise, use linear layer.", default="TRUE")
    parser.add_argument("--desired_features", type=str,
                        help="None means no features desired. Other examples include last, 2_to_last.", default="None")
    parser.add_argument("--latent_size", type=int,
                        help="Dimension of embeddings.", default=128)
    parser.add_argument("--num_rp_per_cls", type=int,
                        help="Number of reciprocal points per class.", default=1)
    parser.add_argument("--gamma", type=float,
                        help="", default=1)

    parser.add_argument("--backbone_type", type=str,
                        help="architecture of backbone", default="wide_resnet")

    parser.add_argument("--closed_dataset_folder_path", type=str,
                        help="path to closed dataset folder.", default="./cifar10_plus/")
    parser.add_argument("--open_dataset_name", type=str,
                        help="name of folder where open dataset lives. if operating on closed dataset, put NONE.",
                        default="open_animals_10_split0")
    parser.add_argument("--cifar100_path", type=str,
                        help="path to cifar100.", default="../data/cifar-100-python/")

    parser.add_argument("--model_folder_path", type=str,
                        help="full file path to folder where the baseline is located",
                        default="./wide_resnet/")

    args = parser.parse_args()

    if args.dataset_type == 'TEST':
        with open(args.closed_dataset_folder_path + 'closed_test_img_list.pkl', 'rb') as f:
            seen_data = pickle.load(f)
        with open(args.closed_dataset_folder_path + args.open_dataset_name + '/open_test_img_list.pkl', 'rb') as f:
            unseen_data = pickle.load(f)
    else:
        raise Exception('Unsupported dataset type')

    with open(args.closed_dataset_folder_path + 'closed_train_mean.pkl', 'rb') as f:
        train_mean = pickle.load(f)
    with open(args.closed_dataset_folder_path + 'closed_train_std.pkl', 'rb') as f:
        train_std = pickle.load(f)
    img_normalize = transforms.Normalize(mean=train_mean,
                                         std=train_std)

    with open(args.closed_dataset_folder_path + args.open_dataset_name + '/open_label_to_idx.pkl', 'rb') as f:
        open_label_to_idx = pickle.load(f)
    with open(args.closed_dataset_folder_path + args.open_dataset_name + '/open_idx_to_label.pkl', 'rb') as f:
        open_idx_to_label = pickle.load(f)
    with open(args.cifar100_path + 'test', 'rb') as fo:
        open_test_obj = pickle.load(fo, encoding='bytes')
    with open(args.cifar100_path + 'meta', 'rb') as fo:
        open_meta_dict = pickle.load(fo, encoding='bytes')

    with open(args.closed_dataset_folder_path + 'label_to_idx.pkl', 'rb') as f:
        label_to_idx = pickle.load(f)
    with open(args.closed_dataset_folder_path + 'idx_to_label.pkl', 'rb') as f:
        idx_to_label = pickle.load(f)
    with open(args.closed_dataset_folder_path + 'train_obj.pkl', 'rb') as f:
        train_obj = pickle.load(f)
    with open(args.closed_dataset_folder_path + 'test_obj.pkl', 'rb') as f:
        test_obj = pickle.load(f)
    with open(args.closed_dataset_folder_path + 'meta.pkl', 'rb') as f:
        meta_dict = pickle.load(f)

    files = os.listdir(args.model_folder_path)
    model_name = None
    for file in files:
        if file[-3:] == '.pt':
            if model_name is not None:
                raise ValueError("Multiple possible models.")
            model_name = file[:-3]

    seen_dataset = CIFARDataset(seen_data, test_obj, meta_dict, label_to_idx,
                                transforms.Compose([
                                    transforms.ToTensor(),
                                    #img_normalize,
                                ]), openset=False)
    unseen_dataset = CIFARDataset(unseen_data, open_test_obj, open_meta_dict, open_label_to_idx,
                                  transforms.Compose([
                                      transforms.ToTensor(),
                                      #img_normalize,
                                  ]), openset=True)

    seen_loader = DataLoader(seen_dataset, batch_size=64, shuffle=False, num_workers=3)
    unseen_loader = DataLoader(unseen_dataset, batch_size=64, shuffle=False, num_workers=3)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if args.backbone_type == 'OSCRI_encoder':
        model = encoder32(latent_size=args.latent_size, num_classes=4, num_rp_per_cls=args.num_rp_per_cls,
                          gap=args.gap == 'TRUE')
    elif args.backbone_type == 'wide_resnet':
        model = wide_encoder(256, 40, 4, 0, num_classes=4, num_rp_per_cls=args.num_rp_per_cls)
    else:
        raise ValueError(args.backbone_type + ' is not supported.')

    model.load_state_dict(torch.load(args.model_folder_path + model_name + '.pt'))
    model.cuda()
    model.eval()

    seen_confidence_dict = collect_rpl_max(model, seen_loader, args.gamma, cifar=True,
                                           idx_to_label=idx_to_label)
    unseen_confidence_dict = collect_rpl_max(model, unseen_loader, args.gamma, cifar=True,
                                             idx_to_label=open_idx_to_label)

    # Computing AUC
    from sklearn.metrics import roc_auc_score

    preds = []
    labels = []
    dist = []
    for known_class_str, samples in seen_confidence_dict.items():
        for s in samples:
            labels += [1]
            preds += [s['prob']]
            dist += [s['dist']]

    for known_class_str, samples in unseen_confidence_dict.items():
        for s in samples:
            labels += [0]
            preds += [s['prob']]
            dist += [s['dist']]

    auc_prob = roc_auc_score(labels, preds)
    auc_dist = roc_auc_score(labels, dist)
    print('AUC on prob', auc_prob)
    print('AUC on dist', auc_dist)

"""
    print("Dist-Auroc score: " + str(dist_auroc_score))
    print("Prob-Auroc score: " + str(prob_auroc_score))

    metrics = summarize(seen_info, unseen_info, thresh, verbose=False)
    metrics['dist_auroc_lwnealstyle'] = dist_auroc_score
    metrics['prob_auroc_lwnealstyle'] = prob_auroc_score
    metrics['dist_OSR_CSR_AUC'] = dist_metrics['OSR_CSR_AUC']

    print("prob-AUC score: " + str(metrics['OSR_CSR_AUC']))
    print("dist-AUC score: " + str(metrics['dist_OSR_CSR_AUC']))

    with open(metrics_folder + 'metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
"""