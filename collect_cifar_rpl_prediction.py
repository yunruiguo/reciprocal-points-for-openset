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
                        help="Dimension of embeddings.", default=256)
    parser.add_argument("--num_rp_per_cls", type=int,
                        help="Number of reciprocal points per class.", default=1)
    parser.add_argument("--gamma", type=float,
                        help="", default=1)

    parser.add_argument("--backbone_type", type=str,
                        help="architecture of backbone", default="wide_resnet")

    parser.add_argument("--open_dataset_name", type=str,
                        help="name of folder where open dataset lives. if operating on closed dataset, put NONE.",
                        default="./cifar10_splits/split0/")
    parser.add_argument("--cifar100_path", type=str,
                        help="path to cifar100.", default="../data/cifar-100-python/")

    parser.add_argument("--model_folder_path", type=str,
                        help="full file path to folder where the baseline is located",
                        default="./wide_resnet/")

    args = parser.parse_args()
    with open(args.open_dataset_name + '/label_to_idx.pkl', 'rb') as f:
        label_to_idx = pickle.load(f)
    with open(args.open_dataset_name + '/idx_to_label.pkl', 'rb') as f:
        idx_to_label = pickle.load(f)
    with open(args.open_dataset_name + '/open_label_to_idx.pkl', 'rb') as f:
        open_label_to_idx = pickle.load(f)
    with open(args.open_dataset_name + '/open_idx_to_label.pkl', 'rb') as f:
        open_idx_to_label = pickle.load(f)
    with open(args.open_dataset_name + 'test_obj.pkl', 'rb') as fo:
        test_obj = pickle.load(fo)
    with open(args.open_dataset_name + 'open_test_obj.pkl', 'rb') as fo:
        open_test_obj = pickle.load(fo)
    with open(args.open_dataset_name + 'meta.pkl', 'rb') as fo:
        meta_dict = pickle.load(fo)

    files = os.listdir(args.model_folder_path)
    model_name = None
    for file in files:
        if file[-3:] == '.pt':
            if model_name is not None:
                raise ValueError("Multiple possible models.")
            model_name = file[:-3]

    seen_dataset = CIFARDataset(test_obj, meta_dict, label_to_idx,
                                transforms.Compose([
                                    transforms.ToTensor(),
                                ]))
    unseen_dataset = CIFARDataset(open_test_obj, meta_dict, open_label_to_idx,
                                  transforms.Compose([
                                      transforms.ToTensor(),
                                  ]))

    seen_loader = DataLoader(seen_dataset, batch_size=64, shuffle=False, num_workers=3)
    unseen_loader = DataLoader(unseen_dataset, batch_size=64, shuffle=False, num_workers=3)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if args.backbone_type == 'OSCRI_encoder':
        model = encoder32(latent_size=args.latent_size, num_classes=6, num_rp_per_cls=args.num_rp_per_cls,
                          gap=args.gap == 'TRUE')
    elif args.backbone_type == 'wide_resnet':
        model = wide_encoder(args.latent_size, 40, 4, 0, num_classes=6, num_rp_per_cls=args.num_rp_per_cls)
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