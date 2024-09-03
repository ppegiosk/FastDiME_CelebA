
import argparse
import torch
from torch.utils import data

from torchvision import transforms
from torchvision import datasets

from core.image_datasets import CelebADataset, CelebAMiniVal, ShortcutCelebADataset, ShortcutCFDataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from sklearn.metrics import roc_auc_score
import numpy as np
import random

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Shortcut Learing Detection Pipeline')
    parser.add_argument('--percentage', type=float, default=1.0, help='level of encoded correlation between the shortcut feature and the task label')
    parser.add_argument('--n_samples', type=int, default=5000,  help='number of samples for each task label class')
    parser.add_argument('--data', default='/scratch/ppar/data/img_align_celeba', type=str, help='data path')
    parser.add_argument('--path', default='/scratch/ppar/results', type=str, help='results path')
    parser.add_argument('--exp-name', default='fastdime_shortcut', type=str, help='experiment name')
    return parser.parse_args()


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print()
    print('Shortcut Learing Detection Pipeline')
    print(args)
    print()

    print("Dataset Preparation")
    # Stage 1: Curate training set D_k  with k% level of encoded correlation between the shortcut feature and the task label
    trainset_Dk = ShortcutCelebADataset(
        image_size=128,
        data_dir=args.data,
        partition='train',
        random_crop=True,
        random_flip=True,
        normalize=True,
        query_label=31,
        task_label=39,
        shortcut_label_name='Smiling',
        task_label_name='Young',
        percentage=args.percentage,
        n_samples=args.n_samples)


    # Stage 2 (a): Curate an i.i.d. test set test_k with the same correlation rate as the training set
    testset_k = ShortcutCelebADataset(
        image_size=128,
        data_dir=args.data,
        partition='test',
        random_crop=False,
        random_flip=False,
        normalize=True,
        query_label=31,
        task_label=39,
        shortcut_label_name='Smiling',
        task_label_name='Young',
        percentage=args.percentage,
        n_samples=int(args.n_samples//10))


    # Stage 2 (b,c): Curate balanced test set test_u where each class 
    # is represented equally with 50% positive/negative samples
    # and counterfactual balanced test set called testc_u, which is obtained 
    # by generating FastDiME counterfactuals for each image in the test_u
    balanced_testset = ShortcutCFDataset(path=args.path, exp_name=args.exp_name)
    balanced_testset = Subset(balanced_testset, list(range(1000)))


    print(f"Training set D_{args.percentage * 100} % size: {len(trainset_Dk)}")
    print(f"Test set test_{args.percentage * 100} % size: {len(testset_k)}")
    print(f"Balanced test_u and counterfactual testc_u test sets size: {len(balanced_testset)}\n")


    train_Dk_loader = DataLoader(trainset_Dk, batch_size=100, shuffle=True)
    testk_loader = DataLoader(testset_k, batch_size=100, shuffle=False)
    testu_loader = DataLoader(balanced_testset, batch_size=100, shuffle=False)

    set_seed(seed=42)

    print("Model Training")
    # load the pre-trained ResNet-18 model
    model = models.resnet18(pretrained=True)
    # modify the final fully connected layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 1)
    # define a binary cross-entropy loss and an optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.05)
    model = model.to(device)
    num_epochs = 20
    
    # train model on train set D_k
    for epoch in range(num_epochs):
        model.train()
        for train_inputs, _ , train_task_labels in train_Dk_loader:
            train_inputs, train_task_labels = train_inputs.to(device), train_task_labels.to(device)
            optimizer.zero_grad()
            train_outputs = model(train_inputs).squeeze()
            loss = criterion(train_outputs, train_task_labels.float())
            loss.backward()
            optimizer.step()

        # test model on i.i.d test set test_k
        model.eval()
        correct, total = 0, 0
        predictions_list, task_labels_list = [], []
        with torch.no_grad():
            for inputs, _ , task_labels in testk_loader:
                inputs, task_labels = inputs.to(device), task_labels.to(device)
                outputs = model(inputs).squeeze()
                predictions = torch.sigmoid(outputs)
                predictions_list.extend(predictions.cpu().numpy())
                task_labels_list.extend(task_labels.cpu().numpy())
                total += task_labels.size(0)
                correct += (torch.round(predictions) == task_labels.float()).sum().item()

        tesk_k_accuracy = correct / total
        test_k_aucroc = roc_auc_score(np.array(task_labels_list), np.array(predictions_list))
        print(f'Epoch {epoch+1}/{num_epochs}, train loss: {loss:.4f}, test_k ACC: {tesk_k_accuracy:.2%}, test_k AUROC: {test_k_aucroc:.4f}')

    print("Model training and evalaution complete")
    print(f"AUROC test_k =  {test_k_aucroc:.4f}\n")
    # torch.save(model.state_dict(), f'model_f{args.percentage}.pth')

    print("Testing for Shortcut Learing")
    model.eval()
    predictions_list_original, predictions_list_conterfactual = [], []
    task_labels_list_original, task_labels_list_counterfactual = [], []
    shortcut_labels_list_original, shortcut_labels_list_counterfactual = [], []

    with torch.no_grad():
        for original, counterfactual, task_labels, shortcut_labels in testu_loader:
            original, counterfactual = original.to(device), counterfactual.to(device)
            task_labels, shortcut_labels =  task_labels.to(device), shortcut_labels.to(device)

            # get predictions for original images
            outputs_original = model(original).squeeze()
            predictions_original = torch.sigmoid(outputs_original)

            # get predictions for FastDiME counterfactuals
            outputs_counterfactuals = model(counterfactual).squeeze()
            predictions_counterfactual = torch.sigmoid(outputs_counterfactuals)

            # store predictions and labels for original images
            predictions_list_original.extend(predictions_original.cpu().numpy())
            task_labels_list_original.extend(task_labels.cpu().numpy())
            shortcut_labels_list_original.extend(shortcut_labels.cpu().numpy())

            # store predictions and labels for FastDiME counterfactuals
            predictions_list_conterfactual.extend(predictions_counterfactual.cpu().numpy())
            task_labels_list_counterfactual.extend(task_labels.cpu().numpy())
            shortcut_labels_list_counterfactual.extend(shortcut_labels.cpu().numpy())

    accuracy_original = np.mean(np.round(predictions_list_original) == task_labels_list_original)
    accuracy_counterfactual = np.mean(np.round(predictions_list_conterfactual) == task_labels_list_counterfactual)

    # calculate AUCROC separately for original set (test_u) and counterfactual set (testc_u)
    aucroc_original = roc_auc_score(task_labels_list_original, predictions_list_original)
    aucroc_counterfactual = roc_auc_score(task_labels_list_counterfactual, predictions_list_conterfactual)

    predictions_original = np.array(predictions_list_original)
    predictions_counterfactual = np.array(predictions_list_conterfactual)

    # Stage 3: Measure the difference in confidence level between original and shortcut-flipped counterfactuals
    # MAD between original images (test_u) and shortcut counterfactuals (testc_u)
    mad = np.mean(np.abs(predictions_original-predictions_counterfactual))
    # Mean confidence Difference (MD) across two subtests according to their true shortcut label
    shortcut_cases_index = np.argwhere(np.array(shortcut_labels_list_counterfactual)==1)
    non_shortcut_cases_index = np.argwhere(np.array(shortcut_labels_list_counterfactual)==0)
    md_shortcut = np.mean(predictions_original[shortcut_cases_index]-predictions_counterfactual[shortcut_cases_index])
    md_non_shortcut = np.mean(predictions_original[non_shortcut_cases_index]-predictions_counterfactual[non_shortcut_cases_index])

    print(f"Results for dataset with k = {args.percentage * 100}% level of encoded correlation between the shortcut feature and the task label")
    print(f"test_k AUROC = {test_k_aucroc:.4f}")
    print(f"test_u AUCROC = {aucroc_original:.4f}")
    print(f"testc_u AUCROC = {aucroc_counterfactual:.4f}")
    print(f"MAD = {mad:.4f}")
    print(f"MD_s1 = {md_shortcut:.4f}")
    print(f"MD_s0 = {md_non_shortcut:.4f}")

    breakpoint()

if __name__ == "__main__":
    main()