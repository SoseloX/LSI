import torch
import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
import datetime
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
import random


class lang_probe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(lang_probe, self).__init__()
        self.mlp = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        out = self.mlp(x)
        return out

class ProbeDataset(Dataset):
    def __init__(self, activations, labels):
        self.activations = activations
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        activation = self.activations[idx]
        label = self.labels[idx]
        return torch.tensor(activation), torch.tensor(label)





def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='Llama-2-7b-chat-hf', help='model name')
    parser.add_argument('--dataset_name', type=str, default='wikilingual_100', help='feature bank for training probes')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=927, help='seed')
    parser.add_argument('--iter', type=int, default=20, help='seed')
    parser.add_argument('--last_id', type=int, default=5, help='seed')
    args = parser.parse_args()

    print("======= Argument Values =======")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("===============================")
    

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    DEVICE = 'cuda:2'

    models_list = []

    # load activations 
    layer_wise_activations = np.load(f"")
    labels_all = np.load(f"")

    for _ in range(args.iter):
        labels_train, labels_val, activations_train, activations_val = train_test_split(
            labels_all, layer_wise_activations, test_size=0.2, random_state=random.randint(1, 100)
        )

        model = lang_probe(131072, 7).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

        train_dataset = ProbeDataset(activations_train, labels_train)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_dataset = ProbeDataset(activations_val, labels_val)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        best_acc = 0
        for epoch in range(10):
            model.train()
            running_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False)
            for i, (features, labels) in enumerate(progress_bar):
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                features = features.to(torch.float32)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()     

                running_loss += loss.item()
                if i % 10 == 9:  # 每10个batch打印一次平均loss
                    progress_bar.set_postfix({'loss': running_loss / 10})
                    running_loss = 0.0 

            progress_bar.close()

            all_labels = []
            all_preds = []
            model.eval()
            with torch.no_grad():
                for features, labels in val_dataloader:
                    features, labels = features.to(DEVICE), labels.to(DEVICE)
                    features = features.to(torch.float32)
                    outputs = model(features)
                    normalized_outputs = F.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    all_labels.extend(labels.tolist())
                    all_preds.extend(preds.tolist())               


            accuracy = sum([pred == label for pred, label in zip(all_preds, all_labels)]) / len(all_labels)
            print(f"Acc: {accuracy:.2f}")
            if accuracy > best_acc:
                best_acc = accuracy

        model = model.cpu()
        models_list.append(model.state_dict())

    with open(f"", "wb") as f:
        pickle.dump(models_list, f)


        

                    


if __name__ == "__main__":
    if not os.path.exists('log'):
        os.makedirs('log')

    # 获取当前时间并格式化为字符串（如：2024-06-28_231141）
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')

    # 构造日志文件名，包含日期和时间
    log_filename = f''

    # 打开日志文件
    log_file = open(log_filename, 'w')

    # 重定向标准输出和标准错误
    sys.stdout = log_file
    sys.stderr = log_file

    main()
