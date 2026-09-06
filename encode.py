import os
import torch
import argparse
from torch.utils.data import random_split
from models import GCN
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.utils.data import random_split
from utils import get_device, safe_torch_load


def ensure_split_files(data, path):
    split_paths = {
        'train': path + '_train.pt',
        'val': path + '_val.pt',
        'test': path + '_test.pt',
    }

    for split_name, split_path in split_paths.items():
        if os.path.exists(split_path):
            continue

        num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else len(data.raw_texts)
        indices = torch.arange(num_nodes)

        if hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask'):
            mask_map = {
                'train': data.train_mask,
                'val': data.val_mask,
                'test': data.test_mask,
            }
            split_indices = indices[mask_map[split_name]]
        else:
            shuffled = torch.randperm(num_nodes)
            train_size = int(0.7 * num_nodes)
            val_size = int(0.15 * num_nodes)
            if split_name == 'train':
                split_indices = shuffled[:train_size]
            elif split_name == 'val':
                split_indices = shuffled[train_size:train_size + val_size]
            else:
                split_indices = shuffled[train_size + val_size:]

        batches = []
        chunk_size = 10
        for start in range(0, len(split_indices), chunk_size):
            subset = split_indices[start:start + chunk_size]
            batch = Data(subset=subset.clone())
            if hasattr(data, 'raw_texts'):
                batch.raw_texts = [data.raw_texts[int(i)] for i in subset.tolist()]
            batches.append(batch)

        torch.save(batches, split_path)


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=8,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=2e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--path', type=str, default="Instagram/")
args = parser.parse_args()

device = get_device()
criterion = torch.nn.CrossEntropyLoss().to(device)

data = safe_torch_load(args.path + '.pt')
ensure_split_files(data, args.path)
train_loader = safe_torch_load(args.path + "_train.pt")
val_loader = safe_torch_load(args.path + "_val.pt")
test_loader = safe_torch_load(args.path + "_test.pt")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

train = []
for batch in train_loader:
    text = [data.raw_texts[i] for i in batch.subset]
    if hasattr(batch, "unique"):
        text = batch.unique
        for i in range(len(text)):
            text[i] = text[i][2:].strip()
    unique_embeddings = encoder.encode(text)
    unique_embeddings = torch.Tensor(unique_embeddings).to(device)
    batch.unique_embeddings = unique_embeddings
    train.append(batch)
torch.save(train, args.path + "_train.pt")

val = []
for batch in val_loader:
    text = [data.raw_texts[i] for i in batch.subset]
    if hasattr(batch, "unique"):
        text = batch.unique
        for i in range(len(text)):
            text[i] = text[i][3:].strip()
    unique_embeddings = encoder.encode(text)
    unique_embeddings = torch.Tensor(unique_embeddings).to(device)
    batch.unique_embeddings = unique_embeddings
    val.append(batch)
torch.save(val, args.path + "_val.pt")

test = []
for batch in test_loader:
    text = [data.raw_texts[i] for i in batch.subset]
    if hasattr(batch, "unique"):
        text = batch.unique
        for i in range(len(text)):
            text[i] = text[i][3:].strip()
    unique_embeddings = encoder.encode(text)
    unique_embeddings = torch.Tensor(unique_embeddings).to(device)
    batch.unique_embeddings = unique_embeddings
    test.append(batch)
torch.save(test, args.path + "_test.pt")
