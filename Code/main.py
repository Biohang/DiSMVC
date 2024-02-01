import argparse
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import Data
from utils import *
from model import GCN, GAT,Projection_dis_gene,Projection_dis_miRNA,Attention_dis,Dis_score,DiSMVC
from trainer import Trainer

# Set parameters
parser = argparse.ArgumentParser(description="PyTorch JCLModel")
parser.add_argument("--data", default="../Dataset", help="path to dataset")
parser.add_argument("--h_dim", default=128, type=int, help="dimension of hidden layer")
parser.add_argument("--tau", default=1.0, type=float, help="softmax temperature")
parser.add_argument("--lr", default=0.0005, type=float, help="learning rate")
parser.add_argument("--epochs", default=200, type=int, help="train epochs")
parser.add_argument("--disable-cuda", default=True, action="store_true", help="disable CUDA")
args = parser.parse_args()
device = torch.device("cuda" if not args.disable_cuda and torch.cuda.is_available() else "cpu")

# Load human gene net for GCN model
hnadj = load_sparse(args.data+"/hnet.npz")
src = hnadj.row
dst = hnadj.col
hn_edge_weight = torch.tensor(np.hstack((hnadj.data, hnadj.data)), dtype=torch.float)
hn_edge_weight = (hn_edge_weight - hn_edge_weight.min()) / (hn_edge_weight.max() - hn_edge_weight.min())
hn_edge_index = torch.tensor(np.vstack((np.concatenate([src, dst]), np.concatenate([dst, src]))), dtype=torch.long)

# Load disease2gene network
d2g = load_sparse(args.data+"/d2g.npz")
d2g = mx_to_torch_sparse_tesnsor(d2g)

x = generate_sparse_one_hot(d2g.shape[1])
g_data = Data(x=x, edge_index=hn_edge_index, edge_weight=hn_edge_weight)

# Load gene2miRNA network
g2m = load_sparse(args.data+"/gene2miRNA.npz")
g2m = mx_to_torch_sparse_tesnsor(g2m).to_dense()

# Load miRNA-miRNA similarity net for GAT model
mnadj = load_sparse(args.data+"/miRNA2miRNA.npz")
src = mnadj.row
dst = mnadj.col
mn_edge_weight = torch.tensor(np.hstack((mnadj.data, mnadj.data)), dtype=torch.float)
mn_edge_weight = (mn_edge_weight - mn_edge_weight.min()) / (mn_edge_weight.max() - mn_edge_weight.min())
mn_edge_index = torch.tensor(np.vstack((np.concatenate([src, dst]), np.concatenate([dst, src]))), dtype=torch.long)
x_m = generate_sparse_one_hot(mnadj.shape[0])
m_data = Data(x=x_m, edge_index=mn_edge_index, edge_weight=mn_edge_weight)

# Load miRNA2disease network
m2d = load_sparse(args.data+"/miRNA2disease.npz")
m2d = m2d.T
m2d = mx_to_torch_sparse_tesnsor(m2d)

#DiSMVC initialization
g_encoder = GCN(nfeat=g_data.x.shape[1], nhid=args.h_dim)
m_encoder = GAT(nfeat=m_data.x.shape[1], nhid=args.h_dim)
projection_dis_gene = Projection_dis_gene(args.h_dim, args.h_dim)
projection_dis_miRNA = Projection_dis_miRNA(args.h_dim, args.h_dim)
attention_dis = Attention_dis(args.h_dim)
dis_score = Dis_score(args.h_dim)
model = DiSMVC(g_encoder, m_encoder, projection_dis_gene,projection_dis_miRNA,attention_dis,dis_score)
trainer = Trainer(model, tau=args.tau, log_every_n_steps=args.log_every_n_steps, device=device)
trainer.load_data(g_data, m_data, g2m, d2g,m2d, args.data)
print("Finish initialization")

#Train DiSMVC 
trainer.train(args.epochs)
