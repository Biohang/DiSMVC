import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch_geometric.nn import GCNConv, GATConv
import numpy as np



class Projection_dis_gene(nn.Module):

    def __init__(self, input_dim, hid_dim):
        super(Projection_dis_gene, self).__init__()
        self.fc1 = Linear(2*input_dim, 2*hid_dim)
        self.fc2 = Linear(2*hid_dim, hid_dim)
        self.act_fn = nn.ReLU()
        self.layernorm_1 = nn.LayerNorm(2*hid_dim, eps=1e-6)
        self.layernorm_2 = nn.LayerNorm(hid_dim, eps=1e-6)
        self.act_fn_score = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.layernorm_1(x)
        x = self.fc2(x)
        x = self.act_fn(x)
        x = self.layernorm_2(x)
        return x

class Projection_dis_miRNA(nn.Module):

        def __init__(self, input_dim, hid_dim):
            super(Projection_dis_miRNA, self).__init__()
            self.fc1 = Linear(2*input_dim, 2*hid_dim)
            self.fc2 = Linear(2*hid_dim, hid_dim)
            self.act_fn = nn.ReLU()
            self.layernorm_1 = nn.LayerNorm(2*hid_dim, eps=1e-6)
            self.layernorm_2 = nn.LayerNorm(hid_dim, eps=1e-6)
            self.act_fn_score = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.act_fn(x)
            x = self.layernorm_1(x)
            x = self.fc2(x)
            x = self.act_fn(x)
            x = self.layernorm_2(x)
            return x

class Attention_dis(nn.Module):

    def __init__(self, hid_dim):
        super(Attention_dis, self).__init__()
        self.fc1 = Linear( 2*hid_dim, 2*hid_dim)
        self.layernorm_1 = nn.LayerNorm( 2*hid_dim, eps=1e-6)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        attention_weights = torch.softmax(self.fc1(x), dim=-2)  
        x = torch.mul(x,attention_weights)
        x = self.act_fn(x)
        x = self.layernorm_1(x)
        return x

class Dis_score(nn.Module):
    def __init__(self,  hid_dim):
        super(Dis_score, self).__init__()
        self.fc2 = Linear( 2*hid_dim,1)

    def forward(self, x):
        x = self.fc2(x)
        return x

class GCN(nn.Module):
    
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid, bias=True)
        self.conv2 = GCNConv(nhid, nhid, bias=True)

    def forward(self, data):
        try:
            x = self.conv1(data.x, data.edge_index, data.edge_weight)
            x = F.leaky_relu(x)
            x = self.conv2(x, data.edge_index, data.edge_weight)
        except:
            x = self.conv1(data.x, data.edge_index)
            x = F.leaky_relu(x)
            x = self.conv2(x, data.edge_index)
        return x
    
class GAT(nn.Module):

    def __init__(self, nfeat, nhid):
        super(GAT, self).__init__()
        self.conv1 = GATConv(nfeat, nhid, bias=True)
        self.conv2 = GATConv(nhid, nhid, bias=True)

    def forward(self, data):
        try:
            x = self.conv1(data.x, data.edge_index, data.edge_weight)
            x = F.leaky_relu(x)
            x = self.conv2(x, data.edge_index, data.edge_weight)
        except:
            x = self.conv1(data.x, data.edge_index)
            x = F.leaky_relu(x)
            x = self.conv2(x, data.edge_index)
        return x


class DiSMVC(nn.Module):
    
    def __init__(self, 
                 g_encoder, 
                 m_encoder,
                 projection_dis_gene,
                 projection_dis_miRNA,
                 attention_dis,
                 dis_score):
        super(DiSMVC, self).__init__()
        self.g_encoder = g_encoder
        self.m_encoder = m_encoder
        self.projection_dis_gene = projection_dis_gene
        self.projection_dis_miRNA = projection_dis_miRNA
        self.attention_dis = attention_dis
        self.dis_score = dis_score

    def forward(self, g_data, m_data):
        g_h = self.g_encoder(g_data)
        m_h = self.m_encoder(m_data)
        return g_h,m_h
    
    def nonlinear_transformation_dis_gene(self, h):
        z = self.projection_dis_gene(h)
        return z


    def nonlinear_transformation_dis_miRNA(self, h):
        z = self.projection_dis_miRNA(h)
        return z

    def nonlinear_attention_dis(self, h):
        z = self.attention_dis(h)
        return z

    def nonlinear_dis_score(self, h):
        z = self.dis_score(h)
        return z

    def get_gene_embeddings(self, g_data):
        return self.g_encoder(g_data)

    def get_miRNA_embeddings(self, m_data):
        return self.m_encoder(m_data)
