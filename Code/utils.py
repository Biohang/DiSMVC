import torch
import numpy as np
import scipy.sparse as sp


########################################################################
# Sparse Matrix Utils
########################################################################

def load_sparse(path):
    return sp.load_npz(path).tocoo()

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mx_to_torch_sparse_tesnsor(mx):
    sparse_mx = mx.astype(np.float32)
    sparse_mx.eliminate_zeros()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    size = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, size)

def generate_sparse_one_hot(num_ents, dtype=torch.float32):
    diag_size = num_ents
    diag_range = list(range(num_ents))
    diag_range = torch.tensor(diag_range)

    return torch.sparse_coo_tensor(
        indices=torch.vstack([diag_range, diag_range]),
        values=torch.ones(diag_size, dtype=dtype),
        size=(diag_size, diag_size))


########################################################################
# miRNA related Network Construction Utils
########################################################################
def construct_miRNA_disease_network(path):
    dName, dInd = load_dmap(path+"/dis2id.txt")
    mName, mInd = load_dmap(path+"/miRNA2id.txt")
    m2d_adj = np.zeros((len(mName),len(dName)))
    m2d_row = [] 
    m2d_col = [] 
    with open(path + 'miRNA2disease.txt') as f:
         f.readline()
         for line in f:
             miRNA, dis = line.strip('\t').split()
             m2d_row.append(mName[miRNA])
             m2d_col.append(dName[dis])
    m2d_adj[m2d_row,m2d_col] = 1
    data = np.ones(len(m2d_row))
    row = np.array(m2d_row)
    col = np.array(m2d_col)
    m2d = coo_matrix((data,(row,col)),shape=(len(mName),len(dName)))
    sp.save_npz(path + "miRNA2disease.npz",m2d,compressed=True)
    return m2d_adj

def construct_miRNA_similarity_network(path,m2d_adj):
    miRNA_similarity = rbf_kernel(m2d_adj)
    miRNA_similarity_2 = (miRNA_similarity - np.min(miRNA_similarity)) / (np.max(miRNA_similarity) - np.min(miRNA_similarity))
    data = np.array(miRNA_similarity_2.reshape(-1,1))
    data = data.reshape(data.shape[0],)
    index = np.where(data >= 0.8)
    data = data[index]
    row,col = np.where(miRNA_similarity_2 >= 0.8)
    row = np.array(row)
    col = np.array(col)
    m2m = coo_matrix((data,(row,col)),shape=(miRNA_similarity_2.shape[0],miRNA_similarity_2.shape[1]))
    sp.save_npz(path + "miRNA2miRNA.npz",m2m,compressed=True)

def construct_gene_miRNA_network(path):
    gName, gInd = load_dmap(path+"/gene2id.txt")
    mName, mInd = load_dmap(path+"/miRNA2id.txt")
    g2m_adj = np.zeros((len(gName),len(mName)))
    g2m_row = []     
    g2m_col = []   
    with open(path + 'gene2miRNA.txt') as f:
         f.readline()
         for line in f:
             gene, miRNA = line.strip('\t').split()
             g2m_row.append(gName[gene])
             g2m_col.append(mName[miRNA])
    g2m_adj[g2m_row,g2m_col] = 1
    data = np.ones(len(g2m_row))
    row = np.array(g2m_row)
    col = np.array(g2m_col)
    g2m = coo_matrix((data,(row,col)),shape=(len(gName),len(mName)))
    sp.save_npz(path + "gene2miRNA.npz",g2m,compressed=True)





