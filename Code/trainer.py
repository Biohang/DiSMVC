import time
import pickle
import os.path as osp
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics


torch.manual_seed(0)

def pooling(x, y2x):
    x = torch.mm(y2x, x)
    row_sum = torch.sum(y2x, dim=1).clamp(min=1e-8).reshape(-1, 1)
    x = torch.div(x, row_sum)
    return x

    
class Trainer(object):
    def __init__(self, model, tau, log_every_n_steps, device):
        self.model = model.to(device)
        self.tau = tau
        self.log_every_n_steps = log_every_n_steps
        self.device = device
        self.writer = SummaryWriter()
        
    def load_data(self, g_data,m_data, labels, d2g,m2d, dis_path):
        self.g_data = g_data.to(self.device)
        self.m_data = m_data.to(self.device)
        self.labels = labels.to(self.device)
        self.d2g = d2g
        self.m2d = m2d
        self.dis_path = dis_path
     
    # Contrastive loss for gene	 
    def nce_loss_g(self, gz, kgz, labels):
        gz = F.normalize(gz, dim=1)
        kgz = F.normalize(kgz, dim=1)
        similarity_matrix = gz @ kgz.T
        similarity_matrix = torch.exp(similarity_matrix / self.tau)
        similarity_matrix_sum = torch.sum(similarity_matrix, 1, keepdim=True)
        positives_sum = torch.sum(similarity_matrix * labels, 1)
        loss = -torch.log(positives_sum * (similarity_matrix_sum**(-1)) + 1e-8).mean()
        return loss

    # Contrastive loss for miRNA
    def nce_loss_m(self, gz, kgz, labels):
        gz = F.normalize(gz, dim=1)
        kgz = F.normalize(kgz, dim=1)
        similarity_matrix = gz @ kgz.T
        similarity_matrix = torch.exp(similarity_matrix / self.tau)
        similarity_matrix_sum = torch.sum(similarity_matrix, -2, keepdim=True)
        positives_sum = torch.sum(similarity_matrix * labels, -2)
        loss = -torch.log(positives_sum * (similarity_matrix_sum**(-1)) + 1e-8).mean()
        return loss
    
	# Disease pair prediction loss
    def link_loss_dis(self, pre_value, d2d_link_labels):
        loss_function = torch.nn.BCEWithLogitsLoss()  
        loss = loss_function(pre_value.squeeze(), d2d_link_labels)
        return loss

    def train(self, epochs):
        t0 = time.time()
        print(f"Start JCLModel training for {epochs} epochs.")
        training_range = tqdm(range(epochs))
        # Load disease-disease training samples
        d2d_edge_index = np.loadtxt(self.dis_path + "/train_for_ukb_ori_sample.txt")
        d2d_link_labels = np.loadtxt(self.dis_path + "/train_for_ukb_ori_label.txt")
        d2d_edge_index = torch.tensor(d2d_edge_index).type(torch.long)
        d2d_link_labels = torch.tensor(d2d_link_labels)
        params_group1 = [{'params': self.model.g_encoder.parameters()},{'params': self.model.m_encoder.parameters()}]
        params_group2 =  [{'params': self.model.projection_dis_gene.parameters()},{'params': self.model.projection_dis_miRNA.parameters()},{'params': self.model.attention_dis.parameters()},{'params': self.model.dis_score.parameters()}]
        optimizer1 = optim.RMSprop(params_group1, 0.003)
        optimizer2 = optim.SGD(params_group2, lr=0.001)

        for epoch in training_range:
            g_h,m_h = self.model(self.g_data, self.m_data)
            loss_1_g = self.nce_loss_g(g_h, m_h, self.labels)
            loss_1_m = self.nce_loss_m(g_h, m_h, self.labels)
            loss_1 = 0.3 * loss_1_g + 0.7 * loss_1_m
            
			# Construct features for training disease pairs 
			d_h_1 = pooling(g_h, self.d2g.to_dense())
            d_h_2 = pooling(m_h, self.m2d.to_dense())
            d_train_1 = torch.cat((d_h_1[d2d_edge_index[:, 0],], d_h_1[d2d_edge_index[:, 1],]), 1)
            d_train_2 = torch.cat((d_h_2[d2d_edge_index[:, 0],], d_h_2[d2d_edge_index[:, 1],]), 1)
            hid_vec_1 = self.model.nonlinear_transformation_dis_gene(d_train_1)
            hid_vec_2 = self.model.nonlinear_transformation_dis_miRNA(d_train_2)
            hid_vec = torch.cat((hid_vec_1,hid_vec_2),1)
            atten_hid_vec = self.model.nonlinear_attention_dis(hid_vec)
            atten_score_vec = self.model.nonlinear_dis_score(atten_hid_vec)
            loss_2 = self.link_loss_dis(atten_score_vec,d2d_link_labels)
            loss = loss_1 + loss_2
            loss.requires_grad_(True)
            training_range.set_description('Loss %.4f' % loss.item())

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

        t1 = time.time()
        checkpoint_name = "checkpoint_{:03d}.pth.tar".format(epochs)
        torch.save({'epoch':epochs,
                    'model_state_dict':self.model.state_dict(),
                    'optimizer1':optimizer1.state_dict()},
                   f=osp.join(self.writer.log_dir, checkpoint_name))
        auc, ap, *_ = self.infer()

    def infer(self):
        with torch.no_grad():
            self.model.eval()
            g_h = self.model.get_gene_embeddings(self.g_data)
            m_h = self.model.get_miRNA_embeddings(self.m_data)
			
			# Load disease-disease testing samples
            d2d_edge_index_test = np.loadtxt(self.dis_path + "/test_sample_ukb_interpre_high_ori.txt")
            d2d_link_labels_test = np.loadtxt(self.dis_path + "/test_label_ukb_interpre_high_ori.txt")

            #Predict similarity scores for test disease pairs
            d2d_edge_index_test = torch.tensor(d2d_edge_index_test).type(torch.long)
            d2d_link_labels_test = torch.tensor(d2d_link_labels_test)
            d_1 = pooling(g_h, self.d2g.to_dense())
            d_2 = pooling(m_h, self.m2d.to_dense())
            d_test_1 = torch.cat((d_1[d2d_edge_index_test[:, 0],], d_1[d2d_edge_index_test[:, 1],]), 1)
            d_test_2 = torch.cat((d_2[d2d_edge_index_test[:, 0],], d_2[d2d_edge_index_test[:, 1],]), 1)
            hid_vec_test_1= self.model.nonlinear_transformation_dis_gene(d_test_1)
            hid_vec_test_2 = self.model.nonlinear_transformation_dis_miRNA(d_test_2)
            hid_vec_test = torch.cat((hid_vec_test_1,hid_vec_test_2),1)
            atten_hid_vec_test = self.model.nonlinear_attention_dis(hid_vec_test)
            np.savetxt(osp.join(self.writer.log_dir, "disease_pair_embedding.txt"),atten_hid_vec_test,fmt='%.6f')
            atten_score_vec_test = self.model.nonlinear_dis_score(atten_hid_vec_test)
            np.savetxt(osp.join(self.writer.log_dir, "dis_pair_for_node_with_interpretation_score.txt"),atten_score_vec_test,fmt='%.6f')
            ap = metrics.average_precision_score(d2d_link_labels_test, atten_score_vec_test)
            auroc = metrics.roc_auc_score(d2d_link_labels_test, atten_score_vec_test)
            print(f"AUROC: {auroc*100} | AUPRC: {ap*100}")
        return auroc, ap
    
