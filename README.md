# DiSMVC: a multi-view graph collaborative learning framework for measuring disease similarity
This repository contains the source code used in our paper titled DiSMVC: a multi-view graph collaborative learning framework for measuring disease similarity. The code is implemented to realize the proposed predictor, and the dataset and tutorials are also provided to assist users in utilizing the code.

# Introduction
In this study, we propose a new method named DiSMVC for measuring disease similarity. DiSMVC is a supervised graph collaborative framework including two major modules. The former one is cross-view graph contrastive learning module, aiming to enrich disease representation by considering their underlying molecular mechanism from both genetic and transcriptional views, while the latter module is  association pattern joint learning, which can capture deep association patterns by incorporating phenotypically interpretable multimorbidities in a supervised manner. Experimental results indicated that DiSMVC can identify molecularly interpretable similar diseases, and the synergies gained from DiSMVC contributed to its superior performance in measuring disease similarity. 
![image](https://github.com/Biohang/DiSMVC/blob/main/Image/Fig1.jpg)  
**Figure.1**. The framework of DiSMVC. 

# Datasets
##### The prepocessed data of various bio-entity networks are stored in .npz compressed file format, including:  
-----d2g.npz (disease-gene association network)   
-----gene2miRNA.npz (gene-miRNA interaction network)  
-----hnet.npz (gene interaction network)  
-----miRNA2disease.npz (disease-miRNA association network)  
-----miRNA2miRNA.npz (miRNA similarity network)  

##### The prepocessed data of bio-entity ID mapping are stored in .txt file format, including:    
-----dis2id.txt (mapping disease UMLS ID to disease index)   
-----gene2id.txt (mapping gene Entrez ID to gene index)  
-----miRNA2id.txt (mapping miRNA name to miRNA index) 

##### The samples and lables for training and testing phases, including:  
-----train_for_ukb_ori_sample.txt (training disease pairs)  
-----test_sample_ukb_interpre_high.txt (testing disease pairs)  
-----train_for_ukb_ori_label.txt (labels for training samples)  
-----test_label_ukb_interpre_high.txt (labels for testing samples)  

# Usage
##### Basic environment setup:  
-----python 3.8  
-----cuda 11.3  
-----pytorch 1.12.0  

##### Training and Testing  
Training codes includes the scripts in ./Code/main.py and ./Code/trainer.py. model.py records the neural network models in DiSMVC and utils.py records the basic functional functions. DiSMVC will be tested after training.  

##### Running the main script 
'python -u main.py \   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--data={} \ ;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# path to dataset  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--h_dim={} \ ;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# dimension of hidden layer  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--tau={} \ ;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# softmax temperature  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--lr={} \ ;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# learning rate  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--epochs={} \ ;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# train epochs  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--disable-cuda={} \;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# disable CUDA  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'.format(data, h_dim, z_dim, tau, lr, epochs, disable_cuda)  





