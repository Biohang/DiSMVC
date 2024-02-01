# DiSMVC: a multi-view graph collaborative learning framework for measuring disease similarity
This repository contains the source code used in our paper titled DiSMVC: a multi-view graph collaborative learning framework for measuring disease similarity. The code is implemented to realize the proposed predictor, and the dataset and tutorials are also provided to assist users in utilizing the code.

# Introduction
In this study, we propose a novel computational method called DiSMVC to measure disease similarity. DiSMVC is a supervised graph collaborative framework including two major modules cross-view graph contrastive learning and association pattern joint learning. The former aims to enrich disease representation by considering their underlying molecular mechanism from both genetic and transcriptional views, while the latter can capture deep association patterns by incorporating phenotypically interpretable multimorbidities in a supervised manner. Experimental results indicated that DiSMVC can identify molecularly interpretable similar diseases, and the synergies gained from DiSMVC contributed to its superior performance in measuring disease similarity. 
![image](https://github.com/Biohang/DiSMVC/blob/main/Image/Fig1.jpg)  
**Figure.1**. The framework of DiSMVC. There are two main steps: (i) Cross-view graph contrastive learning. Gene interaction network and miRNA similarity network are constructed, based on that node features are extracted by considering their proximity structures via different graph representation algorithms. Graph contrastive learning is implemented to further refine the hidden features of genes and miRNAs. (ii) Association pattern joint learning. Average pooling and concatenate strategies are applied to obtain initial disease pair features based on various prior bio-entity networks. Multi-layer perceptron models are jointly learned to detect hidden association patterns and predict disease similarity scores. 

# Datasets
The prepocessed data of various bio-entity networks are stored in .npz compressed file format, including:  
-----d2g.npz (disease-gene association network)   
-----gene2miRNA.npz (gene-miRNA interaction network)  
-----hnet.npz (gene interaction network)  
-----miRNA2disease.npz (disease-miRNA association network)  
-----miRNA2miRNA.npz (miRNA similarity network)  

The prepocessed data of bio-entity ID mapping are stored in .txt file format, including:    
-----dis2id.txt (mapping disease UMLS ID to disease index)   
-----gene2id.txt (mapping gene Entrez ID to gene index)  
-----miRNA2id.txt (mapping miRNA name to miRNA index) 

The samples and lables for training and testing phases, including:  
-----train_for_ukb_ori_sample.txt (training disease pairs)  
-----test_sample_ukb_interpre_high.txt (testing disease pairs)  
-----train_for_ukb_ori_label.txt (labels for training samples)  
-----test_label_ukb_interpre_high.txt (labels for testing samples)  

# Usage
Basic environment setup:  
-----python 3.8  
-----cuda 11.3
-----pytorch 1.12.0  

Training and Testing  
Training codes includes the scripts in ./Code/main.py and ./Code/trainer.py. model.py records the neural network models in DiSMVC and utils.py records the basic functional functions. DiSMVC will be tested after training.  

'python -u main.py \   
        --data={} \                     # path to dataset  
        --h_dim={} \                    # dimension of layer h  
        --z_dim={} \                    # dimension of layer z  
        --tau={} \                      # softmax temperature  
        --lr={} \                       # learning rate  
        --epochs={} \                   # train epochs  
        --disable-cuda={} \             # disable CUDA  
        '.format(data, h_dim, z_dim, tau, lr, epochs, disable_cuda)  





