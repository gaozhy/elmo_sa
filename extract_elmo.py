#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:57:49 2020

@author: zg
"""
import pandas as pd
from allennlp.commands.elmo import ElmoEmbedder
import scipy
from allennlp.predictors import Predictor 
import torch
from torch import tensor
from nltk.tokenize import sent_tokenize,word_tokenize
import numpy as np
import scipy.io as sio
from sklearn.metrics.pairwise import cosine_similarity
data=pd.read_csv('word_pairs192.csv')
words=[]

for i in range(192):
    tmp=str(data['Probe'][i]+' '+data['Target'][i])
    words.append(word_tokenize(tmp))
        
    

# data=pd.read_excel(r'Example_Data.xlsx')
# df=pd.DataFrame(data,columns=['Responses'])
# phrases=[]
# words=[]
    
    
# for i in df['Responses']:
#     tmp=sent_tokenize(i)
#     phrases.append(tmp)
#     tmp2=word_tokenize(i)    
#     words.append(tmp2)

# elmo = ElmoEmbedder(
#     options_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json', 
#     weight_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
# )
# version 3
elmo=ElmoEmbedder()

allembeddings=[]
allsentences_emb=[]
asso_strength=np.zeros((192,1))
layer1_w1=np.zeros((192,1024))
layer1_w2=np.zeros((192,1024))
layer2_w1=np.zeros((192,1024))
layer2_w2=np.zeros((192,1024))
layer3_w1=np.zeros((192,1024))
layer3_w2=np.zeros((192,1024))

layer1_w1m=np.zeros((192,192))
layer1_w2m=np.zeros((192,192))
layer2_w1m=np.zeros((192,192))
layer2_w2m=np.zeros((192,192))
layer3_w1m=np.zeros((192,192))
layer3_w2m=np.zeros((192,192))

for i in words:
    embeddings=elmo.embed_sentence(i)
    allembeddings.append(embeddings)
    
    
for i in range(192):
    layer1_w1[i,:]=allembeddings[i][0,0,:].reshape(1,-1)
    layer2_w1[i,:]=allembeddings[i][1,0,:].reshape(1,-1)
    layer3_w1[i,:]=allembeddings[i][2,0,:].reshape(1,-1)
    layer1_w2[i,:]=allembeddings[i][0,1,:].reshape(1,-1)
    layer2_w2[i,:]=allembeddings[i][1,1,:].reshape(1,-1)
    layer3_w2[i,:]=allembeddings[i][2,1,:].reshape(1,-1)
    asso_strength[i,:]=cosine_similarity(allembeddings[i][0,0,:].reshape(1,-1),allembeddings[i][0,1,:].reshape(1,-1))
    
    
    
# for i in range(192):
#     for j in range(192):
#         layer1_w1m[i,j]=cosine_similarity(layer1_w1[i,:].reshape(1,-1),layer1_w1[j,:].reshape(1,-1))
#         layer1_w2m[i,j]=cosine_similarity(layer1_w2[i,:].reshape(1,-1),layer1_w2[j,:].reshape(1,-1))
#         layer2_w1m[i,j]=cosine_similarity(layer2_w1[i,:].reshape(1,-1),layer2_w1[j,:].reshape(1,-1))
#         layer2_w2m[i,j]=cosine_similarity(layer2_w2[i,:].reshape(1,-1),layer2_w2[j,:].reshape(1,-1))
#         layer3_w1m[i,j]=cosine_similarity(layer3_w1[i,:].reshape(1,-1),layer3_w1[j,:].reshape(1,-1))
#         layer3_w2m[i,j]=cosine_similarity(layer3_w2[i,:].reshape(1,-1),layer3_w2[j,:].reshape(1,-1))
        
for i in range(192):
    for j in range(192):
        layer1_w1m[i,j]=cosine_similarity(allembeddings[i][0,0,:].reshape(1,-1),allembeddings[j][0,0,:].reshape(1,-1))
        layer1_w2m[i,j]=cosine_similarity(allembeddings[i][0,1,:].reshape(1,-1),allembeddings[j][0,1,:].reshape(1,-1))
        layer2_w1m[i,j]=cosine_similarity(allembeddings[i][1,0,:].reshape(1,-1),allembeddings[j][1,0,:].reshape(1,-1))
        layer2_w2m[i,j]=cosine_similarity(allembeddings[i][1,1,:].reshape(1,-1),allembeddings[j][1,1,:].reshape(1,-1))
        layer3_w1m[i,j]=cosine_similarity(allembeddings[i][2,0,:].reshape(1,-1),allembeddings[j][2,0,:].reshape(1,-1))
        layer3_w2m[i,j]=cosine_similarity(allembeddings[i][2,1,:].reshape(1,-1),allembeddings[j][2,1,:].reshape(1,-1))

    
sio.savemat('layers_feature_wp192_v4.mat',{'layer1_w1':layer1_w1,'layer1_w2':layer1_w2,
                                        'layer2_w1':layer2_w1,'layer2_w2':layer2_w2,
                                        'layer3_w1':layer3_w1,'layer3_w2':layer3_w2,
                                        'layer1_w1m':layer1_w1m,'layer1_w2m':layer1_w2m,
                                        'layer2_w1m':layer2_w1m,'layer2_w2m':layer2_w2m,
                                        'layer3_w1m':layer3_w1m,'layer3_w2m':layer3_w2m,
                                        'assoc_str':asso_strength})            
    
    
    