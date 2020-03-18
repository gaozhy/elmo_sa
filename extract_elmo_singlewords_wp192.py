#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:42:56 2020

@author: zg
"""

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
from scipy.spatial.distance import cosine 
data=pd.read_csv('word_pairs192.csv')

    

# data=pd.read_excel(r'Example_Data.xlsx')
# df=pd.DataFrame(data,columns=['Responses'])
# phrases=[]
# words1=[]
# words2=[]
    
    
# for i in df['Probe']:
#     tmp=sent_tokenize(i)
#     phrases.append(tmp)
#     tmp2=word_tokenize(i)    
#     words.append(tmp2)

# elmo = ElmoEmbedder(
#     options_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json', 
#     weight_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
# )
# # version 3
elmo=ElmoEmbedder()


allembeddings1=[]
allembeddings2=[]
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
for i in range(192):
    token_a=word_tokenize(str(data['Probe'][i]+' '+data['Target'][i]))
    embeddings_a=elmo.embed_sentence(token_a)
    asso_strength[i]=cosine_similarity(embeddings_a[0,0,:].reshape(1,-1),embeddings_a[0,1,:].reshape(1,-1))

for i in range(192):
    for j in range(192):       
        token1=word_tokenize(str(data['Probe'][i]+' '+data['Probe'][j]))
        token2=word_tokenize(str(data['Target'][i]+' '+data['Target'][j]))
        embeddings1=elmo.embed_sentence(token1)
        embeddings2=elmo.embed_sentence(token2)
        allembeddings1.append(embeddings1)
        allembeddings2.append(embeddings2)


        layer1_w1m[i,j]=cosine_similarity(embeddings1[0,0,:].reshape(1,-1),embeddings1[0,1,:].reshape(1,-1))
        layer1_w2m[i,j]=cosine_similarity(embeddings2[0,0,:].reshape(1,-1),embeddings2[0,1,:].reshape(1,-1))

         
sio.savemat('layers_feature_wp192_v4_singleword_try.mat',{'layer1_w1m':layer1_w1m,'layer1_w2m':layer1_w2m,
                                        'assoc_str':asso_strength})            
    
    
    