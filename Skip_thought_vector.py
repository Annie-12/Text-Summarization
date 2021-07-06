#!/usr/bin/env python
# coding: utf-8

# In[2]:


from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()


# In[18]:



print("process is starting")
f = open("inputfile.txt", encoding="utf8")
text = ""
for x in f:
    print(x)
    text += x

# tokenize the sentences

sentences = sent_tokenize(text.strip())
print('Sentences', len(sentences), sentences)


# In[4]:


import skipthoughts

# You would need to download pre-trained models first
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
encoded =  encoder.encode(sentences)


# In[8]:


import numpy as np
from sklearn.cluster import KMeans

n_clusters =int (np.ceil(len(encoded)**0.5))
kmeans = KMeans(n_clusters=n_clusters)
kmeans = kmeans.fit(encoded)


# In[16]:


from sklearn.metrics import pairwise_distances_argmin_min

avg = []
for j in range(n_clusters):
    idx = np.where(kmeans.labels_ == j)[0]
    avg.append(np.mean(idx))

closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, encoded)
ordering = sorted(range(n_clusters), key=lambda k: avg[k])
summary = ' '.join([sentences[closest[idx]] for idx in ordering])


# In[15]:


print(summary)


# In[ ]:




