
# coding: utf-8

# In[6]:

import math
import numpy as np


# In[7]:

print(math.floor(2.6))
print(math.ceil(2.6))
print(math.floor(2.5))
print(math.ceil(2.5))
print(math.floor(2.3))
print(math.ceil(2.3))


# In[8]:

m = 3
np.random.permutation(m)


# In[9]:

X = np.array([[1,2,3],[4,5,6],[7,8,9]])


# In[10]:

X


# In[18]:

permutation = np.random.permutation(m)


# In[19]:

permutation


# In[20]:

X = X[:,list(permutation)]


# In[21]:

X


# In[22]:

2**5


# In[ ]:



