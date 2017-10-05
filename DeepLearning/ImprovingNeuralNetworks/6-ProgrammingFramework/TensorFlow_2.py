
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf


# In[2]:

y_hat = tf.constant(10,name="y_hat")
y = tf.constant(5,name="y")
loss = tf.Variable((y-y_hat)**2,name="loss")
init = tf.global_variables_initializer()


# In[3]:

with tf.Session() as session:
    session.run(init)
    print(session.run(loss))


# In[4]:

a = tf.constant(5)
b = tf.constant(6)
c = tf.multiply(a,b)
print(c)


# In[6]:

with tf.Session() as sess:
    print(sess.run(c))


# In[13]:

x = tf.placeholder(tf.float32,[3,1])
y = 2 * x
with tf.Session() as sess:
    print(sess.run(y,feed_dict={x:np.array([[3.],[4.],[5.]])}))


# In[16]:

def linear_function():
    np.random.seed(1)
    X = tf.constant(np.random.randn(3,1))
    W = tf.constant(np.random.randn(4,3))
    b = tf.constant(np.random.randn(4,1))
    Y = tf.add(tf.matmul(W,X),b)
    with tf.Session() as sess:
        result = sess.run(Y)
    return result    


# In[18]:

linear_function()


# In[19]:

def sigmoid(z):
    x = tf.placeholder(tf.float32)
    sigmoid = tf.sigmoid(x)
    with tf.Session() as session:
        result = session.run(sigmoid,feed_dict={x:z})
    return result    


# In[28]:

logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
print(logits)
print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))


# In[29]:

def cost(logits,labels):
    z = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z,labels=y)
    with tf.Session() as session:
        result = session.run(cost,feed_dict={z:logits,y:labels})
    return result    


# In[30]:

logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
cost = cost(logits, np.array([0,0,1,1]))
print ("cost = " + str(cost))


# In[33]:

def one_hot_matrix(labels,C):
    #axis=1是垂直方向，axis=0水平方向
    one_hot_matrix = tf.one_hot(indices=labels,depth=C,axis=0)
    with tf.Session() as session:
        result = session.run(one_hot_matrix)
    return result    


# In[34]:

labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels, C = 4)
print ("one_hot = " + str(one_hot))


# In[35]:

with tf.Session() as sess:
    print(sess.run(tf.ones([3,2])))


# In[36]:

image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])



# In[37]:

image.shape


# In[ ]:



