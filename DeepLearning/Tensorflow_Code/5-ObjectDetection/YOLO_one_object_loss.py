
# coding: utf-8

# In[1]:


"""
YOLO计算一张图片中的一个目标的loss
"""
"""
[Common]
image_size: 448
batch_size: 16
num_classes: 20
max_objects_per_image: 20
[DataSet]
name: yolo.dataset.text_dataset.TextDataSet
path: data/pascal_voc.txt
thread_num: 5
[Net]
name: yolo.net.yolo_tiny_net.YoloTinyNet
weight_decay: 0.0005
cell_size: 7
boxes_per_cell: 2
object_scale: 1
noobject_scale: 0.5
class_scale: 1
coord_scale: 5
"""


# In[13]:

import tensorflow as tf
import numpy as np


# In[14]:

sess = tf.InteractiveSession()


# In[35]:

label = np.array([160.,160.,100.,200.,11.],dtype=np.float32)
predict = np.random.rand(7,7,30)
predict = predict.astype(np.float32)
x_center = label[0]
y_center = label[1]
w = label[2]
h = label[3]
object_scale = 1
noobject_scale = 0.5
class_scale = 1
coord_scale = 5
print("label_Probilities:",label_Probilities.eval())


# In[36]:

#objects 7*7，含有目标的位置设为1，其余为0
#response 7*7 目标中心所在网格设为1，其余为0
x_min = tf.floor(((x_center - w/2) / 448) * 7)  #0
x_max = tf.ceil(((x_center + w/2) / 448) * 7)   #3
y_min = tf.floor(((y_center - h/2) / 448) * 7)  #0
y_max = tf.ceil(((y_center + h/2) / 448) *7)   #3
temp = tf.cast(tf.stack([y_max-y_min,x_max-x_min]),tf.int32)
objects = tf.ones(temp,tf.float32)
temp_1 = tf.cast(tf.stack([y_min,7-y_max,x_min,7-x_max]),tf.int32)
temp_1 = tf.reshape(temp_1,[2,2])
objects = tf.pad(objects,temp_1)
objects.eval()
x_center_min = tf.floor(x_center/448*7)
y_center_min = tf.floor(y_center/448*7)
response = tf.ones([1,1],tf.float32)
temp_2 = tf.cast(tf.stack([y_center_min,7-(y_center_min+1),x_center_min,7-(x_center_min+1)]),tf.int32)
temp_2 = tf.reshape(temp_2,[2,2])
response = tf.pad(response,temp_2)
print("objects:\n",objects.eval())
print("response:\n",response.eval())


# In[37]:

#预测的类别概率分布
predict_Probilities = predict[:, :, 0:20]
#groud truth probabilities
label_Probilities = tf.one_hot(tf.cast(label[4],tf.int32),20,dtype=tf.float32)
#class loss
class_loss = tf.nn.l2_loss(tf.reshape(objects,(7,7,1)) * (predict_Probilities - label_Probilities )) * class_scale


# In[51]:

##计算confidence loss,分为有object和noobject,对于某个目标，预测值里只有一个bbox的confidence为有object，iou最大那个
#Confidence = P(object)*IOU

#将预测的坐标值映射为相对于整张图片的坐标值
#某个bbox的坐标xcenter和ycenter值在0-1之间，是相对某个网格的，长和宽是绝对值
predict_boxes = predict[:,:,22:]
predict_boxes = tf.reshape(predict_boxes,[7,7,2,4]) #两个bbox
predict_boxes = predict_boxes*[448/7,448/7,448,448] 
base_boxes = np.zeros([7,7,4])
base_boxes = base_boxes.astype(np.float32)
for y in range(7):
    for x in range(7):
        base_boxes[y,x,:] = [(448/7)*y,(448/7)*x,0,0] 
#由于每个网格有多个预测框，所以需要重复叠加使其大小与predict_boxes一致
base_boxes = tf.tile(tf.reshape(base_boxes,[7,7,1,4]),[1,1,2,1])
predict_boxes = predict_boxes+base_boxes 
#计算IOU
boxes1 = predict_boxes
boxes2 = label[0:4]
boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                      boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
boxes2 = tf.cast(tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                      boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2]),tf.float32)

left_up = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])  #相交的左上角坐标
right_down = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:]) #相交的右下角坐标
#intersection
intersection = right_down - left_up #[w,h]
inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]
mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)
inter_square = mask * inter_square  #长和宽只要有一个小于0则面积为0

#calculate the boxs1 square and boxs2 square
square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])
predict_iou = inter_square/(square1 + square2 - inter_square + 1e-6) 
predict_iou = tf.cast(predict_iou,tf.float32)
label_Confidence = predict_iou * tf.reshape(response,[7,7,1]) #包含object中心的网格的两个bbox的confidence不为0，其余都为0
#两个bbox只取iou较大的那个，另一个也置为0，如下I tensor

#calculate I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL] 
#
I = predict_iou * tf.reshape(response, (7,7,1))
max_I = tf.reduce_max(I, 2, keep_dims=True) #这儿2表示第2维，（0,1,2）三维
I = tf.cast((I >= max_I), tf.float32) * tf.reshape(response, (7,7,1))
#calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
no_I = tf.ones_like(I, dtype=tf.float32) - I 

#predict_confidence
predict_Confidence = predict[:,:,20:22]

#object_confidence_loss
object_loss = tf.nn.l2_loss(I * (predict_Confidence - label_Confidence)) * object_scale
#object_loss = tf.nn.l2_loss(I * (p_C - (C + 1.0)/2.0)) * self.object_scale

#noobject_confidence_loss
#noobject_loss = tf.nn.l2_loss(no_I * (p_C - C)) * self.noobject_scale
noobject_loss = tf.nn.l2_loss(no_I * (predict_Confidence)) * noobject_scale


# In[53]:

#calculate truth x,y,sqrt_w,sqrt_h 0-D  只计算包含object中心的网格对应的iou最大的那个bbox
x = label[0]
y = label[1]
sqrt_w = tf.sqrt(tf.abs(label[2]))
sqrt_h = tf.sqrt(tf.abs(label[3]))
#calculate predict p_x, p_y, p_sqrt_w, p_sqrt_h 3-D [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
p_x = predict_boxes[:, :, :, 0]
p_y = predict_boxes[:, :, :, 1]
p_sqrt_w = tf.sqrt(tf.minimum(448 * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
p_sqrt_h = tf.sqrt(tf.minimum(448 * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))
#coord_loss
coord_loss = (tf.nn.l2_loss(I * (p_x - x)/(448/7)) +
                tf.nn.l2_loss(I * (p_y - y)/(448/7)) +
                tf.nn.l2_loss(I * (p_sqrt_w - sqrt_w))/ 448 +
                tf.nn.l2_loss(I * (p_sqrt_h - sqrt_h))/448) * coord_scale



# In[55]:

loss = class_loss + object_loss + noobject_loss + coord_loss

