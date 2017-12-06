# coding: utf-8
# # LSTM 文本生成 字符级别
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import codecs

raw_text = codecs.open('Winston_Churchil.txt','r',encoding='utf-8').read()
raw_text = raw_text.lower()
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

for item in ((c, i) for i, c in enumerate(chars)):
    print(item)


#训练集  输入前100个字符，输出下一个
seq_length = 100
x = []
y = []
for i in range(0, len(raw_text) - seq_length):
    given = raw_text[i:i + seq_length]
    predict = raw_text[i + seq_length]
    x.append([char_to_int[char] for char in given])
    y.append(char_to_int[predict])

# 此刻，楼上这些表达方式，类似就是一个词袋，或者说 index。
# 
# 接下来做两件事：
# - 已经有了一个input的数字表达（index），要把它变成LSTM需要的数组格式： [样本数，时间步伐，特征维度]
# - 对于output，用one-hot做output的预测可以有更好的效果，相对于直接预测一个准确的y数值的话。
n_patterns = len(x)
n_vocab = len(chars)
# 把x变成LSTM需要的样子
x = numpy.reshape(x, (n_patterns, seq_length, 1))
# 简单normal到0-1之间
x = x / float(n_vocab)
# output变成one-hot
y = np_utils.to_categorical(y)
# # LSTM模型构建

model = Sequential()
model.add(LSTM(256,input_shape=(x.shape[1],x.shape[2]))) #不关心样本个数
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="adam")
model.fit(x,y,nb_epoch=50,batch_size=4096)

def predict_next(input_array):
    x = numpy.reshape(input_array, (1, seq_length, 1))
    x = x / float(n_vocab)
    y = model.predict(x)
    return y

def string_to_index(raw_input):
    res = []
    for c in raw_input[(len(raw_input)-seq_length):]:
        res.append(char_to_int[c])
    return res

def y_to_char(y):
    largest_index = y.argmax()
    c = int_to_char[largest_index]
    return c

def generate_article(init, rounds=200):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_char(predict_next(string_to_index(in_string)))
        in_string += n
    return in_string

init = 'His object in coming to New York was to engage officers for that service. He came at an opportune moment'
article = generate_article(init)
print(article)

