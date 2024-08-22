import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import keras
#data
with open('corpus.txt','r') as myfile:
	data=myfile.read()
# data="abc abc abc abc abc a"


#parameters
input_size=len(data)
num_classes=len(set(data))
vocabulary=sorted(set(data))
print("Corpus is made of",num_classes,"letters ")
time_steps=5
batch_size=100

#build dics
value_to_idx=dict((v,i) for i,v in enumerate(vocabulary))
idx_to_value=dict((i,v) for i,v in enumerate(vocabulary))
print(value_to_idx)
print(idx_to_value)


def toonehot(text):
	integer_encoded=[value_to_idx[value] for value in text]
	one_hot=[]

	#one hot
	for  value in text:
		idx=value_to_idx.get(value)
		x_temp=np.zeros(num_classes)
		x_temp[idx]=1
		one_hot.append(x_temp)
	return one_hot


#generate text
print("---------------------------------------")
print("TEXT GENERATION")
# load the network weights
filename = "classifier256.h5"
model=keras.models.load_model(filename)

# pick a random seed
start = "je su" #input has to be lower case
integer_list=[value_to_idx[value] for value in start]
print("initial integer list:",integer_list)
inputs=np.asarray(toonehot(start))
print(inputs)
inputs=np.reshape(inputs,(1,inputs.shape[0],inputs.shape[1]))
print(type(inputs))
print(inputs.shape)
# generate characters
for i in range(1000):
	print(i,"th loop")
	next=model.predict(inputs)
	integer_list.append(np.random.choice(num_classes,p=next.ravel()))
	last_five=integer_list[-5:]
	print(last_five)
	inputs=[]
	for value in last_five:
		x_temp=np.zeros(num_classes)
		x_temp[value]=1
		inputs.append(x_temp)
	inputs=np.asarray(inputs)
	inputs=np.reshape(inputs,(1,inputs.shape[0],inputs.shape[1]))
generated_text=[idx_to_value[idx] for idx in integer_list]
print(''.join(generated_text))
print ("Done.")
