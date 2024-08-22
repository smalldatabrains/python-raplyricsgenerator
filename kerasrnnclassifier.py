#classics
import matplotlib.pyplot as plt
import numpy as np
import random
#sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

#keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#data
with open('corpus.txt','r') as myfile:
	data=myfile.read()
# data="abc abc abc abc abc a"


#parameters
input_size=len(data)
num_classes=len(set(data))
vocabulary=sorted(set(data))
print("Corpus is made of",num_classes,"letters ")
time_steps=10
batch_size=300

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

one_hot=toonehot(data)
#data prepatation for neural network
inputs=[]
labels=[]

for i in range(0,len(one_hot)-time_steps,1):
	X_buffer=one_hot[i:i+time_steps]
	y_buffer=one_hot[i+time_steps]
	inputs.append(X_buffer)
	labels.append(y_buffer)
inputs=np.asarray(inputs)
labels=np.asarray(labels)
# print(inputs)
# print(labels)

print("inputs has shape",inputs.shape)
print("labels has shape",labels.shape)


#separate train (70%) and test set (30%)


#building the network
model=Sequential()
model.add(LSTM(256,input_shape=(time_steps,num_classes)))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss="categorical_crossentropy",optimizer="adam")

#training
history=model.fit(inputs,labels,epochs=100,batch_size=batch_size,verbose=1,shuffle=False)

#visualization of the results
plt.plot(history.history['loss'],label='train')
plt.legend()
plt.show()

model.save('classifier256.h5')

#predictions
Ypred=model.predict(inputs[0:1])
print(Ypred)
print("Ypred has shape",Ypred.shape)

predictions=np.argmax(Ypred)
print(predictions)


