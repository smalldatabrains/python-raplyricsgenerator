import os
import numpy as np

files=os.listdir("data/")
print(files)
os.chdir("data/")

data=[]

for file in files:
	f=open(file,'r')
	verses= [line.strip().lower() for line in f]
	data.extend(verses)

corpus=' '.join(data)
print(corpus)
print(len(corpus),"letters in our database")

text_file=open("../corpus.txt","w")
text_file.write(corpus)
text_file.close()