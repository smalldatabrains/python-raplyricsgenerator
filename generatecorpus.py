import os
import numpy as np

files=os.listdir("data/")
print(files)
os.chdir("data/")

data=[]

for file in files:
	f=open(file,'r',encoding="utf-8", errors='ignore')
	verses= [line.strip().lower() for line in f]
	data.extend(verses)

corpus=' '.join(data)
print(corpus)
print(len(corpus),"letters in our database")

text_file=open("../corpus.txt","w",encoding="utf-8")
text_file.write(corpus)
text_file.close()