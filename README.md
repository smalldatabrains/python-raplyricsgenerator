# python-raplyricsgenerator
An lstm model based on your favorite songs to generate rap lyrics. It's a first attempt with Keras Framework.

## data
collect data in the data folder : txt files of your favorite song or lyrics. You can use website such as genius.com to build your corpus. Initila corpus is made of french rap songs.
run ```generatecorpus.py``` to generate input data according to network needs. Vocabulary size (ie list of characters that can be printed) is made out of this corpus. Every matrix is being computed regarding this list of characters, therefore everytime you change the corpus, you have to retrain a new model.

## training
train the model by running ```kerasrnnclassifier.py```. Trained model will be saved with h5 file format. A small number of epoch is preset (100 epoch needs around an hour of training on a CPU), but you may consider increase this number at monitor your deep learning KPI : loss curve decreasing, accuracy, etc...

## generate
generate new lyrics according to your taste by settting the beginning of a sentence in ```generator.py``` and by running it.

The rnn model is a letter based model (generating a new letter with the help of former letters). Word based model was too difficult to achieve in term of computation at that time. Letter based models allow to deal with input dimensions less than 100 which is quite easy to handle for a computer in deep learning (whereas vocabulary models usually consists on more than 20 thousands words.). Techniques of word embedding would allow to reduce the dimensions of your system.