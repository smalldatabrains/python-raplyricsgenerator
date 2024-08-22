# python-raplyricsgenerator
An lstm model based on your favorite songs to generate rap lyrics. It's a first attempt with Keras Framework.

## data
collect data in the data folder : txt files of your favorite song or lyrics. You can use website such as genius.com to build your corpus. Initila corpus is made of french rap songs.
run ```generatecorpus.py``` to generate input data according to network needs.

## training
train the model by running ```kerasrnnclassifier.py```. Trained model will be saved with h5 file format.

## generate
generate new lyrics according to your taste by settting the beginning of a sentence in ```generator.py``` and by running it.

The rnn model is a letter based model (generating a new letter with the help of former letters). Word based model was too difficult to achieve in term of computation at that time. Letter based models allow to deal with input dimension = 26 which is quite easy to handle for a computer in deep learning.