import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from scipy.misc import imsave, imread, imresize
import numpy as np
import argparse
from keras.models import model_from_yaml
import re
import base64
import pickle


def load_model(bin_dir):
   
    # load YAML and create model
    yaml_file = open('%s/model.yaml' % bin_dir, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    model.load_weights('%s/model.h5' % bin_dir)
    return model

def predict(x):
    
    # read parsed image back in 8-bit, black and white mode (L)
    x = np.invert(x)
    # Visualize new array
    imsave('resized.png', x)
    x = imresize(x,(28,28))

    # reshape image data for use in neural network
    x = x.reshape(1,28,28,1)

    # Convert type to float32
    x = x.astype('float32')

    # Normalize to prevent issues with model
    x /= 255

    # Predict from model
    out = model.predict(x)

    # Generate response
    response = {'prediction': chr(mapping[(int(np.argmax(out, axis=1)[0]))]),
                'confidence': str(max(out[0]) * 100)[:6]}

    return response

def load():
    model = load_model('bin/')
    mapping = pickle.load(open('%s/mapping.p' % 'bin/', 'rb'))

    
     

