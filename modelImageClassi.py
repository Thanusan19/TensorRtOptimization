import tensorflow as tf
import datetime

#-----------------------------------------------------
from tensorflow.summary import FileWriter
from tensorflow.python.compiler.tensorrt import trt_convert as trt

import uff

from random import randint # generate a random test case
from PIL import Image
from matplotlib.pyplot import imshow #to show test case
import time #import system tools
import os
#-----------------------------------------------------
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import sys
import common 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#----------------------------------------------------
MAX_BATCH_SIZE=100000
batch_size=128
inference_Loop=1
IMG_HEIGHT = 15
IMG_WIDTH = 15



PATH_pb ='/home/localadmin/NNOptimization/TensorRtOptimization/imageClassification/saved_model/test_saved_model.pb'
#PATH_pb='/home/localadmin/Documents/MLP/model_ckpt/frozen_model.pb'

X = np.load('/home/localadmin/Documents/TEST_SET_S2S_X.npy')
Y = np.load('/home/localadmin/Documents/TEST_SET_S2S_Y.npy')

print(X.shape[1:])

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
elapsed_time_TRT=0

class ModelData(object):
    MODEL_FILE = "modelImageClassi.uff"
    INPUT_NAME = "inputs"
    INPUT_SHAPE = (IMG_HEIGHT,IMG_WIDTH)
    OUTPUT_NAME = 'conv2d/Relu'

def GiB(val):
    return val * 1 << 30

def build_engine(model_file):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size =GiB(4)
        builder.max_batch_size=MAX_BATCH_SIZE
        # Parse the Uff Network
        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        parser.parse(model_file, network)
        # Build and return an engine.
        return builder.build_cuda_engine(network)


def do_inference(context, bindings, inputs, outputs, stream, batch_size=batch_size):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    start_TRT = datetime.datetime.now()
    # Run inference.
    for i in range(inference_Loop):
    	context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    end_TRT = datetime.datetime.now()
    elapsed_time_TRT=(end_TRT-start_TRT).total_seconds()
    print("Elapsed time with TensorRT: ",elapsed_time_TRT)

    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def main():

	#----------------------------------#
	#Conversion TF graph def as UFF #
	#---------------------------------#
	uff_model = uff.from_tensorflow_frozen_model(PATH_pb,[ModelData.OUTPUT_NAME],output_filename = ModelData.MODEL_FILE)


	#---------#
	#Load data#
	#---------#
	_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
	path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
	PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

	train_dir = os.path.join(PATH, 'train')
	validation_dir = os.path.join(PATH, 'validation')
	train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
	train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
	validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pict$
	validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pict$

	#----------------#
	#Data Preparation#
	#----------------#

	#Read image from disk + preprocess them into tensors + set up generators to convert images into batc$
	train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
	validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

	#Applies rescaling and resizes
	train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
	                                                           directory=train_dir,
	                                                           shuffle=True,
	                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
	                                                           class_mode='binary')

	val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
	                                                              directory=validation_dir,
	                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
	                                                              class_mode='binary')

	#----------------------------------#
	#Build the engine and run inference#
	#---------------------------------#
	#model_file=ModelData.MODEL_FILE
	#builder=build_engine(model_file)


	#with builder as engine:

	   # Build an engine, allocate buffers and create a stream.
	   #inputs, outputs, bindings, stream = common.allocate_buffers(engine)
	   #with engine.create_execution_context() as context:
	       #l=[0]*MAX_BATCH_SIZE*X.shape[1]*X.shape[2]
	       #for k in range(X.shape[0]):
	          #for i in range(X.shape[1]):
	             #for j in range(X.shape[2]):
	                #l[k*X.shape[1]*X.shape[2] + j*X.shape[1] + i]=X[k][i][j]
	       #np.copyto(inputs[0].host,l)

	       # The common.do_inference function will return a list of outputs - we only have one in this case.
	       #[output] =do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)



if __name__ == '__main__':
    main()
