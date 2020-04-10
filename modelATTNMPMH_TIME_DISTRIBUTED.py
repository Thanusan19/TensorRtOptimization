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
#----------------------------------------------------
MAX_BATCH_SIZE=100000
batch_size=2000
inference_Loop=1



ROOT = '/home/localadmin/Documents/ATTNMPMH_TIME_DISTRIBUTED/model_ckpt'
PATH = ROOT+'/Best_1S'
X = np.load('/home/localadmin/Documents/TEST_SET_S2S_X.npy')
Y = np.load('/home/localadmin/Documents/TEST_SET_S2S_Y.npy')

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
elapsed_time_TRT=0

class ModelData(object):
    MODEL_FILE = "modelATTNMPMH_TIME_DISTRIBUTED.uff" #"model.uff"
    INPUT_NAME ="inputs:0"
    INPUT_SHAPE = X.shape[1:]
    OUTPUT_NAME ='time_distributed_10/outputs/BiasAdd'

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

	sess = tf.InteractiveSession()

	saver = tf.train.import_meta_graph(PATH+'.meta')
	saver.restore(sess, PATH)

	# RNN, GRU OK
	HS = np.zeros((3,X.shape[0],64))

	# LSTM 
	#HS = np.zeros((3,2,X.shape[0],64))
	# MLP, CNN
	# No HS, input --> inputs, pred [:,-1,:] --> [:,:]


	start = datetime.datetime.now()
	for i in range(inference_Loop):
	    PRED = sess.run(ModelData.OUTPUT_NAME,feed_dict={ModelData.INPUT_NAME:X}) #,'hidden_state:0':HS})
	end = datetime.datetime.now()

	elapsed_time = (end-start).total_seconds()
	#print(PRED.shape)
	print("Elapsed time without TensorRT: ",elapsed_time)

	#--------------------------------------------------#
	#Get output nodes  Names
	#--------------------------------------------------#
	graph = sess.graph
	#print([node.name for node in graph.as_graph_def().node])
	output_node_names=[node.name for node in graph.as_graph_def().node]

	#----------------------------------------------------------------#
	#Make a frozen model(.pb) of the TF  model in order to convert it into UFF#
	#----------------------------------------------------------------#
	# We use a built-in TF helper to export variables to constants
	output_graph_def = tf.graph_util.convert_variables_to_constants(
	           sess, # The session is used to retrieve the weights
	           tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
	           output_node_names # The output node names are used to select the usefull nodes
	        ) 

	input_checkpoint=PATH
	# We precise the file fullname of our freezed graph
	absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
	output_graph = absolute_model_dir + "/frozen_model.pb"

	# Finally we serialize and dump the output graph to the filesystem
	with tf.gfile.GFile(output_graph, "wb") as f:
	    f.write(output_graph_def.SerializeToString())
	#print("%d ops in the final graph." % len(output_graph_def.node))

	#----------------------------------#
	#Conversion TF graph def as UFF #
	#---------------------------------#
	uff_model = uff.from_tensorflow_frozen_model(ROOT+'/frozen_model.pb',[ModelData.OUTPUT_NAME],output_filename = ModelData.MODEL_FILE)


	#----------------------------------#
	#Build the engine and run inference#
	#---------------------------------#
	model_file=ModelData.MODEL_FILE
	builder=build_engine(model_file)



if __name__ == '__main__':
    main()
