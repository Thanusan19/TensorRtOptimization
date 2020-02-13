import tensorrt as trt
import tensorflow as tf
import numpy as np
import datetime

#-----------------------------------------------------
from tensorflow.summary import FileWriter
from tensorflow.python.compiler.tensorrt import trt_convert as trt

import uff
#from tensorrt.parsers import uffparser

#import pycuda.driver as cuda
#import pycuda.autoinit

from random import randint # generate a random test case
from PIL import Image
from matplotlib.pyplot import imshow #to show test case
import time #import system tools
import os
#-----------------------------------------------------


ROOT = '/home/localadmin/Documents/MLP/model_ckpt'
PATH = ROOT+'/Best_1S'
X = np.load('/home/localadmin/Documents/TEST_SET_S2S_X.npy')
Y = np.load('/home/localadmin/Documents/TEST_SET_S2S_Y.npy')

sess = tf.InteractiveSession()

saver = tf.train.import_meta_graph(PATH+'.meta')
saver.restore(sess, PATH)

PRED = sess.run('output/BiasAdd:0',feed_dict={'inputs:0':X})#,'hidden_state:0':np.zeros((3,X.shape[0],64))})

# RNN, GRU OK
HS = np.zeros((3,X.shape[0],64))
# LSTM 
#HS = np.zeros((3,2,X.shape[0],64))
# MLP, CNN
# No HS, input --> inputs, pred [:,-1,:] --> [:,:]


start = datetime.datetime.now()
for i in range(10):
    PRED = sess.run('output/BiasAdd:0',feed_dict={'inputs:0':X})#,'hidden_state:0':HS})
end = datetime.datetime.now()

elapsed_time = (end-start).total_seconds()
#RMSE = np.sqrt(np.mean((PRED[:,-1,:] - Y[:,-1,:])**2))
#STD = np.std((PRED[:,-1,:] - Y[:,-1,:])**2)
RMSE = np.sqrt(np.mean((PRED[:,:] - Y[:,-1,:])**2))
STD = np.std((PRED[:,:] - Y[:,-1,:])**2)

print("Elapsed time without TensorRT: ")
print(elapsed_time)

print(Y[:,-1,:])

FileWriter("__tb", sess.graph)
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
print("%d ops in the final graph." % len(output_graph_def.node))

#----------------------------------#
#Conversion TF graph def as UFF #
#---------------------------------#

#with tf.Session() as sess:
    # First deserialize your frozen graph:
#    with tf.gfile.GFile('/home/localadmin/Documents/GRU/model_ckpt/frozen_model.pb', 'rb') as f:
#        frozen_graph = tf.GraphDef()
 #       frozen_graph.ParseFromString(f.read())
    # Now you can create a TensorRT inference graph from your
    # frozen graph:
  #  converter = trt.TrtGraphConverter(
  #          input_graph_def=frozen_graph,
  #          nodes_blacklist=['logits', 'classes']) #output nodes
  #  trt_graph = converter.convert()
    # Import the TensorRT graph into a new graph and run:
    #output_node = tf.import_graph_def(
    #    trt_graph,
    #    return_elements=['logits', 'classes'])
    #sess.run(output_node)           


uff_model = uff.from_tensorflow_frozen_model(ROOT+'/frozen_model.pb',['output/BiasAdd'],output_filename = 'model.uff')

