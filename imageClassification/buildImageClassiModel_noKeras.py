#-------------------------------------
#Image CLassification model
#
#Tutorial URL=https://www.tensorflow.org/tutorials/images/classification
#-------------------------------------

import tensorflow as tf
import tensorflow.keras.layers as Layers
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#
#Global Variables
#
batch_size = 128
epochs = 3
IMG_HEIGHT = 64
IMG_WIDTH = 64


#
#Load data
#
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures


#
#Understand the data
#
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)


#
#Data Preparation
#
train_cats_list = [os.path.join(train_cats_dir,x) for x in os.listdir(train_cats_dir)]
train_dogs_list = [os.path.join(train_dogs_dir,x) for x in os.listdir(train_dogs_dir)]

bs = 64

def get_train_batch(train_cats_list, train_dogs_list,bs):
    cats = np.random.choice(train_cats_list, int(bs/2))
    dogs = np.random.choice(train_dogs_list, int(bs/2))
    IMG_BUFFER = []
    LBL_BUFFER = []
    for i in cats:
        IMG_BUFFER.append(cv2.resize(cv2.imread(i,-1),(IMG_HEIGHT,IMG_WIDTH),interpolation = cv2.INTER_AREA))
        LBL_BUFFER.append([0])
    for i in dogs:
        IMG_BUFFER.append(cv2.resize(cv2.imread(i,-1),(IMG_HEIGHT,IMG_WIDTH),interpolation = cv2.INTER_AREA))
        LBL_BUFFER.append([1])
    BATCH_X = np.array(IMG_BUFFER)
    BATCH_Y = np.array(LBL_BUFFER)
    s = np.arange(BATCH_X.shape[0])
    np.random.shuffle(s)
    BATCH_X = BATCH_X[s]
    BATCH_Y = BATCH_Y[s]
    return BATCH_X, BATCH_Y

x = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, 3], name='inputs')
y = tf.placeholder(tf.float32, shape=[None,1], name='target')
step = tf.placeholder(tf.int32, name='step')
is_training = tf.placeholder(tf.bool, name='is_training')
drop_rate = tf.placeholder(tf.float32, name='keep_prob')

xr = Layers.Conv2D(16, 3, padding='same', activation='relu')(x)
xr = Layers.MaxPooling2D(pool_size=(2, 2))(xr)
xr = Layers.Conv2D(32, 3, padding='same', activation='relu')(xr)
xr = Layers.MaxPooling2D()(xr)
xr = Layers.Conv2D(64, 3, padding='same', activation='relu')(xr)
xr = Layers.MaxPooling2D()(xr)
xr = Layers.Flatten()(xr)
xr = Layers.Dense(256, activation='relu')(xr)
y_ = Layers.Dense(1, activation='sigmoid', name='out')(xr)

bce = tf.keras.losses.BinaryCrossentropy()
loss = bce(y, y_)

train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

#
#Train the model
#

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    batch_xs = np.zeros([64,64,64,3])
    batch_ys = np.zeros([64,1])
    for i in range(1000):
        batch_xs, batch_ys = get_train_batch(train_cats_list, train_dogs_list,bs)
        _ = sess.run(train_step, feed_dict = {x: batch_xs,
                                              y: batch_ys,
                                              drop_rate: 0.1,
                                              step: i,
                                              is_training: True})
     

    my_graph=tf.get_default_graph()
    tf.train.write_graph(my_graph, 'saved_model/',
                     'test_saved_model.pb', as_text=False)

    [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

