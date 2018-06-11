
# coding: utf-8

# In[1]:


import cv2 
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer


# one time activity starts****************************************************


# In[ ]:



import numpy as np


IMAGE_SIZE = 150

CHANNEL = 1


def tf_resize_images(X_img_file_paths):
    X_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, CHANNEL))
    tf_img = tf.image.resize_images(X, (IMAGE_SIZE, IMAGE_SIZE), 
                                    tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Each image is resized individually as different image may be of different size.
        for index, file_path in enumerate(X_img_file_paths):
            image = cv2.imread(file_path,0)
            if image is not None:
              image = image.reshape(image.shape[0],image.shape[1],1)
              resized_img = sess.run(tf_img, feed_dict = {X: image})
              X_data.append(resized_img)

    X_data = np.array(X_data, dtype = np.float32) # Convert to numpy
    return X_data

# Extracting train data
files = os.listdir("......./Train_Images")
filepaths = []
for x in files:
    if x.endswith('.png'):
      filepaths.append("......./Train_Images"+ x)
filepaths.sort()


X = tf_resize_images(filepaths)

print(" Printing train file")
np.save('......./Train_Images/X_150.npy', X, allow_pickle = False)

# Extracting test data

tfiles = os.listdir("........../Test_Images")
tfilepaths = []
for x in tfiles:
    if x.endswith('.png'):
      tfilepaths.append("........../Test_Images"+ x)
tfilepaths.sort()

X_tdata =  tf_resize_images(tfilepaths)
tfiles = os.listdir("/........../Test_Images")
print(" Printing test file")
np.save('........../Test_Images/X_tdata_150.npy', X_tdata , allow_pickle = False)

# one time activity ends****************************************************


# Run from here every time...... Its a lot faster. If you wanna change dimensions then rerun whole code again once changeing image size.

X = np.load('....../X_150.npy')

X_td = np.load('......./X_tdata_150.npy')


df = pd.read_csv("........../Train_Labels.csv")
num_pics = 640
df = df.sort_values("Images")
y = df['Labels'][:num_pics]
X_ = X
encoder = LabelBinarizer()
y = encoder.fit_transform(y)
yhat = []
for i  in y:
  i = list(i)
  if i[0] == 0:
      i[0] = 1
      i.append(0)
      yhat.append(i)
  else:
      i[0] = 0
      i.append(1)
      yhat.append(i)
yhat = np.array(yhat)

yhat = yhat.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X_, yhat, test_size=0.20, random_state = 20)

width = 150 # width of the image in pixels 
height = 150 # height of the image in pixels
flat = width * height # number of pixels in one image 
class_output = 2 # number of possible classifications for the problem
n_classes = 2
dropout = 0.75
batch_size = 100 # Dont reduce batch size. Its causing issues


X  = tf.placeholder(tf.float32, shape=[None, width, height, 1])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)



def convolutional_neural_network(X):
#     x = tf.reshape(x, shape=[-1, 128, 128, 1])
    conv1 = tf.layers.conv2d(X, 32, 5, activation=tf.nn.relu, padding = 'SAME')
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

    conv2 = tf.layers.conv2d(conv1, 64, 5, activation=tf.nn.relu , padding = 'SAME')
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
    
    conv3 = tf.layers.conv2d(conv2, 128, 3, activation=tf.nn.relu , padding = 'SAME')
    conv3 = tf.layers.max_pooling2d(conv3, 2, 2)
    
    conv4 = tf.layers.conv2d(conv3, 64, 3, activation=tf.nn.relu , padding = 'SAME')
    conv4 = tf.layers.max_pooling2d(conv4, 2, 2)
    
    conv5 = tf.layers.conv2d(conv4, 32, 3, activation=tf.nn.relu , padding = 'SAME')
    conv5 = tf.layers.max_pooling2d(conv5, 2, 2)

    fc = tf.contrib.layers.flatten(conv5)
    fc = tf.layers.dense(fc, 128, activation=tf.nn.relu)
#     fc = tf.layers.batch_normalization(fc)
    fc = tf.layers.dropout(fc, rate=dropout)

    output = tf.layers.dense(fc, 2)
    return output

def train_neural_network(X):
    prediction = convolutional_neural_network(X)
    prediction = tf.reshape(prediction, [-1, n_classes])
#     y_ = tf.reshape(y_, [-1, 2])
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction ,labels = y_) )
#     optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9,beta2=0.999, epsilon=1e-08).minimize(cost)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9,momentum=0.1,epsilon=1e-10).minimize(cost)
    old = 0
    hm_epochs = 50
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for bid in range(int(num_pics/batch_size)):
                epoch_x = X_train[bid*batch_size:(bid+1)*batch_size]
                epoch_y = y_train[bid*batch_size:(bid+1)*batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={ X: epoch_x, y_: epoch_y, keep_prob: dropout})
                epoch_loss += c                
            print('Epoch' , epoch , " cost", c )
            if c < 0.01 or old == c:
                break
            else:
                old = c
        
        print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',c)
  
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({X:X_test, y_:y_test, keep_prob: dropout}))
        result = prediction.eval(feed_dict={ X: X_td, keep_prob: dropout})
        return result
        
result = train_neural_network(X)

result = tf.nn.sigmoid(result)

sess = tf.InteractiveSession()
result = sess.run(result)

# converting probabilities to absolute values
results = []
for p in result:
    if p[0] > p[1]:
        results.append(0)
    else:
        results.append(1)
len([q for q in results if q == 0])
        
# Making a list of file names
tfiles = os.listdir("............../Test_Images")
tfiles.sort()
f = [f[:-4] for f in tfiles ]

# Making a result CSV
df = pd.DataFrame()
df['Images'] = f
df['Labels'] = results
df.to_csv('................../Result.csv', index=False)

