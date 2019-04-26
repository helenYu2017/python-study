import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

h=tf.constant("Hello,Tensorflow!")
s=tf.Session();
print(h)
print(s)