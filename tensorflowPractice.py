# -*- coding: utf-8 -*-
"""
Tensorflow practice

Created on Sat Apr  2 17:50:42 2016

@author: Tao
"""
import tensorflow as tf
import numpy as np
from termcolor import colored

x_data = np.random.rand(3).astype(np.float32)
y_data = x_data * 0.1 + 0.3
print x_data
print y_data

W = tf.Variable(tf.random_uniform([5,3], -10, 15, dtype = tf.float32))
b = tf.Variable(tf.ones([3,3]))
print '========== this is W: ',W
print '+++++++++++ this is b : ',b 

print
print



docs = """ how to instantiate a constant tensor:
    we need to define a list of values,
    the shape of the matrix or vectors,
    the data type.
"""
print colored(docs, 'red')

C1 = tf.constant([1, 2, 3, 4, 5], shape=[1,5], dtype = tf.int32 )
C2 = tf.constant([[3], [4]], shape=[3,1], dtype = tf.float32)
print colored('the tensor constant C1 is :\n',
              'green'), C1
print colored('the tensor constant C2 is :\n', 
              'green'), C2


print 'the shape of C2', C2.get_shape()
print 

C3 = tf.as_dtype(tf.float32)
print colored('convert C1 value data type from int32 to float32',
              'blue'), C3 

#Product1 = tf.matmul(C1, C2)
#print Product1
C4 = tf.random_shuffle(C1)
Product2 = tf.sqrt(tf.abs(tf.sin(tf.matmul(W, C2))))
init_op = tf.initialize_all_variables()

sess = tf.Session()
print '---- Session object is : ', sess

result = sess.run(init_op)
print result

W_result = sess.run(W)
print W_result

b_result = sess.run(b)
print 'this is the result for b\n', b_result

#result = sess.run(Product1)
#print result

Prod2_result = sess.run(Product2)
print Prod2_result

C4_result = sess.run(C4)
print C4_result
print sess.run(C1)