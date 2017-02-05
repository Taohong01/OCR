
"""Builds the MNIST network.
Implements the inference/
                loss/
                training pattern
                for model building.
1. inference() - Builds the model as far as is required
    for running the network forward to make predictions.
    i.e., from x ->> y, input to output for predictions,
    This is called inference()
2. loss() - Adds to the inference model the layers required
    to generate loss.
    from y to calcuate the deviation from true values,
    using such as, (y-t)^2 etc, loss function

3. training() - Adds to the loss model the Ops required
    to generate and apply gradients.
    take dervative of loss function with respect to each
    coupling coefficients, w using backpropogation.

This file is used by the various "fully_connected_*.py"
files and not meant to be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

# The MNIST dataset has 10 classes,
# representing the digits 0 through 9. Because there are 10 digits 0,...9.
NUM_CLASSES = 10
# this is the final prediction output


# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
# the size of image is 28 x 28.

def inference(images, hidden1_units, hidden2_units):
  """Build the MNIST model up to where it may be used for inference.
  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

    these are the inpurt arguments needed for forward calculation
    first, we need an input image,
    then the number of hidden units in each layer, and there are 2 hidden
    layers in total.


  Returns:
    softmax_linear: Output tensor with the computed logits.

    the return output is supposedly from a softmax having 10 outputs,
    corresponding 0, 1, ..., 9

    softmax() =?
    define a number of classes, for a fix number of classes
    it outputs only one 1 to one of the class and the rest
    of the outputs are zeros.
    sigmoid(z)=1/(1-exp(-z))= {1: when z->> +inf, 0: when z->> -inf},
    softmax is a generalization of sigmoid for arbitrary number of classes.
    Apparently, it needs a normalization process in order to make sure
    the output satisfy a normalization condition.

    what outputs should a softmax be? 0.0, 0.1, 0.3, ...., 0.9?
    softmax(zj) = exp(zj)/Sigma_i = {1,..k}(exp(zi))
    the meaning of softmax can be explained as the probability of
    class j when input argment equals zj. softmax value is always
    in between 0 and 1. The sum of the softmax for different zj
    all together will be one. This satifies the probability conservation
    principle.
  """
  # Hidden 1: this is a defintion of the paramters used in hidden layer 1
  # there are 2 sets of parameters, one is weights and the other is biases.
  # because this is a fully connected model(the filter is not a patch)
  # the parameter for each filter is given by,
  # the image pixel number x hidden units.
  # the number of biases are given by the hidden unites.
  # because there is image pixel number + 1 bias inputs for each hidden unit,
  # to make sure the input value is smaller or close to 10,
  # we assume the weight distribution between -1, +1, is rescaled by
  # a factor of 1/sqrt(image pixel number)
  # similar for second hidden units, the only difference is that the
  # inputs are not from the image, but the hidden unites of the first
  # layer.

  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)



  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)





  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits # these are only logits, not the output of the softmax yet


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss


def training(loss, learning_rate):
  """Sets up the training Ops.
  Creates a summarizer to track the loss over time in TensorBoard.
  Creates an optimizer and applies the gradients to all trainable variables.
  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))