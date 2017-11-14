#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import tensorflow as tf
import numpy as np
import keras


def compute_square_norm(input, axis=None):
  """
  求输入向量的模的平方，需要输入向量和聚合的axis，返回输出向量。
  原理是对向量所有维度都平方，然后对某一个维度进行聚合，如果不需要保留其他维度初始值可以直接用tf.reduce_sum聚合所有维度。
  注意，返回值保留和聚合的axis，因此输入向量shape和输出向量shape相同。
    
  Args:
    input: The tensor to compute square norm. Example: [[1, 2, 3], [1, 4, 9]]
      
  Return:
    The tensor with square norm, which has same shape of input.
  """

  square_input = tf.square(input)
  square_norm = tf.reduce_sum(square_input, axis=axis, keep_dims=True)

  return square_norm


def compute_norm_from_squarenorm(input):
  """
  从向量的膜的平方求向量的膜，输入是前面计算得到的向量的模的平方，输出是向量的模。
  原理是对向量直接求开方。
  
  Args:
    input: The tensor to compute the norm.
  
  Retrun:
    The tensor with norm, which has same shape of input.
  """

  epsilon = 1e-10
  norm = tf.sqrt(input + epsilon)

  return norm


def dynamic_routing(input):
  """
  CapsNet的动态路由算法实现，输入是底层的capsule向量，输出是上层的capsule向量。
  
  Args:
    input: Example shape is [batch_size, 1152, 1, 8, 1].
    
  Return:
    Example shape is [batch_size, 1, 10, 16, 1]
  """

  batch_size = 8
  routing_iteration = 3

  # 初始化b为0
  # Shape is [1, 1152, 10, 1, 1]
  b_ij = tf.constant(np.zeros([1, 1152, 10, 1, 1], dtype=np.float32))

  # 初始化w为随机值
  # Shape is [1, 1152, 10, 8, 16]
  w = tf.get_variable(
      'Weight',
      shape=(1, 1152, 10, 8, 16),
      dtype=tf.float32,
      initializer=tf.random_normal_initializer(stddev=0.01))

  # 修改Input tensor增加batch size
  # [batch_size, 1152, 1, 8, 1] -> [batch_size, 1152, 10, 8, 1]
  input = tf.tile(input, [1, 1, 10, 1, 1])
  # [1, 1152, 10, 8, 16] -> [batch_size, 1152, 10, 8, 16]
  w = tf.tile(w, [batch_size, 1, 1, 1, 1])

  # 通过input和w矩阵相乘u_ij
  # [batch_size, 1152, 10, 8, 16] * [batch_size, 1152, 10, 8, 1] -> [batch_size, 1152, 10, 16, 1]
  u_ij = tf.matmul(w, input, transpose_a=True)

  # 循环遍历来更新b_ij和c_ij
  for i in range(routing_iteration):
    # 通过对b_bj求softmax得到c_ij
    # [1, 1152, 10, 1, 1] -> [1, 1152, 10, 1, 1]
    c_ij = tf.nn.softmax(b_ij, dim=3)
    # [1, 1152, 10, 1, 1] -> [batch_size, 1152, 10, 1, 1]
    c_ij = tf.tile(c_ij, [batch_size, 1, 1, 1, 1])

    # 通过c_ij和u_ij矩阵相乘得到s_j
    # [batch_size, 1152, 10, 1, 1] * [batch_size, 1152, 10, 16, 1] -> [batch_size, 1152, 10, 16, 1]
    s_j = tf.multiply(c_ij, u_ij)
    # [batch_size, 1152, 10, 16, 1] -> [batch_size, 1, 10, 16, 1]
    s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True)

    # 通过s_j求squash得到v_j
    # [batch_size, 1, 10, 16, 1] -> [batch_size, 1, 10, 16, 1]
    v_j = squash(s_j)
    # [batch_size, 1, 10, 16, 1] -> [batch_size, 1152, 10, 16, 1]
    v_j_tile = tf.tile(v_j, [1, 1152, 1, 1, 1])

    # 通过u_ij和v_j矩阵相乘得到b_ij增量值
    # [batch_size, 1152, 10, 16, 1] * [batch_size, 1152, 10, 16, 1] -> [batch_size, 1152, 10, 1, 1]
    b_increment = tf.matmul(u_ij, v_j_tile, transpose_a=True)

    # 更新b_ij值
    # [batch_size, 1152, 10, 1, 1] -> [1, 1152, 10, 1, 1]
    b_increment = tf.reduce_sum(b_increment, axis=0, keep_dims=True)
    b_ij += b_increment

  return v_j


def squash(input):
  """
  CapsNet的squash函数实现，输入是需要激活的向量，输出是激活后的向量，可以保证输出向量的模在0到1之间。
  原理是对对输入的某一维度进行求模和模的平方，然后用论文中定义的公式即可。
  注意这里是给MNIST的CapsNet模型使用，要求对第3维（索引从0开始）进行压缩和激活。
  
  Args:
    input: The tensor to activate, example shape is [batch, 1, 1152, 10, 1, 1].
  Return:
    The tensor which has the same shape of the input.
  """

  input_square_norm = compute_square_norm(input, 3)
  input_norm = compute_norm_from_squarenorm(input_square_norm)
  input_squash = (input_square_norm /
                  (input_square_norm + 1)) * (input / input_norm)

  return input_squash


def load_mnist():
  (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

  X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
  Y_train = keras.utils.to_categorical(Y_train.astype('float32'))
  X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
  Y_test = keras.utils.to_categorical(Y_test.astype('float32'))

  return X_train, Y_train, X_test, Y_test


def get_batch_data(batch_size):
  X_train, Y_train, X_test, Y_test = load_mnist()

  data_queues = tf.train.slice_input_producer([X_train, Y_train])

  X, Y = tf.train.shuffle_batch(
      data_queues,
      batch_size=batch_size,
      capacity=batch_size * 64,
      min_after_dequeue=batch_size * 32,
      allow_smaller_final_batch=False)

  return (X, Y)


def train():
  batch_size = 8
  m_plus = 0.9
  m_minus = 0.1
  lambda_value = 0.5
  epsilon = 1e-10

  # Shapes are [batch, 28, 28, 1] and [batch, 10]
  X_train, Y_train = get_batch_data(batch_size)
  """
  X_train, Y_train = load_mnist()
  X_train = tf.reshape(X_train[0], shape=(1, 28, 28, 1))
  Y_train = tf.reshape(Y_train[0], shape=(1, 10))
  """

  with tf.variable_scope("Conv1"):
    # 卷积神经网络，filter kernel为9*9, filter个数为256, stipe为1，padding为0
    # [batch_size, 28, 28, 1] -> [batch_size, 20, 20, 256]
    layer = tf.contrib.layers.conv2d(
        X_train, num_outputs=256, kernel_size=9, stride=1, padding="VALID")

    with tf.variable_scope("PrimaryCaps"):
      # 做了8次卷积神经网络，filter kernel为9*9, filter个数为32, strip为2, padding为0
      # [batch_size, 20, 20, 256] -> [batch_size, 6, 6, 256]
      layer = tf.contrib.layers.conv2d(layer, 32 * 8, 9, 2, padding="VALID")
      # [batch_size, 6, 6, 256] -> [batch_size, 1152, 8, 1]
      layer = tf.reshape(layer, (batch_size, 6 * 6 * 32, 8, 1))

    # [batch_size, 1152, 8, 1] -> [batch_size, 1152, 8, 1]
    layer = squash(layer)

    with tf.variable_scope("DigitCaps"):
      # [batch_size, 1152, 8, 1] -> [batch_size, 1152, 1, 8, 1]
      layer = tf.reshape(layer, shape=(batch_size, -1, 1, 8, 1))

      # [batch_size, 1152, 1, 8, 1] -> [batch_size, 1, 10, 16, 1]
      layer = dynamic_routing(layer)

      # 将s_j投入 squeeze 函数以得出 DigitCaps 层的输出向量
      # [batch_size, 1, 10, 16, 1] -> [batch_size, 10, 16, 1]
      layer = tf.squeeze(layer, axis=1)

  # [batch_size, 10, 16, 1] -> [batch_size, 10, 1, 1]
  v_length = tf.sqrt(
      tf.reduce_sum(tf.square(layer), axis=2, keep_dims=True) + epsilon)
  # [batch_size, 10, 1, 1] -> [batch_size, 10]
  tf.reshape(v_length, shape=(batch_size, 10))

  # [batch_size, 10] -> [batch_size, 10]
  left = tf.square(tf.maximum(0.0, m_plus - v_length))
  left = tf.to_double(left)

  # [batch_size, 10] -> [batch_size, 10]
  right = tf.square(tf.maximum(0.0, v_length - m_minus))
  right = tf.to_double(right)

  # Margin loss
  # [batch_size, 10] -> [batch_size, 10]
  loss_vector = Y_train * left + lambda_value * (1 - Y_train) * right

  # [batch_size, 10] -> [batch_size] -> []
  loss = tf.reduce_mean(tf.reduce_sum(loss_vector, axis=1))

  optimizer = tf.train.AdamOptimizer()
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)

  init_op = tf.global_variables_initializer()
  tf.summary.scalar('loss', loss)

  epoch_number = 1
  with tf.Session() as sess:
    sess.run(init_op)

    num_batch = int(60000 / batch_size)
    """
    for epoch_index in range(epoch_number):
      for step in range(num_batch):
        _, loss_value = sess.run([train_op, loss])
        print("Epoch: {}, step, {}, loss: {}".format(epoch_index, step, loss_value))
    """

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    try:
      while not coord.should_stop():
        for epoch_index in range(epoch_number):
          for step in range(num_batch):
            _, loss_value = sess.run([train_op, loss])
            print("Epoch: {}, step, {}, loss: {}".format(
                epoch_index, step, loss_value))
    except tf.errors.OutOfRangeError:
      exit(0)
    finally:
      coord.request_stop()
      coord.join(threads)


def main():
  train()


if __name__ == "__main__":
  main()
