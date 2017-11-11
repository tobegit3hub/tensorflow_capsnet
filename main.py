#!/usr/bin/env python

import tensorflow as tf
import numpy as np



def compute_square_norm(input, axis=None):
  """
  求输入向量的模的平方。
    
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
  从向量的膜的平方求向量的膜。
  """

  norm = tf.sqrt(input)

  return norm




def dynamic_routing(input):
  """
  动态路由算法。
  
  """

  batch_size = 1
  routing_iteration = 3


  # 初始化b为0
  # Shape is [1, 1152, 10, 1, 1]
  b_ij = tf.constant(np.zeros([1, 1152, 10, 1, 1], dtype=np.float32))

  # 初始化w为随机值
  # Shape is [1, 1152, 10, 8, 16]
  W = tf.get_variable('Weight', shape=(1, 1152, 10, 8, 16), dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))

  # 修复Tensor增加batch size
  # [batch, 1152, 1, 8, 1] -> [batch, 1152, 10, 8, 1]
  input = tf.tile(input, [1, 1, 10, 1, 1])
  # [1, 1152, 10, 8, 16] -> [batch, 1152, 10, 8, 16]
  W = tf.tile(W, [batch_size, 1, 1, 1, 1])


  # 通过input和w矩阵相乘u_ij
  # Sahep is [batch, 1152, 10, 16, 1]
  u_ij = tf.matmul(W, input, transpose_a=True)

  # 循环遍历来更新b_ij和c_ij
  for i in range(routing_iteration):

    # 通过对b_bj求softmax得到c_ij
    # Shape is [1, 1152, 10, 1, 1]
    c_ij = tf.nn.softmax(b_ij, dim=3)
    # [1, 1152, 10, 1, 1] -> [batch, 1152, 10, 1, 1]
    c_ij = tf.tile(c_ij, [batch_size, 1, 1, 1,1])


    # 通过c_ij和u_ij矩阵相乘得到s_j
    # Shape is [batch, 1, 10, 16, 1]
    s_j = tf.multiply(c_ij, u_ij)
    s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True)

    # 通过s_j求squash得到v_j
    # Shape is [batch, 1, 10, 16, 1]
    v_j = squash(s_j)

    # [] -> []
    v_j = tf.tile(v_j, [1, 1152, 1, 1, 1])
    # 通过u_ij和v_j矩阵相乘得到b_ij增量值
    # Shape is [batch, 1152, 10, 1, 1]
    b_increment = tf.matmul(u_ij, v_j, transpose_a=True)

    b_increment = tf.reduce_sum(b_increment, axis=0, keep_dims=True)
    b_ij += b_increment


  return v_j




def squash(input):
  """
  
  Args;
  
  Return:
    
  """

  # [batch, 1, 1152, 10, 1, 1]?

  square_norm = compute_square_norm(input, 3)
  norm = compute_norm_from_squarenorm(square_norm)
  squash_input = (square_norm / (square_norm + 1)) * (input / norm)

  return squash_input





def model():
  batch_size = 1
  X_train = None


  with tf.variable_scope('Conv1_layer'):
    # 卷积神经网络，filter kernel为9*9, filter个数为256, stipe为1，padding为0
    # [] -> [batch, 20, 20, 256]
    layer = tf.contrib.layers.conv2d(X_train, num_outputs=256, kernel_size=9, stride=1, padding='VALID')


    # PrimaryCaps层
    with tf.variable_scope('PrimaryCaps_layer'):
      # 做了8次卷积神经网络，filter kernel为9*9, filter个数为32, strip为2, padding为0
      # [] -> []
      layer = tf.contrib.layers.conv2d(layer, 32 * 8, 9, 2, padding="VALID")
      # [] -> [batch, 1152, 8, 1]
      layer = tf.reshape(layer, (batch_size, -1, 8, 1))

    # [batch, 1152, 8, 1] -> [batch_size, 1152, 8, 1]
    layer = squash(layer)


    # DigitCaps层
    with tf.variable_scope('DigitCaps_layer'):

      # [] ->  [batch_size, 1152, 1, 8, 1]
      layer = tf.reshape(layer, shape=(batch_size, -1, 1, 1152, 1))

      layer = dynamic_routing(layer)

      # 将s_j投入 squeeze 函数以得出 DigitCaps 层的输出向量
      # -> [batch_size, 10, 16, 1]
      layer = tf.squeeze(layer, axis=1)

    return layer



def main():

  print("Hello")


if __name__ == "__main__":
  main()
