def load_mnist():
  dataset_directory = "./MNIST"

  file = open(os.path.join(dataset_directory, 'train-images-idx3-ubyte'))
  loaded = np.fromfile(file=file, dtype=np.uint8)
  X_train = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

  file = open(os.path.join(dataset_directory, 'train-labels-idx1-ubyte'))
  loaded = np.fromfile(file=file, dtype=np.uint8)
  Y_train = loaded[8:].reshape((60000)).astype(np.float)

  file = open(os.path.join(dataset_directory, 't10k-images-idx3-ubyte'))
  loaded = np.fromfile(file=file, dtype=np.uint8)
  X_test = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

  file = open(os.path.join(dataset_directory, 't10k-labels-idx1-ubyte'))
  loaded = np.fromfile(file=file, dtype=np.uint8)
  Y_test = loaded[8:].reshape((10000)).astype(np.float)

  X_train = tf.convert_to_tensor(X_train / 255., tf.float32)
  Y_train = tf.one_hot(Y_train, depth=10, axis=1, dtype=tf.float32)
  Y_test = tf.one_hot(Y_test, depth=10, axis=1, dtype=tf.float32)

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
