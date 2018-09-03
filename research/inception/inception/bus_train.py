"""A binary to train Inception on the bus data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

from inception import inception_train
from inception.bus_data import BusData

FLAGS = tf.app.flags.FLAGS


def main(_):
  dataset = BusData(subset=FLAGS.subset)
  assert dataset.data_files()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  inception_train.train(dataset)


if __name__ == '__main__':
  tf.app.run()
