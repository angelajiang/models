import tensorflow as tf
from datasets import flowers
from os import sys

slim = tf.contrib.slim

# Selects the 'validation' dataset.
dataset = flowers.get_split('validation', sys.argv[1])

# Creates a TF-Slim DataProvider which reads the dataset in the background
# during both training and testing.
provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = provider.get(['image', 'label'])
