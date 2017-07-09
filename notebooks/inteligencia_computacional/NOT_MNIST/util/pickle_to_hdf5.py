# ------------------------- IMPORTS ------------------------- #
import numpy as np
import math
from six.moves import cPickle as pickle
from six.moves import range
import os
import h5py
# ------------------------- LOAD DATA ------------------------- #
pickle_file = 'notMNIST.pickle'


with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels  = save['train_labels']
	valid_dataset = save['valid_dataset']
	valid_labels  = save['valid_labels']
	test_dataset  = save['test_dataset']
	test_labels   = save['test_labels']
	del save
	print "Training set", train_dataset.shape, train_labels.shape
	print "Valid set",    valid_dataset.shape, valid_labels.shape
	print "Test set",     test_dataset.shape,  test_labels.shape




image_size = 28


def reformat(dataset):
	dataset = dataset.reshape((-1,1,image_size,image_size)).astype(np.float32)
	return dataset


train_dataset = reformat(train_dataset)
valid_dataset = reformat(valid_dataset)
test_dataset  = reformat(test_dataset)

print "\nDATA AFTER RESHAPE"
print "Training set ", train_dataset.shape
print "Valid set ", valid_dataset.shape
print "Test set ", test_dataset.shape





with h5py.File(('train.h5'), 'w') as f:
		f['data'] = train_dataset
		f['label'] = train_labels

with h5py.File(('test.h5'), 'w') as f:
		f['data'] = test_dataset
		f['label'] = test_labels

with h5py.File(('valid.h5'), 'w') as f:
		f['data'] = valid_dataset
		f['label'] = valid_labels











