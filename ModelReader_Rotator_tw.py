import numpy as np
import random
import tensorflow as tf
import math
from libs.activations import lrelu
import tflearn
from sklearn.manifold import TSNE
from tsne import bh_sne
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import batch_norm
import scipy
import scipy.io as sio
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# % matplotlib inline
# To use, start by QT_API=pyqt ETS_TOOLKIT=qt4 jupyter notebook --no-browser --port=8889
import mayavi.mlab as mlab
mlab.options.offscreen = True
from imayavi import *
from tvtk.tools import visual
from PIL import Image
from tensorflow.python.framework import ops
import warnings

def read_and_decode_single_example_x0(filename, FLAGS):
	filename_queue = tf.train.string_input_producer([filename])
	options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
	reader = tf.TFRecordReader(options = options)
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		serialized_example,
		features={
			'x3d_0': tf.FixedLenFeature([27000], tf.int64),
			'id': tf.FixedLenFeature([1], tf.int64)
		})
	x3d_0 = features['x3d_0']
	idx = features['id']
	return (x3d_0, idx)

def read_and_decode_single_example_x(filename, FLAGS):
	# first construct a queue containing a list of filenames.
	# this lets a user split up there dataset in multiple files to keep
	# size down
	filename_queue = tf.train.string_input_producer([filename])
	# Unlike the TFRecordWriter, the TFRecordReader is symbolic
	options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
	reader = tf.TFRecordReader(options = options)
	# One can read a single serialized example from a filename
	# serialized_example is a Tensor of type string..
	_, serialized_example = reader.read(filename_queue)
	# The serialized example is converted back to actual values.
	# One needs to describe the format of the objects to be returned
	features = tf.parse_single_example(
		serialized_example,
		features={
			# We know the length of both fields. If not the
			# tf.VarLenFeature could be used
			'x3d': tf.FixedLenFeature([27000], tf.int64),
			'x3d_0': tf.FixedLenFeature([27000], tf.int64),
			'x2d_gnd': tf.FixedLenFeature([900], tf.int64),
			'x2d_rgb': tf.FixedLenFeature([154587], tf.string),
			'view': tf.FixedLenFeature([3], tf.float32), # [az, el, yaw]
			'y3d': tf.FixedLenFeature([1], tf.float32), # twist this time
			'axisangle': tf.FixedLenFeature([4], tf.float32), # [angle, axis]
			'id': tf.FixedLenFeature([1], tf.int64)
		})
	# features = tf.parse_single_example(
	# 	serialized_example,
	# 	dense_keys=['x3d', 'x3d', 'x2d_gnd', 'x2d_rgb', 'y3d'],
	# 	dense_types=[tf.int64, tf.int64, tf.int64, tf.string, tf.float32])
	# now return the converted data
	x3d = features['x3d']
	x3d_0 = features['x3d_0']
	x2d_gnd = features['x2d_gnd']
	x2d_rgb = tf.decode_raw(features['x2d_rgb'], tf.uint8)
	x2d_rgb = tf.reshape(x2d_rgb, [227, 227, 3])
	x2d_rgb.set_shape([227, 227, 3])
	y3d = features['y3d']
	view = features['view']
	angleaxis = features['axisangle']
	idx = features['id']
	return (x3d, x3d_0, x2d_gnd, x2d_rgb, view, y3d, angleaxis, idx)

class ModelReader_Rotator_tw(object):
	def __init__(self, flags):
		## All variables ##
		global FLAGS
		FLAGS = flags
		# print '==== FLAGS', FLAGS
		self.out_size = (30, 30, 30)
		self.is_training = tf.placeholder(dtype=bool,shape=[],name='is_training')

		self.batch_size_x0 = FLAGS.models_in_batch # models used in a batch
		x03d_gnd, idx = read_and_decode_single_example_x0(FLAGS.data_train_net0, FLAGS)
		self.x0_batch_flat, self.x0_batch_idx = tf.train.shuffle_batch(
			[x03d_gnd, idx], batch_size=self.batch_size_x0,
			num_threads=2,
			capacity=1500 + 2 * self.batch_size_x0,
			min_after_dequeue=1000,
			allow_smaller_final_batch=True)
		
		x03d_gnd_test, idx_test = read_and_decode_single_example_x0(FLAGS.data_test_net0, FLAGS)
		self.x0_batch_flat_test, self.x0_batch_test_idx = tf.train.shuffle_batch(
			[x03d_gnd_test, idx_test], batch_size=self.batch_size_x0,
			num_threads=2,
			capacity=500 + 2 * self.batch_size_x0,
			min_after_dequeue=500,
			allow_smaller_final_batch=True)
		self.x0_batch = tf.cond(self.is_training, \
			lambda: tf.to_float(tf.reshape(self.x0_batch_flat, [-1, 30, 30, 30, 1])), # models_in_batch
			lambda: tf.to_float(tf.reshape(self.x0_batch_flat_test, [-1, 30, 30, 30, 1])))
		self.x0_batch_idx = tf.cond(self.is_training, \
			lambda: tf.reshape(self.x0_batch_idx, [-1, 1]), # models_in_batch
			lambda: tf.reshape(self.x0_batch_test_idx, [-1, 1]))

		self.batch_size_x = FLAGS.batch_size # models used in a batch
		x3d, x3d_0, x2d_gnd, x2d_rgb, view, _, angleaxis, idx = read_and_decode_single_example_x(FLAGS.data_train_net1, FLAGS)
		self.x_batch_flat, self.x_gnd_batch_flat, self.x2d_gnd_batch_flat, self.x2d_rgb_batch_flat, \
		self.view_batch_flat, self.angleaxis_batch_flat, self.x_batch_idx = tf.train.shuffle_batch(
			[x3d, x3d_0, x2d_gnd, x2d_rgb, view, angleaxis, idx], batch_size=self.batch_size_x,
			num_threads=5,
			capacity=10000 + 5 * self.batch_size_x,
			min_after_dequeue=5000, 
			allow_smaller_final_batch=False)
		x3d_test, x3d_0_test, x2d_gnd_test, x2d_rgb_test, view_test, _, angleaxis_test, idx_test  = read_and_decode_single_example_x(FLAGS.data_test_net1, FLAGS)
		self.x_batch_flat_test, self.x_gnd_batch_flat_test, self.x2d_gnd_batch_flat_test, self.x2d_rgb_batch_flat_test, \
		self.view_batch_flat_test, self.angleaxis_batch_flat_test, self.x_batch_test_idx = tf.train.shuffle_batch(
			[x3d_test, x3d_0_test, x2d_gnd_test, x2d_rgb_test, view_test, angleaxis_test, idx_test], batch_size=self.batch_size_x,
			num_threads=5,
			capacity=3000 + 5 * self.batch_size_x,
			min_after_dequeue=3000,
			allow_smaller_final_batch=False)
		self.x_batch = tf.cond(self.is_training, \
			lambda: tf.to_float(tf.reshape(self.x_batch_flat, [-1, 30, 30, 30, 1])),
			lambda: tf.to_float(tf.reshape(self.x_batch_flat_test, [-1, 30, 30, 30, 1])))
		self.x_gnd_batch = tf.cond(self.is_training, \
			lambda: tf.to_float(tf.reshape(self.x_gnd_batch_flat, [-1, 30, 30, 30, 1])),
			lambda: tf.to_float(tf.reshape(self.x_gnd_batch_flat_test, [-1, 30, 30, 30, 1])))
		self.x2d_gnd_batch = tf.cond(self.is_training, \
			lambda: tf.to_float(tf.reshape(self.x2d_gnd_batch_flat, [-1, 30, 30])),
			lambda: tf.to_float(tf.reshape(self.x2d_gnd_batch_flat_test, [-1, 30, 30])))
		self.x2d_rgb_batch = tf.cond(self.is_training, \
			lambda: tf.to_float(tf.reshape(self.x2d_rgb_batch_flat, [-1, 227, 227, 3])),
			lambda: tf.to_float(tf.reshape(self.x2d_rgb_batch_flat_test, [-1, 227, 227, 3])))
		## REMBER TO SUB MEAN!
		self.view_batch = tf.cond(self.is_training, \
			lambda: tf.to_float(tf.reshape(self.view_batch_flat, [-1, 3])),
			lambda: tf.to_float(tf.reshape(self.view_batch_flat_test, [-1, 3])))
		self.angleaxis_batch = tf.cond(self.is_training, \
			lambda: tf.to_float(tf.reshape(self.angleaxis_batch_flat, [-1, 4])),
			lambda: tf.to_float(tf.reshape(self.angleaxis_batch_flat_test, [-1, 4])))
		self.y_batch = tf.mul(tf.slice(self.angleaxis_batch, [0, 0], [-1, 1]), tf.slice(self.angleaxis_batch, [0, 1], [-1, 3]))
		self.x_batch_idx = tf.cond(self.is_training, \
			lambda: tf.reshape(self.x_batch_idx, [-1, 1]), # models_in_batch
			lambda: tf.reshape(self.x_batch_test_idx, [-1, 1]))
		# self.az_el_batch = tf.slice(self.view_batch, [0, 0], [-1, 2])
		self.R_s_flat = self._az_el_yaw_s_degree_to_Rs_flat(self.view_batch)
		# self.R_s_flat = self._tws_to_Rs_flat(self.y_batch)

		# if FLAGS.if_indeptP == False:
		# 	self.p_s = self._az_el_s_degree_to_p_s(self.az_el_batch)
		# else:
		# 	self.p_s = self._az_el_s_degree_to_p_s_indept(self.az_el_batch)
		self.p_s = self.y_batch

		# self.dyn_channels = tf.shape(self.x0_batch)[-1]
		self.dyn_batch_size = FLAGS.batch_size

		# grid
		self.out_height = self.out_size[0]
		self.out_width = self.out_size[1]
		self.out_depth = self.out_size[2]
		grid = self._meshgrid(self.out_height, self.out_width, self.out_depth) #(4, height*width*depth)
		grid = tf.expand_dims(grid, 0) #(1, 4, height*width*depth)
		grid = tf.reshape(grid, [-1]) #(4*height*width*depth,)
		grid = tf.tile(grid, tf.pack([self.dyn_batch_size])) #(4*height*width*depth*num_batch,)
		grid = tf.reshape(grid, tf.pack([self.dyn_batch_size, 4, -1])) #(num_batch, 4, height*width*depth)
		self.grid = tf.cast(grid, 'float32')

		## Batch dup and rotation
		# self.idx_s = tf.placeholder(tf.int32, [None, 1])
		# self.Rs_flat_batch_gather_feed = tf.placeholder(tf.float32, [None, 16])
		# self.x0_batch_gather = tf.squeeze(tf.gather(self.x0_batch, self.idx_s), [1])
		# self.x0_batch_gather_trans = tf.clip_by_value(self._transform(self.x0_batch_gather, self.Rs_flat_batch_gather_feed, self.out_size), 0., 1.)

	# def _single_az_el_to_rotate_matrix_R_flat(self, az_el):
	# 	def _axisAngle2Mat(axis, angle):
	# 		c = tf.reshape(tf.cos(angle), [])
	# 		s = tf.reshape(tf.sin(angle), [])
	# 		t = 1. - c
	# 		# axis = tf.contrib.layers.unit_norm(axis, dim=0)
	# 		x = tf.reshape(tf.gather(axis, [0]), [])
	# 		y = tf.reshape(tf.gather(axis, [1]), [])
	# 		z = tf.reshape(tf.gather(axis, [2]), [])
	# 		# print '============== tf.concat(1, [t*x*x+c, t*x*y-z*s, t*x*z+y*s]).get_shape().as_list()', tf.concat(1, [t*x*x+c, t*x*y-z*s, t*x*z+y*s]).get_shape().as_list()
	# 		return tf.pack([[t*x*x+c, t*x*y-z*s, t*x*z+y*s], \
	# 			[t*x*y+z*s, t*y*y+c, t*y*z-x*s], \
	# 			[t*x*z-y*s, t*y*z+x*s, t*z*z+c]])
	# 	az_rad = tf.gather(az_el, [0])
	# 	el_rad = tf.gather(az_el, [1])
	# 	x_axis = tf.constant([[0.], [1.], [0.]])
	# 	# tw1 = tf.mul(x_axis, az_rad)
	# 	# R1 = tf.cast(self._single_expm(self._single_hat(tw1)), tf.float32)
	# 	R1 = _axisAngle2Mat(x_axis, az_rad)
	# 	# print '============== R1.get_shape().as_list()', x_axis.get_shape().as_list(), az_rad.get_shape().as_list(), R1.get_shape().as_list()
	# 	new_x_axis = tf.matmul(R1, tf.constant([[-1.], [0.], [0.]]))
	# 	# tw2 = tf.mul(new_x_azis, el_rad)
	# 	# R2 = tf.cast(self._single_expm(self._single_hat(tw2)), tf.float32)
	# 	R2 = _axisAngle2Mat(new_x_axis, el_rad)
	# 	R = tf.matmul(R2, R1)
	# 	R_flat = self._single_R_to_homo_flat(R)
	# 	# print '============== R.get_shape().as_list()', R.get_shape().as_list(), R_flat.get_shape().as_list()
	# 	return R_flat

	# Define custom py_func which takes also a grad op as argument:
	def _py_func(self, func, inp, Tout, stateful=True, name=None, grad=None):
		# Need to generate a unique name to avoid duplicates:
		rnd_name = 'custom_expm_grad' + str(np.random.randint(0, 1E+8))
		tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
		g = tf.get_default_graph()
		with g.gradient_override_map({"PyFunc": rnd_name}):
			return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
		
	def _custom_expm_impl(self, x):
		return scipy.linalg.expm(x)

	# Def custom square function using np.square instead of tf.square:
	def _myExpm(self, x, name=None):
		with ops.op_scope([x], name, "MyExpm") as name:
			expm_x = self._py_func(self._custom_expm_impl,
							[x],
							[tf.float32],
							name=name,
							grad=self._custom_expm_grad)  # <-- here's the call to the gradient
			return expm_x[0]

	def _custom_expm_grad_impl(self, A, gA):
		w, V = scipy.linalg.eig(A, right=True)
		U = scipy.linalg.inv(V).T
		exp_w = np.exp(w)
		X = np.subtract.outer(exp_w, exp_w) / np.subtract.outer(w, w)
		np.fill_diagonal(X, exp_w)
		Y = U.dot(V.T.dot(gA).dot(U) * X).dot(V.T)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", np.ComplexWarning)
			return Y.astype(A.dtype)

	# Actual gradient:
	def _custom_expm_grad(self, op, gA):
		A = op.inputs[0]
		return tf.py_func(self._custom_expm_grad_impl,[A, gA],[tf.float32])

	def _single_tw_to_rotate_matrix_R_flat(self, tw):
		def hat(tw):
			tw1 = tf.reshape(tf.slice(tw, [0], [1]), [])
			tw2 = tf.reshape(tf.slice(tw, [1], [1]), [])
			tw3 = tf.reshape(tf.slice(tw, [2], [1]), [])
			hat = tf.pack([[0., -tw3, tw2], \
				[tw3, 0., -tw1], \
				[-tw2, tw1, 0.]])
			return hat
		return self._single_R_to_homo_flat(tf.cast(self._myExpm(hat(tw)), tf.float32))

	def _single_az_el_yaw_to_rotate_matrix_R_flat(self, az_el_yaw):
		def _axisAngle2Mat(axis, angle):
			c = tf.reshape(tf.cos(angle), [])
			s = tf.reshape(tf.sin(angle), [])
			t = 1. - c
			# axis = tf.contrib.layers.unit_norm(axis, dim=0)
			x = tf.reshape(tf.gather(axis, [0]), [])
			y = tf.reshape(tf.gather(axis, [1]), [])
			z = tf.reshape(tf.gather(axis, [2]), [])
			# print '============== tf.concat(1, [t*x*x+c, t*x*y-z*s, t*x*z+y*s]).get_shape().as_list()', tf.concat(1, [t*x*x+c, t*x*y-z*s, t*x*z+y*s]).get_shape().as_list()
			return tf.pack([[t*x*x+c, t*x*y-z*s, t*x*z+y*s], \
				[t*x*y+z*s, t*y*y+c, t*y*z-x*s], \
				[t*x*z-y*s, t*y*z+x*s, t*z*z+c]])
		az_rad = tf.gather(az_el_yaw, [0])
		# az_rad = tf.zeros_like(az_rad) + 90. / 180. * np.pi
		el_rad = tf.gather(az_el_yaw, [1])
		yaw_rad = tf.gather(az_el_yaw, [2])
		# el_rad = tf.zeros_like(el_rad) 
		x_axis = tf.constant([[0.], [1.], [0.]])
		# tw1 = tf.mul(x_axis, az_rad)
		# R1 = tf.cast(self._single_expm(self._single_hat(tw1)), tf.float32)
		R1 = _axisAngle2Mat(x_axis, az_rad)
		# print '============== R1.get_shape().as_list()', x_axis.get_shape().as_list(), az_rad.get_shape().as_list(), R1.get_shape().as_list()
		new_x_axis = tf.matmul(R1, tf.constant([[-1.], [0.], [0.]]))
		# tw2 = tf.mul(new_x_azis, el_rad)
		# R2 = tf.cast(self._single_expm(self._single_hat(tw2)), tf.float32)
		R2 = _axisAngle2Mat(new_x_axis, el_rad)
		new_x_axis3 = tf.matmul(tf.matmul(R2, R1), tf.constant([[0.], [0.], [-1.]]))
		R3 = _axisAngle2Mat(new_x_axis3, -yaw_rad)
		R = tf.matmul(R3, tf.matmul(R2, R1))
		R_flat = self._single_R_to_homo_flat(R)
		# print '============== R.get_shape().as_list()', R.get_shape().as_list(), R_flat.get_shape().as_list()
		return R_flat

	# def _p_s_to_az_el(self, p_s):
	# 	p_s0, p_s1, p_s2 = tf.split(1, 3, p_s)
	# 	sin_el = p_s2
	# 	el = tf.asin(sin_el)
	# 	def atan2(y, x):
	# 		angle = tf.zeros_like(x)
	# 		angle = tf.select(tf.logical_and(tf.greater(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x), angle)
	# 		angle = tf.select(tf.logical_and(tf.greater(x,0.0),  tf.less_equal(y,0.0)), tf.atan(y/x) + np.pi*2, angle)
	# 		angle = tf.select(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
	# 		angle = tf.select(tf.logical_and(tf.less(x,0.0),  tf.less_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
	# 		angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
	# 		angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), 1.5*np.pi * tf.ones_like(x), angle)
	# 		angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), np.nan * tf.zeros_like(x), angle)
	# 		return angle
	# 	if FLAGS.if_indeptP == False:
	# 		az = atan2(tf.div(p_s0, tf.cos(el)), tf.div(p_s1, tf.cos(el)))
	# 	else:
	# 		az = atan2(p_s1, p_s0)
	# 	return tf.concat(1, [tf.reshape(az, [-1, 1]), tf.reshape(el, [-1, 1])])

	def _single_R_to_homo_flat(self, R):
		R_cat = tf.concat(1, [R, tf.cast(tf.zeros([3, 1]), tf.float32)])
		R_cat = tf.concat(0, [R_cat, tf.constant([0., 0., 0., 1.], shape=[1, 4], dtype=tf.float32)])
		R_flat = tf.reshape(R_cat, [1, -1]) #(16,)
		return R_flat

	# def _p_s_to_Rs_flat(self, p_s):
	# 	az_el_s = self._p_s_to_az_el(p_s)
	# 	return tf.reshape(tf.map_fn(lambda x: self._single_az_el_to_rotate_matrix_R_flat(x), az_el_s), [-1, 16]), az_el_s

	def _az_el_yaw_s_degree_to_Rs_flat(self, az_el_s_degree):
		az_el_s = az_el_s_degree * np.pi / 180.
		return tf.reshape(tf.map_fn(lambda x: self._single_az_el_yaw_to_rotate_matrix_R_flat(x), az_el_s), [-1, 16])

	def _tws_to_Rs_flat(self, tws):
		return tf.reshape(tf.map_fn(lambda x: self._single_tw_to_rotate_matrix_R_flat(x), tws), [-1, 16])

	# def _az_el_s_degree_to_p_s(self, az_el_s_degree):
	# 	az_el_s = az_el_s_degree * np.pi / 180.
	# 	az_col = tf.slice(az_el_s, [0, 0], [-1, 1])
	# 	el_col = tf.slice(az_el_s, [0, 1], [-1, 1])
	# 	sin_az = tf.sin(az_col)
	# 	cos_az = tf.cos(az_col)
	# 	sin_el = tf.sin(el_col)
	# 	cos_el = tf.cos(el_col)
	# 	return tf.concat(1, [tf.mul(cos_el, sin_az), tf.mul(cos_el, cos_az), sin_el])

	# def _az_el_s_degree_to_p_s_indept(self, az_el_s_degree):
	# 	az_el_s = az_el_s_degree * np.pi / 180.
	# 	az_col = tf.slice(az_el_s, [0, 0], [-1, 1])
	# 	el_col = tf.slice(az_el_s, [0, 1], [-1, 1])
	# 	sin_az = tf.sin(az_col)
	# 	cos_az = tf.cos(az_col)
	# 	sin_el = tf.sin(el_col)
	# 	cos_el = tf.cos(el_col)
	# 	return tf.concat(1, [cos_az, sin_az, sin_el])

	def _interpolate(self, im, x, y, z, out_size):
		# im: tensor of inputs [num_batch, height, width, depth, num_channels=1]
		# x, y, z: coords of new grid [num_batch, height,width, depth,]
		# out_size : int; the size of the output [out_height, out_width, out_depth]
		# constants
		# num_batch = tf.shape(im)[0]
		height = tf.shape(im)[1]
		width = tf.shape(im)[2]
		depth = tf.shape(im)[3]
		channels = tf.shape(im)[4]

		x = tf.cast(x, 'float32')
		y = tf.cast(y, 'float32')
		z = tf.cast(z, 'float32')
		height_f = tf.cast(height, 'float32')
		width_f = tf.cast(width, 'float32')
		depth_f = tf.cast(depth, 'float32')
		out_height = out_size[0]
		out_width = out_size[1]
		out_depth = out_size[2]
		zero = tf.zeros([], dtype='int32')
		max_z = tf.cast(tf.shape(im)[1] - 1, 'int32') # TODO: the axis
		max_y = tf.cast(tf.shape(im)[2] - 1, 'int32')
		max_x = tf.cast(tf.shape(im)[3] - 1, 'int32')

		# scale indices from [-1, 1] to [0, width/height]
		x = (x + 1.0)*(width_f-1.) / 2.0 # dim 1: x, width
		y = (y + 1.0)*(height_f-1.) / 2.0 # dim 0: y height
		z = (z + 1.0)*(depth_f-1.) / 2.0 # dim 2: z:depth

		# do sampling
		x0 = tf.cast(tf.floor(x), 'int32')
		x1 = x0 + 1
		y0 = tf.cast(tf.floor(y), 'int32')
		y1 = y0 + 1
		z0 = tf.cast(tf.floor(z), 'int32')
		z1 = z0 + 1

		ImIdx = tf.tile(tf.reshape(tf.range(self.dyn_batch_size), [-1, 1, 1, 1]), [1, width, height, depth]) # [batchSize, W, H, D]
		ImVecBatch = tf.reshape(im, [-1, channels]) # [batchSize*W*H*D, C]
		ImVecBatchOutside = tf.concat(0, [ImVecBatch, tf.zeros([1, channels])]) # [batchSize*H*W+1, C]

		y0x0z0 = ((ImIdx * height + y0) * width + x0) * depth + z0 # [batchSize, W, H, D]
		y0x0z1 = ((ImIdx * height + y0) * width + x0) * depth + z1 # [batchSize, W, H, D]
		y0x1z0 = ((ImIdx * height + y0) * width + x1) * depth + z0 # [batchSize, W, H, D]
		y0x1z1 = ((ImIdx * height + y0) * width + x1) * depth + z1 # [batchSize, W, H, D]
		y1x0z0 = ((ImIdx * height + y1) * width + x0) * depth + z0 # [batchSize, W, H, D]
		y1x0z1 = ((ImIdx * height + y1) * width + x0) * depth + z1 # [batchSize, W, H, D]
		y1x1z0 = ((ImIdx * height + y1) * width + x1) * depth + z0 # [batchSize, W, H, D]
		y1x1z1 = ((ImIdx * height + y1) * width + x1) * depth + z1 # [batchSize, W, H, D]

		yxz_out = tf.fill([self.dyn_batch_size, width, height, depth], self.dyn_batch_size * height * width * depth)

		def insideIm(Yint, Xint, Zint):
			return (Xint>=0)&(Xint<width)&(Yint>=0)&(Yint<height)&(Zint>=0)&(Zint<depth)
		y0x0z0 = tf.select(insideIm(y0, x0, z0), y0x0z0, yxz_out)
		y0x0z1 = tf.select(insideIm(y0, x0, z1), y0x0z1, yxz_out)
		y0x1z0 = tf.select(insideIm(y0, x1, z0), y0x1z0, yxz_out)
		y0x1z1 = tf.select(insideIm(y0, x1, z1), y0x1z1, yxz_out)
		y1x0z0 = tf.select(insideIm(y1, x0, z0), y1x0z0, yxz_out)
		y1x0z1 = tf.select(insideIm(y1, x0, z1), y1x0z1, yxz_out)
		y1x1z0 = tf.select(insideIm(y1, x1, z0), y1x1z0, yxz_out)
		y1x1z1 = tf.select(insideIm(y1, x1, z1), y1x1z1, yxz_out)

		I1 = tf.to_float(tf.gather(ImVecBatchOutside, y0x0z0))
		I2 = tf.to_float(tf.gather(ImVecBatchOutside, y0x0z1))
		I3 = tf.to_float(tf.gather(ImVecBatchOutside, y0x1z0))
		I4 = tf.to_float(tf.gather(ImVecBatchOutside, y0x1z1))
		I5 = tf.to_float(tf.gather(ImVecBatchOutside, y1x0z0))
		I6 = tf.to_float(tf.gather(ImVecBatchOutside, y1x0z1))
		I7 = tf.to_float(tf.gather(ImVecBatchOutside, y1x1z0))
		I8 = tf.to_float(tf.gather(ImVecBatchOutside, y1x1z1))

		# and finally calculate interpolated values
		x0_f = tf.cast(x0, 'float32')
		x1_f = tf.cast(x1, 'float32')
		y0_f = tf.cast(y0, 'float32')
		y1_f = tf.cast(y1, 'float32')
		z0_f = tf.cast(z0, 'float32')
		z1_f = tf.cast(z1, 'float32')
		w1 = tf.expand_dims((x1_f-x) * (y1_f-y) * (z1_f-z), -1)
		w2 = tf.expand_dims((x1_f-x) * (y1_f-y) * (z-z0_f), -1)
		w3 = tf.expand_dims((x-x0_f) * (y1_f-y) * (z1_f-z), -1)
		w4 = tf.expand_dims((x-x0_f) * (y1_f-y) * (z-z0_f), -1)
		w5 = tf.expand_dims((x1_f-x) * (y-y0_f) * (z1_f-z), -1)
		w6 = tf.expand_dims((x1_f-x) * (y-y0_f) * (z-z0_f), -1)
		w7 = tf.expand_dims((x-x0_f) * (y-y0_f) * (z1_f-z), -1)
		w8 = tf.expand_dims((x-x0_f) * (y-y0_f) * (z-z0_f), -1)

		output = tf.add_n([w1*I1, w2*I2, w3*I3, w4*I4, w5*I5, w6*I6, w7*I7, w8*I8])
		# output = I8 - I1
		return output

	def _meshgrid(self, height, width, depth):
		x_t, y_t, z_t = np.meshgrid(np.linspace(-1, 1, width),
							   np.linspace(-1, 1, height),
							   np.linspace(-1, 1, depth))
		ones = np.ones(np.prod(x_t.shape))
		grid = np.vstack([x_t.flatten(), y_t.flatten(), z_t.flatten(), ones]).astype('float32') #(4, height*width*depth)
		return grid

	def _transform(self, input_dim, theta, out_size):
		height = tf.shape(input_dim)[1]
		width = tf.shape(input_dim)[2]
		depth = tf.shape(input_dim)[3]
		num_channels = tf.shape(input_dim)[4]
		theta = tf.reshape(theta, [-1, 4, 4])
		self.theta = tf.cast(theta, 'float32')

		# Transform A x (x_t, y_t, z_t, 1)^T -> (x_s, y_s, z_s)
		self.T_g = tf.batch_matmul(self.theta, self.grid) #(num_batch, 4, 4) * (num_batch, 4, height*width*depth) => (num_batch, 4, out_height*_out_width*depth)
		x_s, y_s, z_s, w_s = tf.split(1, 4, self.T_g)
		x_s_flat = tf.reshape(x_s/w_s, [-1, width, height, depth]) # (num_batch, 1*height*width*depth, )
		y_s_flat = tf.reshape(y_s/w_s, [-1, width, height, depth])
		z_s_flat = tf.reshape(z_s/w_s, [-1, width, height, depth])
		# we have got the 3d grid transformed

		input_transformed = self._interpolate(
			input_dim, x_s_flat, y_s_flat, z_s_flat,
			out_size)
		output = tf.reshape(
			input_transformed, tf.pack([self.dyn_batch_size, self.out_height, self.out_width, self.out_depth, num_channels]))
		return output
