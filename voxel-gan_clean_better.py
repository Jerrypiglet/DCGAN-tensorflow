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
from ModelReader_Rotator_tw import ModelReader_Rotator_tw
# from show import *
from tvtk.tools import visual
from PIL import Image
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters

np.random.seed(0)
tf.set_random_seed(0)
flags = tf.flags
flags.DEFINE_integer("resume_from", -1, "resume from")
flags.DEFINE_boolean("train_net", True, "train net: True for training net 0, false for training net 1")
flags.DEFINE_boolean("restore_encoder0", False, "training net 1 and restoring net 0 from saved points")
flags.DEFINE_integer("n_z", 100, "hidden size")
flags.DEFINE_integer("batch_size", 200, "batch_size")
flags.DEFINE_integer("models_in_batch", 40, "models in a batch")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
# flags.DEFINE_float("reweight_recon", 1.0, "weight for recon loss")
# flags.DEFINE_float("reweight_reproj", 0.0, "weight for reproj loss")
# flags.DEFINE_float("reweight_gan", 1.0, "weight for gan loss")
# flags.DEFINE_float("reweight_euc_s", 1.0, "weigth for euc loss of style")
# flags.DEFINE_float("reweight_euc_p", 1.0, "weigth for euc loss of pose")
flags.DEFINE_string("folder_name_restore_from", "", "name of restore folder")
flags.DEFINE_string("folder_name_net0_restore_from", "", "name of restore folder")
flags.DEFINE_string("folder_name_save_to", "", "name of save folder")
flags.DEFINE_boolean("if_disp", True, "if display s and p")
flags.DEFINE_integer("disp_every_step", 1, "disp every ? step")
flags.DEFINE_integer("training_epochs", 5000, "total training epochs")
flags.DEFINE_boolean("if_summary", True, "if save summary")
flags.DEFINE_boolean("if_draw", True, "if draw latent")
flags.DEFINE_integer("draw_every", 5, "draw every ? epoch")
flags.DEFINE_boolean("if_save", True, "if save")
flags.DEFINE_integer("save_every_step", 1, "save every ? step")
flags.DEFINE_boolean("if_test", False, "if test")
flags.DEFINE_integer("test_every_step", 10, "test every ? step")
flags.DEFINE_string("data_train_net0", "", "load train data from")
flags.DEFINE_string("data_test_net0", "", "load test data from")
flags.DEFINE_string("data_train_net1", "", "load train data from")
flags.DEFINE_string("data_test_net1", "", "load test data from")
flags.DEFINE_boolean("if_BN", False, "if batch_norm")
flags.DEFINE_boolean("if_BN_out", False, "if batch_norm for x output layer")
flags.DEFINE_boolean("if_show", True, "if show mode")
flags.DEFINE_boolean("if_unlock_decoder0", False, "if unlock decoder0")
# flags.DEFINE_boolean("if_gndS", False, "if_gndS")
# flags.DEFINE_boolean("if_gndP", True, "if_gndP")
# flags.DEFINE_boolean("if_p_trainable", True, "if recog p is trainable")
# flags.DEFINE_boolean("if_s_trainable", True, "if recog s is trainable")
global FLAGS
FLAGS = flags.FLAGS

def weight_variable(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.02))

def bias_variable(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

def conv3d(x, W, stride=2):
	return tf.nn.conv3d(x, W, strides=[1, stride, stride, stride, 1], padding='SAME')

def deconv3d(x, W, output_shape, stride=2):
	return tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, stride, stride, stride, 1], padding='SAME')

def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

class BatchNormalization(object):

	def __init__(self, shape, name, decay=0.9, epsilon=1e-5):
		with tf.variable_scope(name):
			self.beta = tf.Variable(tf.constant(0.0, shape=shape), name="beta") # offset
			self.gamma = tf.Variable(tf.constant(1.0, shape=shape), name="gamma") # scale
			self.ema = tf.train.ExponentialMovingAverage(decay=decay)
			self.epsilon = epsilon

	def __call__(self, x, train):
		self.train = train
		n_axes = len(x.get_shape()) - 1
		batch_mean, batch_var = tf.nn.moments(x, range(n_axes))
		mean, variance = self.ema_mean_variance(batch_mean, batch_var)
		return tf.nn.batch_normalization(x, mean, variance, self.beta, self.gamma, self.epsilon)

	def ema_mean_variance(self, mean, variance):
		def with_update():
			ema_apply = self.ema.apply([mean, variance])
			with tf.control_dependencies([ema_apply]):
				return tf.identity(mean), tf.identity(variance)
		return tf.cond(self.train, with_update, lambda: (self.ema.average(mean), self.ema.average(variance)))

# code from https://github.com/openai/improved-gan
class VirtualBatchNormalization(object):

	def __init__(self, x, name, epsilon=1e-5, half=None):
		"""
		x is the reference batch
		"""
		assert isinstance(epsilon, float)

		self.half = half
		shape = x.get_shape().as_list()
		needs_reshape = len(shape) != 4

		if needs_reshape:
			orig_shape = shape
			if len(shape) == 5:
				x = tf.reshape(x, [shape[0], 1, shape[1]*shape[2]*shape[3], shape[4]])
			elif len(shape) == 2:
				x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
			elif len(shape) == 1:
				x = tf.reshape(x, [shape[0], 1, 1, 1])
			else:
				assert False, shape
			shape = x.get_shape().as_list()

		with tf.variable_scope(name) as scope:
			assert name.startswith("d_") or name.startswith("g_")
			self.epsilon = epsilon
			self.name = name
			if self.half is None:
				half = x
			elif self.half == 1:
				half = tf.slice(x, [0, 0, 0, 0], [shape[0] // 2, shape[1], shape[2], shape[3]])
			elif self.half == 2:
				half = tf.slice(x, [shape[0] // 2, 0, 0, 0], [shape[0] // 2, shape[1], shape[2], shape[3]])
			else:
				assert False
			self.mean = tf.reduce_mean(half, [0, 1, 2], keep_dims=True)
			self.mean_sq = tf.reduce_mean(tf.square(half), [0, 1, 2], keep_dims=True)
			self.batch_size = int(half.get_shape()[0])
			assert x is not None
			assert self.mean is not None
			assert self.mean_sq is not None
			out = self._normalize(x, self.mean, self.mean_sq, "reference")
			if needs_reshape:
				out = tf.reshape(out, orig_shape)
			self.reference_output = out

	def __call__(self, x):
		shape = x.get_shape().as_list()
		needs_reshape = len(shape) != 4

		if needs_reshape:
			orig_shape = shape
			if len(shape) == 5:
				x = tf.reshape(x, [shape[0], 1, shape[1]*shape[2]*shape[3], shape[4]])
			elif len(shape) == 2:
				x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
			elif len(shape) == 1:
				x = tf.reshape(x, [shape[0], 1, 1, 1])
			else:
				assert False, shape
			shape = x.get_shape().as_list()

		with tf.variable_scope(self.name, reuse=True) as scope:
			new_coeff = 1. / (self.batch_size + 1.)
			old_coeff = 1. - new_coeff
			new_mean = tf.reduce_mean(x, [0, 1, 2], keep_dims=True)
			new_mean_sq = tf.reduce_mean(tf.square(x), [0, 1, 2], keep_dims=True)
			mean = new_coeff * new_mean + old_coeff * self.mean
			mean_sq = new_coeff * new_mean_sq + old_coeff * self.mean_sq
			out = self._normalize(x, mean, mean_sq, "live")
			if needs_reshape:
				out = tf.reshape(out, orig_shape)
			return out

	def _normalize(self, x, mean, mean_sq, message):
		# make sure this is called with a variable scope
		shape = x.get_shape().as_list()
		assert len(shape) == 4
		self.gamma = tf.get_variable("gamma", [shape[-1]], initializer=tf.random_normal_initializer(1., 0.02))
		self.beta = tf.get_variable("beta", [shape[-1]], initializer=tf.constant_initializer(0.))
		gamma = tf.reshape(self.gamma, [1, 1, 1, -1])
		beta = tf.reshape(self.beta, [1, 1, 1, -1])
		assert self.epsilon is not None
		assert mean_sq is not None
		assert mean is not None
		std = tf.sqrt(self.epsilon + mean_sq - tf.square(mean))
		out = x - mean
		out = out / std
		# out = tf.Print(out, [tf.reduce_mean(out, [0, 1, 2]),
		#    tf.reduce_mean(tf.square(out - tf.reduce_mean(out, [0, 1, 2], keep_dims=True)), [0, 1, 2])],
		#    message, first_n=-1)
		out = out * gamma
		out = out + beta
		return out

def vbn(x, name):
	f = VirtualBatchNormalization(x, name)
	return f(x)

class VariationalAutoencoder(object):
	def __init__(self, params, transfer_fct=tf.nn.sigmoid):
		self.params = params
		self.transfer_fct = transfer_fct
		self.transfer_fct_conv = lrelu
		# self.batch_normalization = batch_normalization
		self.learning_rate = params['learning_rate']
		self.batch_size = self.params['batch_size']
		self.z_size = self.params["n_z"]
		self.x_size = self.params["n_input"]

		if FLAGS.train_net:
			summary_folder = params['summary_folder'] + '/net0'
		# else:
		# 	summary_folder = params['summary_folder'] + '/net1'
		self.train_writer = tf.train.SummaryWriter(summary_folder)
		self.global_step = tf.Variable(-1, name='global_step', trainable=False)

		# Create autoencoder network
		print '-----> Defining network...'
		self._create_network()
		# Define loss function based variational upper-bound and corresponding optimizer
		print '-----> Defining loss...'
		self._create_loss_optimizer()
		print '---------- Defining loss done.'

		self.restorer_encoder0 = tf.train.Saver(slim.get_variables_to_restore(), max_to_keep=2, write_version=tf.train.SaverDef.V2)
		self.saver_for_resume = tf.train.Saver(slim.get_variables_to_restore(exclude=["is_training"]), max_to_keep=50, write_version=tf.train.SaverDef.V2)
		self.restorer = tf.train.Saver(slim.get_variables_to_restore(exclude=["is_training"]), max_to_keep=50, write_version=tf.train.SaverDef.V2)

		config = tf.ConfigProto(
			# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4),
			# device_count = {'GPU': 0}
			)
		config.gpu_options.allow_growth = True
		# config.log_device_placement = True
		config.allow_soft_placement = True
		self.sess = tf.Session(config=config)
		init_op = tf.group(tf.initialize_all_variables(),
					   tf.initialize_local_variables())
		self.sess.run(init_op)
		# # Start input enqueue threads.
		self.coord = tf.train.Coordinator()
		self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

		if not os.path.exists(params['summary_folder']):
			os.makedirs(params['summary_folder'])
		if not os.path.exists(params['summary_folder']+'/net0'):
			os.makedirs(params['summary_folder']+'/net0')
		if not os.path.exists(params['summary_folder']+'/net1'):
			os.makedirs(params['summary_folder']+'/net1')
			print("+++++ Created smummary folder path: %s" % params['summary_folder'])

		if not os.path.exists(params['model_folder']):
			os.makedirs(params['model_folder'])
		if not os.path.exists(params['model_folder']+'/net0'):
			os.makedirs(params['model_folder']+'/net0')
		if not os.path.exists(params['model_folder']+'/net1'):
			os.makedirs(params['model_folder']+'/net1')
			print("+++++ Created snapshot folder path: %s" % params['model_folder'])
		print '-----> Initializing transformer finished'

	def _create_network(self):
		with tf.device('/gpu:0'):
			self.is_training = tf.placeholder(dtype=bool,shape=[],name='is_training')
			self.train_net = tf.placeholder(dtype=bool,shape=[],name='train_net')
			self.is_queue = tf.placeholder(dtype=bool,shape=[],name='is_queue')
			with tf.variable_scope("step"):
				self.global_step = tf.cond(self.is_training,
					lambda: tf.assign_add(self.global_step, 1), lambda: tf.assign_add(self.global_step, 0))
			self.gen = ModelReader_Rotator_tw(FLAGS)

		with tf.device('/gpu:0'):
			## Define net 0
			# self.x0 = tf.cond(self.train_net, lambda: self.gen.x0_batch, lambda: self.gen.x_gnd_batch) # aligned models
			self.x0 = self.gen.x0_batch # aligned models
			self.x0 = tf.transpose(self.x0, perm=[0, 2, 3, 1, 4])
			self.dyn_batch_size_x0 = tf.shape(self.x0)[0]

		with tf.device('/gpu:0'):
			self.z = tf.placeholder(tf.float32, [None, self.z_size], name='z')
			self.z_sum = tf.histogram_summary("z", self.z)
			self.y = self._discriminator(self.x0)
			self.x_, self.x_nogate = self._generator(self.z)
			self.G_sum = tf.histogram_summary("G", tf.reshape(self.x_nogate, [-1]))

		with tf.device('/gpu:0'):
			self.y_ = self._discriminator(self.x_, reuse=True)

	def _create_loss_optimizer(self):
		# self.d_loss_real = tf.reduce_mean(
		# 	tf.nn.sigmoid_cross_entropy_with_logits(
		# 		logits=self.D_logits, targets=tf.ones_like(self.D)))
		# self.d_loss_fake = tf.reduce_mean(
		# 	tf.nn.sigmoid_cross_entropy_with_logits(
		# 		logits=self.D_logits_, targets=tf.zeros_like(self.D_)))
		# # self.d_loss_real = tf.reduce_mean(
		# # 	tf.nn.sparse_softmax_cross_entropy_with_logits(
		# # 		logits=self.D_logits, labels=tf.ones([self.batch_size], dtype=tf.int64)))
		# # self.d_loss_fake = tf.reduce_mean(
		# # 	tf.nn.sparse_softmax_cross_entropy_with_logits(
		# # 		logits=self.D_logits_, labels=tf.zeros([self.batch_size], dtype=tf.int64)))

		self.accu_real = tf.reduce_mean(self.y)
		self.accu_fake = 1. - tf.reduce_mean(self.y_)

		# self.g_loss = tf.reduce_mean(
		# 	tf.nn.sigmoid_cross_entropy_with_logits(
		# 		logits=self.D_logits_, targets=tf.ones_like(self.D_)))
		# # self.g_loss = tf.reduce_mean(
		# # 	tf.nn.sparse_softmax_cross_entropy_with_logits(
		# 		logits=self.D_logits_, labels=tf.ones([self.batch_size], dtype=tf.int64)))

		label_real = np.zeros([batch_size, 2], dtype=np.float32)
		label_fake = np.zeros([batch_size, 2], dtype=np.float32)
		label_real[:, 0] = 1
		label_fake[:, 1] = 1

		self.g_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, tf.constant(label_real)))
		self.d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, tf.constant(label_fake)))
		self.d_loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, tf.constant(label_real)))

		# self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
		# self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)
													
		# self.d_loss = self.d_loss_real + self.d_loss_fake

		self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
		self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

		t_vars = tf.trainable_variables()
		# print t_vars
		# print slim.get_variables_to_restore(include=["discriminator", "generator"])
		self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
		# print [var.name for var in t_vars if 'discriminator' in var.name]
		self.g_vars = [var for var in t_vars if 'generator' in var.name]

		# slim.get_variables_to_restore(include=["encoder0","decoder0","step"])

		self.d_optim = tf.train.AdamOptimizer(1e-4, beta1=FLAGS.beta1) \
							.minimize(self.d_loss, var_list=self.d_vars)
		self.g_optim = tf.train.AdamOptimizer(1e-4, beta1=FLAGS.beta1) \
							.minimize(self.g_loss, var_list=self.g_vars)

		self.g_sum = tf.merge_summary([self.z_sum, self.G_sum, self.g_loss_sum])
		self.d_sum = tf.merge_summary(
				[self.z_sum, self.G_sum, self.d_loss_real_sum, self.d_loss_sum])

	# def BatchNorm(self, inputT, trainable, scope=None):
	# 	if trainable:
	# 		print '########### BN trainable!!!'
	# 	return tflearn.layers.normalization.batch_normalization(inputT, trainable=trainable)

	def _discriminator(self, input_tensor, trainable=True, reuse=False):
		with tf.variable_scope("discriminator") as scope:
			if reuse:
				scope.reuse_variables()

			with slim.arg_scope([slim.fully_connected], trainable=FLAGS.train_net):
				input_shape=[None, 27000]

				# x_tensor = tf.reshape(input_tensor, [-1, 30, 30, 30, 1])
				noisy_x = input_tensor + tf.random_normal(tf.pack([dyn_batch_size, 32, 32, 32, 1]))
				current_input = noisy_x

				def conv_layer(current_input, kernel_shape, strides, scope, transfer_fct, is_training, if_batch_norm, padding, trainable):
					# kernel = tf.Variable(
					# 	tf.random_uniform(kernel_shape, -1.0 / (math.sqrt(kernel_shape[3]) + 20), 1.0 / (math.sqrt(kernel_shape[3]) + 20)), 
					# 	trainable=trainable)
					# biases = tf.Variable(tf.zeros(shape=[kernel_shape[-1]], dtype=tf.float32), trainable=trainable)
					kernel = weight_variable(kernel_shape)
					biases = bias_variable([kernel_shape[-2]])

					if if_batch_norm:
						bn_func = BatchNormalization([kernel_shape[4]], scope)
						current_output = transfer_fct(
							bn_func(
								tf.add(conv3d(current_input, kernel), biases),
								)
							)
					else:
						current_output = transfer_fct(
								tf.add(conv3d(current_input, kernel), biases),
							)
					return current_output
				def transfer_fct_none(x):
					return x

				current_input = conv_layer(current_input, [4, 4, 4, 1, 32], [1, 2, 2, 2, 1], 'BN-0', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="SAME", trainable=trainable)
				print current_input.get_shape().as_list()
				current_input = conv_layer(current_input, [4, 4, 4, 32, 64], [1, 2, 2, 2, 1], 'BN-1', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="SAME", trainable=trainable)
				print current_input.get_shape().as_list()
				current_input = conv_layer(current_input, [4, 4, 4, 64, 128], [1, 2, 2, 2, 1], 'BN-2', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="SAME", trainable=trainable)
				print current_input.get_shape().as_list()
				current_input = conv_layer(current_input, [4, 4, 4, 128, 256], [1, 2, 2, 2, 1], 'BN-3', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="SAME", trainable=trainable)
				print current_input.get_shape().as_list()
				# current_input = conv_layer(current_input, [4, 4, 4, 512, 1], [1, 1, 1, 1, 1], 'BN-4', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="SAME", trainable=trainable)
				# print current_input.get_shape().as_list()

				self.before_flatten_shape = current_input.get_shape().as_list()
				self.flatten_shape = tf.pack([-1, np.prod(current_input.get_shape().as_list() [1:])])
				h = tf.reshape(current_input, self.flatten_shape)
				self.flatten_length = h.get_shape().as_list()[1]

				print '---------- _>>> discriminator: flatten length:', self.flatten_length

				self.n_kernels = 300
				self.dim_per_kernel = 50
				kernel_h5 = weight_variable([2*2*2*256+self.n_kernels, 2])
				kernel_md = weight_variable([2*2*2*256, self.n_kernels*self.dim_per_kernel])
				bias_h5 = bias_variable([2]),
				bias_md = bias_variable([self.n_kernels])

				m = tf.matmul(h, kernel_md)
				m = tf.reshape(m, [-1, self.n_kernels, self.dim_per_kernel])
				abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(m, 3) - tf.expand_dims(tf.transpose(m, [1, 2, 0]), 0)), 2)
				f = tf.reduce_sum(tf.exp(-abs_dif), 2) + bias_md

				h = tf.concat(1, [h, f])
		        y = tf.matmul(h, kernel_h5) + bias_h5
		        return y

				# hidden_tensor = tf.contrib.layers.fully_connected(flattened, self.flatten_length//2, activation_fn=self.transfer_fct_conv, trainable=trainable)
				# hidden_tensor = tf.contrib.layers.fully_connected(hidden_tensor, self.flatten_length//4, activation_fn=self.transfer_fct_conv, trainable=trainable)
				# hidden_tensor = tf.contrib.layers.fully_connected(hidden_tensor, 1, activation_fn=None, trainable=trainable)
				# return (tf.nn.sigmoid(hidden_tensor), hidden_tensor)

	def _generator(self, input_sample, trainable=True):
		with tf.variable_scope("generator") as scope:
			dyn_batch_size = tf.shape(input_sample)[0]
			hidden_tensor_inv = vbn(tf.contrib.layers.fully_connected(input_sample, self.flatten_length//4, activation_fn=self.transfer_fct_conv, trainable=trainable))
			# hidden_tensor_inv = tf.contrib.layers.fully_connected(hidden_tensor_inv, self.flatten_length//2, activation_fn=self.transfer_fct_conv, trainable=trainable)
			hidden_tensor_inv = vbn(tf.contrib.layers.fully_connected(hidden_tensor_inv, self.flatten_length, activation_fn=self.transfer_fct_conv, trainable=trainable))

			current_input = tf.reshape(hidden_tensor_inv, [-1, 2, 2, 2, 512])
			print 'current_input', current_input.get_shape().as_list()

			def deconv_layer(current_input, kernel_shape, strides, output_shape, scope, transfer_fct, is_training, if_batch_norm, padding, trainable):
				# kernel = tf.truncated_normal(kernel_shape, dtype=tf.float32, stddev=1e-1)
				# kernel = tf.Variable(
				# 	tf.random_uniform(kernel_shape, -1.0 / (math.sqrt(kernel_shape[3]) + 20), 1.0 / (math.sqrt(kernel_shape[3]) + 20)), 
				# 	trainable=trainable)
				kernel = weight_variable(kernel_shape)
				biases = bias_variable([kernel_shape[-2]])
				# biases = tf.Variable(tf.zeros(shape=[kernel_shape[-2]], dtype=tf.float32), trainable=trainable)
				if if_batch_norm:
					current_output = transfer_fct(
						vbn(
							tf.add(deconv3d(current_input, kernel,
								output_shape), biases),
							trainable=trainable, scope=scope
							)
						)
				else:
					current_output = transfer_fct(
						tf.add(deconv3d(current_input, kernel,
							output_shape), biases),
						trainable=trainable, scope=scope
						)
				return current_output
			def transfer_fct_none(x):
				return x
			current_input = deconv_layer(current_input, [4, 4, 4, 128, 256], [1, 2, 2, 2, 1], tf.pack([dyn_batch_size, 4, 4, 4, 256]), 'BN-deconv-0', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="SAME", trainable=trainable)
			print current_input.get_shape().as_list()
			current_input = deconv_layer(current_input, [4, 4, 4, 64, 128], [1, 2, 2, 2, 1], tf.pack([dyn_batch_size, 8, 8, 8, 128]), 'BN-deconv-1', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding ="SAME", trainable=trainable)
			print current_input.get_shape().as_list()
			current_input = deconv_layer(current_input, [4, 4, 4, 32, 64], [1, 2, 2, 2, 1], tf.pack([dyn_batch_size, 15, 15, 15, 64]), 'BN-deconv-2', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="SAME", trainable=trainable)
			print current_input.get_shape().as_list()
			current_input = deconv_layer(current_input, [4, 4, 4, 1, 32], [1, 2, 2, 2, 1], tf.pack([dyn_batch_size, 30, 30, 30, 1]), 'BN-deconv-3', transfer_fct_none, is_training=self.is_training, if_batch_norm=False, padding="SAME", trainable=trainable)
			print current_input.get_shape().as_list()
			print '---------- _<<< generator: flatten length:', self.flatten_length
			return (tf.nn.tanh(current_input), current_input)

	# def _train_align0(self, is_training):
	# 	_, cost, cost_recon, cost_gan, merged, \
	# 	x0_reduce, x, x_recon, x_idx, z_mean, z_log_sigma_sq, z, steps, is_training = self.sess.run(
	# 		(self.optimizer, self.cost, self.recon_loss0, self.latent_loss0, self.merged_summaries0, \
	# 			self.x0_reduce, self.x0, self.x0_recon, self.gen.x0_batch_idx, self.z0_mean, self.z0_log_sigma_sq, self.z0, self.global_step, self.is_training), \
	# 		feed_dict={self.is_training: True, self.gen.is_training: True, self.is_queue: True, self.train_net: True})
	# 	print x0_reduce[:1]
	# 	print z_mean[:2], np.amax(z_mean), np.amin(z_mean)
	# 	std_diev = np.sqrt(np.exp(z_log_sigma_sq[:2]))
	# 	print std_diev, np.amax(std_diev), np.amin(std_diev)
	# 	return (cost, cost_recon, cost_gan, merged, x, x_recon, x_idx, z_mean, z_log_sigma_sq, z, steps)

	# def _test_align0(self, is_training):
	# 	cost, cost_recon, cost_gan, merged, x, x_recon, x_idx, z_mean, z_log_sigma_sq, z, steps, is_training = self.sess.run(
	# 		(self.cost, self.recon_loss0, self.latent_loss0, self.merged_summaries0_test, \
	# 			self.x0, self.x0_recon, self.gen.x0_batch_idx, self.z0_mean, self.z0_log_sigma_sq, self.z0, self.global_step, self.is_training), \
	# 		feed_dict={self.is_training: False, self.gen.is_training: False, self.is_queue: True, self.train_net: True})
	# 	return (cost, cost_recon, cost_gan, merged, x, x_recon, x_idx, z_mean, z_log_sigma_sq, z, steps)

def draw_sample(fig, data, ms, colormap='rainbow', camera_view=False, p=None):
	# data = np.reshape(data, (30, 30, 30))
	zz, yy, xx = np.where(data >= 0)
	ss = data[np.where(data >= 0)] * 1.0
	ms.set(x=xx, y=yy, z=zz, scalars=ss)
	if p == None:
		p = np.array([1, 0, 0])
	else:
		p = p[[1, 0, 2]]
	
	cam_p = p * 15 + np.array([15, 15, 15])
	# print cam_p
	cam_axis = p / np.linalg.norm(cam_p)
	cam_p = cam_p + p * arrow_length
	ar1.x = cam_p[0]
	ar1.y = cam_p[1]
	ar1.z = cam_p[2]
	ar1.pos = ar1.pos / arrow_length
	ar1.axis = -p

	if camera_view:
		mlab.view(azimuth=180, elevation=-90, focalpoint=[15., 15., 15.], figure=fig)
	else:
		mlab.view(*view_default, figure=fig)
	im = imayavi_return_inline(fig=fig)
	return im

def prepare_for_training(gan):
	np.set_printoptions(suppress=True)
	np.set_printoptions(precision=2)
	np.set_printoptions(threshold=np.inf) # whether output entire matrix without clipping
	print '+++++ prepare_for_training...'
	# Training cycle
	if FLAGS.if_draw:
		global figM
		figM = mlab.figure(size=(300, 300))
		# mlab.axes(x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=True, figure=figM)
		data = np.random.rand(30, 30, 30)
		zz, yy, xx = np.where(data >= 0)
		ss = data[np.where(data >= 0)] * 1.0
		colormap='rainbow'
		s = mlab.points3d(xx, yy, zz, ss,
					  mode="cube",
					  colormap=colormap,
					  scale_factor=1,
					  figure=figM,
					 )
		s.scene.light_manager.lights[0].activate = True;
		s.scene.light_manager.lights[0].intensity = 0.7;
		s.scene.light_manager.lights[1].activate = True;
		s.scene.light_manager.lights[1].intensity = 0.3;
		s.scene.light_manager.lights[2].activate = True;
		s.scene.light_manager.lights[2].intensity = 1.0;
		s.scene.light_manager.lights[3].activate = False;
		mlab.xlabel('x', object=s)
		global ms
		ms = s.mlab_source
		global view_default
		view_default = mlab.view()
		
		global figM_2d
		figM_2d = mlab.figure(size=(300, 300))
		# mlab.axes(x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=True, figure=figM_2d)
		data_2d = np.random.rand(30, 30, 1)
		yy_2d, zz_2d, xx_2d = np.where(data_2d >= 0)
		ss_2d = data_2d[np.where(data_2d >= 0)] * 1.0
		colormap='rainbow'
		s_2d = mlab.points3d(xx_2d, yy_2d, zz_2d, ss_2d,
					  mode="cube",
					  colormap=colormap,
					  scale_factor=1,
					  figure=figM_2d,
					 )
		s_2d.scene.light_manager.lights[0].activate = True;
		s_2d.scene.light_manager.lights[0].intensity = 0.7;
		s_2d.scene.light_manager.lights[1].activate = True;
		s_2d.scene.light_manager.lights[1].intensity = 0.3;
		s_2d.scene.light_manager.lights[2].activate = True;
		s_2d.scene.light_manager.lights[2].intensity = 1.0;
		s_2d.scene.light_manager.lights[3].activate = False;
		mlab.xlabel('x', object=s_2d)
		global ms_2d
		ms_2d = s_2d.mlab_source

		visual.set_viewer(figM)
		global arrow_length
		arrow_length = 5
		global ar1
		ar1 = visual.Arrow(x=40,y=15,z=15, color=(1.0, 1.0, 0.0))
		ar1.length_cone = 0.5
		ar1.radius_cone = 0.2
		ar1.radius_shaft = 0.1
		ar1.actor.scale = [arrow_length, arrow_length, arrow_length]
		if FLAGS.train_net == True:
			ar1.visibility = False

		# print view_default

		global pltfig_3d
		pltfig_3d = plt.figure(1, figsize=(45, 15))
		plt.show(block=False)
		# if FLAGS.if_show == False:
		# 	global pltfig_3d_recon
		# 	pltfig_3d_recon = plt.figure(2, figsize=(45, 15))
		# 	plt.show(block=False)
		# 	global pltfig_2d
		# 	pltfig_2d = plt.figure(5, figsize=(45, 15))
		# 	plt.show(block=False)
		# 	global pltfig_2d_reproj
		# 	pltfig_2d_reproj = plt.figure(6, figsize=(45, 15))
		# 	plt.show(block=False)
		# 	global pltfig_2d_reproj_gnd
		# 	pltfig_2d_reproj_gnd = plt.figure(7, figsize=(45, 15))
		# 	plt.show(block=False)
	def _count_train_set(filename):
		count = 0
		options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
		for serialized_example in tf.python_io.tf_record_iterator(filename, options):
			count = count + 1
		if FLAGS.train_net:
			print("++++++++++ Total training samples: %d; %d batches in an epoch." % (count, int(count/FLAGS.models_in_batch)))
		# else:
		# 	print("++++++++++ Total training samples: %d; %d batches in an epoch." % (count, int(count/FLAGS.batch_size)))
		return count
	global num_samples
	global num_samples_test
	print '+++++ counting samples...'
	if FLAGS.train_net:
		# num_samples = _count_train_set(FLAGS.data_train_net0)
		# num_samples_test = _count_train_set(FLAGS.data_test_net0)
		num_samples = 5776
		num_samples_test = 1474 
	# else:
	# num_samples = _count_train_set(FLAGS.data_train_net1)
	# num_samples_test = _count_train_set(FLAGS.data_test_net1)

	if FLAGS.if_disp:
		global pltfig_z0
		pltfig_z0 = plt.figure(3, figsize=(20, 8))
		plt.show(block=False)
	global tsne_model
	tsne_model = TSNE(n_components=2, random_state=0, init='pca')
	print '+++++ prepare_for_training finished.'

def train(gan):
	prepare_for_training(gan)
	# sample_z = np.random.uniform(-1, 1, size=(gan.batch_size, gan.z_size))
	try:
		while not gan.coord.should_stop():
			start_time = time.time()

			batch_z = np.random.normal(0., 1., [gan.batch_size, gan.z_size]) \
							.astype(np.float32)

			accu_real, accu_fake = gan.sess.run([gan.accu_real, gan.accu_fake],
				feed_dict={gan.z: batch_z, gan.is_training: True, gan.gen.is_training: True, gan.is_queue: True, gan.train_net: True})
			accu = 0.5 *(accu_real + accu_fake)
			print accu, accu_fake, accu_real
			# Update D network
			# if accu < 0.7:
			_, summary_str_1 = gan.sess.run([gan.d_optim, gan.d_sum],
				feed_dict={gan.z: batch_z, gan.is_training: True, gan.gen.is_training: True, gan.is_queue: True, gan.train_net: True})
			# else:
			# 	summary_str_1 = gan.sess.run(gan.d_sum,
			# 		feed_dict={gan.z: batch_z, gan.is_training: True, gan.gen.is_training: True, gan.is_queue: True, gan.train_net: True})

			# Update G network
			_, summary_str_2 = gan.sess.run([gan.g_optim, gan.g_sum],
				feed_dict={gan.z: batch_z, gan.is_training: True, gan.gen.is_training: True, gan.is_queue: True, gan.train_net: True})

			# # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
			# _, summary_str_3 = gan.sess.run([gan.g_optim, gan.g_sum],
			# 	feed_dict={gan.z: batch_z, gan.is_training: True, gan.gen.is_training: True, gan.is_queue: True, gan.train_net: True})

			x_recon, errD_fake, errD_real, errG, step = gan.sess.run([gan.G, gan.d_loss_fake, gan.d_loss_real, gan.g_loss, gan.global_step],
				feed_dict={gan.z: batch_z, gan.is_training: True, gan.gen.is_training: True, gan.is_queue: True, gan.train_net: True})

			epoch_show = math.floor(float(step) * FLAGS.models_in_batch / float(num_samples))
			batch_show = math.floor(step - epoch_show * (num_samples / FLAGS.models_in_batch))

			if FLAGS.if_summary:
				gan.train_writer.add_summary(summary_str_1, step)
				gan.train_writer.add_summary(summary_str_2, step)
				# gan.train_writer.add_summary(summary_str_3, step)
				gan.train_writer.flush()
				if FLAGS.train_net:
					print "STEP", '%03d' % (step), "Epo", '%03d' % (epoch_show), "ba", '%03d' % (batch_show), \
					"accu: %.4f, d_loss: %.8f, g_loss: %.8f" % (accu, errD_fake+errD_real, errG)

			if FLAGS.if_save and step != 0 and step % FLAGS.save_every_step == 0:
				save_gan(gan, step, epoch_show, batch_show)

			# if FLAGS.if_test and step % FLAGS.test_every_step == 0 and step != 0:
			# if FLAGS.train_net:
			# 	cost, cost_recon, cost_gan, merged, x, x_recon, x_idx, z_mean, z_log_sigma_sq, z, step = gan._test_align0(is_training=False)
			# if FLAGS.if_summary:
			# 	gan.train_writer.add_summary(merged, step)
			# 	gan.train_writer.flush()
			# 	if FLAGS.train_net:
			# 		print "TESTING net 0... "\
			# 		"cost =", "%.4f = %.4f + %.4f" % (\
			# 			cost, cost_recon * FLAGS.reweight_recon, cost_gan * FLAGS.reweight_gan), \
			# 		"-- recon = %.4f, gan = %.4f" % (cost_recon / gan.x_size, cost_gan)
			if FLAGS.if_draw and step % FLAGS.draw_every == 0:
				print 'Drawing reconstructed sample from testing batch...'
				# plt.figure(1)
				# for test_idx in range(15):
				# 	im = draw_sample(figM, x[test_idx].reshape((30, 30, 30)), ms)
				# 	plt.subplot(3, 5, test_idx+1)
				# 	plt.imshow(im)
				# 	plt.axis('off')
				# pltfig_3d.suptitle('Target models at step %s of %s'%(step, FLAGS.folder_name_save_to), fontsize=20, fontweight='bold')
				# pltfig_3d.canvas.draw()
				# pltfig_3d.savefig(params['summary_folder']+'/%d-pltfig_3d_gnd.png'%step)
				plt.figure(1)
				print x_recon.shape
				for test_idx in range(2):
					im = draw_sample(figM, x_recon[test_idx].reshape((30, 30, 30)), ms)
					plt.subplot(3, 5, test_idx+1)
					plt.imshow(im)
					plt.axis('off')
				pltfig_3d.suptitle('Reconstructed models at step %s of %s'%(step, FLAGS.folder_name_save_to), fontsize=20, fontweight='bold')
				pltfig_3d.canvas.draw()
				pltfig_3d.savefig(params['summary_folder']+'/%d-pltfig_3d_recon.png'%step)
				# if FLAGS.train_net == False:
				# 	plt.figure(5)
				# 	for test_idx in range(15):
				# 		plt.subplot(3, 5, test_idx+1)
				# 		plt.imshow(((x2d_rgb[test_idx] + 0.5) * 255.).astype(np.uint8))
				# 		plt.axis('off')
				# 	pltfig_2d.suptitle('Target RGB image at step %s of %s'%(step, FLAGS.folder_name_save_to), fontsize=20, fontweight='bold')
				# 	pltfig_2d.canvas.draw()
				# 	pltfig_2d.savefig(params['summary_folder']+'/%d-pltfig_rgb.png'%step)
				# 	plt.figure(6)
				# 	for test_idx in range(15):
				# 		im = draw_sample(figM_2d, x_proj[test_idx].reshape((30, 30, 1)), ms_2d, camera_view=True)
				# 		plt.subplot(3, 5, test_idx+1)
				# 		plt.imshow(im)
				# 		plt.axis('off')
				# 	pltfig_2d_reproj.suptitle('Reprojection at step %s of %s'%(step, FLAGS.folder_name_save_to), fontsize=20, fontweight='bold')
				# 	pltfig_2d_reproj.canvas.draw()
				# 	pltfig_2d_reproj.savefig(params['summary_folder']+'/%d-pltfig_2d_proj.png'%step)
				# 	plt.figure(7)
				# 	for test_idx in range(15):
				# 		im = draw_sample(figM_2d, x2d_gnd[test_idx].reshape((30, 30, 1)), ms_2d, camera_view=True)
				# 		plt.subplot(3, 5, test_idx+1)
				# 		plt.imshow(im)
				# 		plt.axis('off')
				# 	pltfig_2d_reproj_gnd.suptitle('Gnd truth projection at step %s of %s'%(step, FLAGS.folder_name_save_to), fontsize=20, fontweight='bold')
				# 	pltfig_2d_reproj_gnd.canvas.draw()
				# 	pltfig_2d_reproj_gnd.savefig(params['summary_folder']+'/%d-pltfig_2d_gnd.png'%step)
			end_time = time.time()
			elapsed = end_time - start_time
			print "--- Time %f seconds."%elapsed
	except tf.errors.OutOfRangeError:
		print('Done training.')
	finally:
		# When done, ask the threads to stop.
		gan.coord.request_stop()
	# Wait for threads to finish.
	gan.coord.join(gan.threads)
	gan.sess.close()

def save_gan(gan, step, epoch, batch):
	if FLAGS.train_net:
		net_folder = gan.params['model_folder'] + '/net0'
	# else:
	# 	net_folder = gan.params['model_folder'] + '/net1'
	save_path = gan.saver_for_resume.save(gan.sess, \
		net_folder + '/%s-step%d-epoch%d-batch%d.ckpt' % (params['save_name'], step, epoch, batch), \
		global_step=step)
	print("-----> Model saved to file: %s; step = %d" % (save_path, step))

def restore_gan(gan, params):
	# if FLAGS.train_netgan.saver_for_restore_encoder0.restore(gan.sess, latest_checkpoint)
	if FLAGS.train_net:
		net_folder = gan.params['model_folder_restore'] + '/net0'
	else:
		net_folder = gan.params['model_folder_restore'] + '/net1'
	net0_folder = gan.params['model_folder_net0_restore'] + '/net0'

	if FLAGS.restore_encoder0:
		if "ckpt" not in net0_folder:
			latest_checkpoint = tf.train.latest_checkpoint(net0_folder)
		else:
			latest_checkpoint = FLAGS.folder_name_net0_restore_from
		print "+++++ Loading net0 from: %s" % (latest_checkpoint)
		gan.restorer_encoder0.restore(gan.sess, latest_checkpoint)
	else:
		if FLAGS.resume_from == 1:
			if "ckpt" not in gan.params['model_folder_restore']:
				latest_checkpoint = tf.train.latest_checkpoint(gan.params['model_folder_restore'] + '/')
			else:
				latest_checkpoint = FLAGS.folder_name_restore_from
			print "+++++ Resuming: Model restored from folder: %s file: %s" % (gan.params['model_folder_restore'], latest_checkpoint)
			gan.restorer.restore(gan.sess, latest_checkpoint)
# ## Illustrating reconstruction quality

params = dict(n_input=27000,
	summary_folder='./summary/' + FLAGS.folder_name_save_to,
	model_folder='./snapshot/' + FLAGS.folder_name_save_to,
	model_folder_restore='./snapshot/' + FLAGS.folder_name_restore_from,
	model_folder_net0_restore='./snapshot/' + FLAGS.folder_name_net0_restore_from,
	save_name='model_test',
	batch_size=FLAGS.batch_size,
	learning_rate=FLAGS.learning_rate,
	n_z=FLAGS.n_z)  # dimensionality of latent space)

if FLAGS.resume_from == -1 and FLAGS.if_show == False:
	print '===== Starting new; summary folder removed. Press enter to remove'
	# raw_input("Press Enter to continue...")
	os.system('rm -rf %s/ %s/' % (params["summary_folder"], params["model_folder"]))

# global alexnet_data
# alexnet_data = np.load("./alexnet/bvlc_alexnet.npy").item()
# print '===== Alexnet loaded.'
gan = VariationalAutoencoder(params)

global net_folder
if FLAGS.train_net:
	net_folder = gan.params['model_folder'] + '/net0'
# else:
# 	net_folder = gan.params['model_folder'] + '/net1'
if FLAGS.resume_from == 1 and FLAGS.folder_name_restore_from == "":
	print "+++++ Setting restore folder name to the saving folder name..."
	params["model_folder_restore"] = net_folder
print '===== FLAGS.restore_encoder0:', FLAGS.restore_encoder0
print '===== FLAGS.folder_name_restore_from:', FLAGS.folder_name_restore_from
print '===== FLAGS.folder_name_net0_restore_from:', FLAGS.folder_name_net0_restore_from
if FLAGS.resume_from != -1 or FLAGS.restore_encoder0 == True:
	restore_gan(gan, params)

if FLAGS.if_show == False:
	print("-----> Starting Training")
	print 'if_unlock_decoder0', FLAGS.if_unlock_decoder0
	print [var.op.name for var in tf.trainable_variables()]
	train(gan)
else:
	print("-----> Starting Showing")
	if FLAGS.train_net:
		show_0(gan)
	# else:
	# 	show_1(gan)	