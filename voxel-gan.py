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
flags.DEFINE_integer("n_z", 20, "hidden size")
flags.DEFINE_boolean("if_conv", False, "if conv")
flags.DEFINE_integer("batch_size", 200, "batch_size")
flags.DEFINE_integer("models_in_batch", 40, "models in a batch")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_float("reweight_recon", 1.0, "weight for recon loss")
flags.DEFINE_float("reweight_reproj", 0.0, "weight for reproj loss")
flags.DEFINE_float("reweight_vae", 1.0, "weight for vae loss")
flags.DEFINE_float("reweight_euc_s", 1.0, "weigth for euc loss of style")
flags.DEFINE_float("reweight_euc_p", 1.0, "weigth for euc loss of pose")
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
flags.DEFINE_string("data_mean", "", "load mean data from")
flags.DEFINE_string("data_test_net0", "", "load test data from")
flags.DEFINE_string("data_train_net1", "", "load train data from")
flags.DEFINE_string("data_test_net1", "", "load test data from")
flags.DEFINE_boolean("if_BN", False, "if batch_norm")
flags.DEFINE_boolean("if_BN_out", False, "if batch_norm for x output layer")
flags.DEFINE_boolean("if_show", True, "if show mode")
flags.DEFINE_boolean("if_alex", False, "if use alexnet for regressor")
flags.DEFINE_boolean("if_unlock_decoder0", False, "if unlock decoder0")
flags.DEFINE_boolean("if_gndS", False, "if_gndS")
flags.DEFINE_boolean("if_gndP", True, "if_gndP")
flags.DEFINE_boolean("if_p_trainable", True, "if recog p is trainable")
flags.DEFINE_boolean("if_s_trainable", True, "if recog s is trainable")
global FLAGS
FLAGS = flags.FLAGS

def xavier_init(fan_in, fan_out, constant=1):
	""" Xavier initialization of network weights"""
	# https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
	low = -constant*np.sqrt(6.0/(fan_in + fan_out))
	high = constant*np.sqrt(6.0/(fan_in + fan_out))
	return tf.random_uniform((fan_in, fan_out),
							 minval=low, maxval=high,
							 dtype=tf.float32)

def xavier_fully(tensor_in, out_size):
	in_size = tensor_in.get_shape().as_list()[1]
	W_fc = tf.Variable(xavier_init(in_size, out_size))
	b_fc = tf.Variable(tf.zeros([out_size], dtype=tf.float32)),
	hidden_tensor = tf.matmul(tensor_in, W_fc) + b_fc
	#hidden_tensor = tflearn.layers.normalization.batch_normalization(hidden_tensor)
	return hidden_tensor

# def batch_normalization(tensor_in):
# 	#print "no batch_normalization"
# 	#return tensor_in
# 	return tflearn.layers.normalization.batch_normalization(tensor_in)

def dropout(tensor_in, par):
	print "no dropout"
	#return tf.nn.dropout(tensor_in, par)
	return tensor_in

class VariationalAutoencoder(object):
	def __init__(self, params, transfer_fct=tf.sigmoid):
		self.params = params
		self.transfer_fct = transfer_fct
		self.transfer_fct_conv = lrelu
		# self.batch_normalization = batch_normalization
		self.dropout = dropout
		self.learning_rate = params['learning_rate']
		self.batch_size = self.params['batch_size']
		self.z_size = self.params["n_z"]
		self.x_size = self.params["n_input"]

		if FLAGS.train_net:
			summary_folder = params['summary_folder'] + '/net0'
		else:
			summary_folder = params['summary_folder'] + '/net1'
		self.train_writer = tf.train.SummaryWriter(summary_folder)
		self.global_step = tf.Variable(-1, name='global_step', trainable=False)

		# Create autoencoder network
		print '-----> Defining network...'
		self._create_network()
		# Define loss function based variational upper-bound and corresponding optimizer
		print '-----> Defining loss...'
		self._create_loss_optimizer()
		print '---------- Defining loss done.'

		self.restorer_encoder0 = tf.train.Saver(slim.get_variables_to_restore(include=["encoder0","decoder0","step"]), max_to_keep=2, write_version=tf.train.SaverDef.V2)
		self.saver_for_resume = tf.train.Saver(slim.get_variables_to_restore(exclude=["is_training"]), max_to_keep=50, write_version=tf.train.SaverDef.V2)

		# variables_to_restore0 = slim.get_variables_to_restore(include=["encoder0","decoder0","step"], exclude=["Adam"])
		# variables_to_restore0 = {k.op.name:k for k in variables_to_restore0 if (not("Adam" in k.op.name) and not("is_training" in k.op.name))}
		# self.restorer_encoder0 = tf.train.Saver(variables_to_restore0, write_version=tf.train.SaverDef.V2)

		# variables_to_restore = tf.all_variables()
		# variables_to_restore = {k.op.name:k for k in variables_to_restore if (
		# 	not("decoder0/fully_connected/biases/Adam" in k.op.name) and 
		# 	not("decoder0/fully_connected/weights/Adam" in k.op.name) and 
		# 	not("decoder0/fully_connected_1/biases/Adam" in k.op.name) and 
		# 	not("decoder0/fully_connected_1/weights/Adam" in k.op.name) and 
		# 	not("decoder0/fully_connected_2/biases/Adam" in k.op.name) and 
		# 	not("decoder0/fully_connected_2/weights/Adam" in k.op.name) and 
		# 	not("decoder0/fully_connected_3/biases/Adam" in k.op.name) and 
		# 	not("decoder0/fully_connected_3/weights/Adam" in k.op.name) and 
		# 	not("is_training" in k.op.name))}
		# self.restorer = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2)

		# variables_to_restore = slim.get_variables_to_restore(exclude=["RMSProp", "is_training"])
		# variables_to_restore = {k.op.name:k for k in variables_to_restore if (not("RMSProp" in k.op.name) and not("is_training" in k.op.name))}
		# self.restorer = tf.train.Saver(variables_to_restore, max_to_keep=50, write_version=tf.train.SaverDef.V2)

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

		def x_2_s(x, output_size):
			fc_layer6 = slim.fully_connected(x, 1000, activation_fn=self.transfer_fct_conv)
			fc_layer7 = slim.fully_connected(fc_layer6, 500, activation_fn=self.transfer_fct_conv)
			fc_layer8 = slim.fully_connected(fc_layer7, output_size, activation_fn=self.transfer_fct_conv)
			return fc_layer8
		def s_2_x(s):
			fc_layer10 = slim.fully_connected(s, 100, activation_fn=self.transfer_fct_conv)
			fc_layer11 = slim.fully_connected(fc_layer10, 500, activation_fn=self.transfer_fct_conv)
			fc_layer12 = slim.fully_connected(fc_layer11, 1000, activation_fn=self.transfer_fct_conv)
			return fc_layer12

		with tf.device('/gpu:0'):
			## Define net 0
			self.x0 = tf.cond(self.train_net, lambda: self.gen.x0_batch, lambda: self.gen.x_gnd_batch) # aligned models
			self.x0 = tf.transpose(self.x0, perm=[0, 2, 3, 1, 4])
			self.x0_flatten = tf.reshape(self.x0, [-1, 27000])
			self.dyn_batch_size_x0 = tf.shape(self.x0)[0]
			with tf.variable_scope("encoder0"):
				with slim.arg_scope([slim.fully_connected], trainable=FLAGS.train_net):
					# if FLAGS.if_conv == False:
					# 	self.x0_reduce = slim.fully_connected(x_2_s(self.x0_flatten, 100), self.z_size, activation_fn=self.transfer_fct_conv, trainable=FLAGS.train_net)
					# else:
					# 	self.x0_reduce = slim.fully_connected(
					# 		self._x_2_z_conv(self.x0_flatten, 100, trainable=FLAGS.train_net), 
					# 		self.z_size, activation_fn=self.transfer_fct_conv, trainable=FLAGS.train_net)
					# self.z0_mean = slim.fully_connected(self.x0_reduce, self.z_size, activation_fn=None, trainable=FLAGS.train_net)
					# self.z0_mean = tf.clip_by_value(self.z0_mean, -10., 10.)
					# self.z0_log_sigma_sq = slim.fully_connected(self.x0_reduce, self.z_size, activation_fn=self.transfer_fct_conv, trainable=FLAGS.train_net)
					# self.z0_log_sigma_sq = tf.clip_by_value(self.z0_log_sigma_sq, -10., 10.)
					# eps0 = tf.random_normal(tf.pack([self.dyn_batch_size_x0, self.z_size]))
					# self.z0 = self.z0_mean + eps0 * tf.sqrt(tf.exp(self.z0_log_sigma_sq))
					if FLAGS.if_conv == False:
						self.x0_reduce = slim.fully_connected(x_2_s(self.x0_flatten, 100), self.z_size, activation_fn=self.transfer_fct_conv, trainable=FLAGS.train_net)
					else:
						self.x0_reduce = self._x_2_z_conv(self.x0_flatten, trainable=FLAGS.train_net)
					self.z0_mean = self._fcAfter_x_2_z_conv(self.x0_reduce, self.z_size, trainable=FLAGS.train_net)
					self.z0_mean = tf.clip_by_value(self.z0_mean, -50., 50.)
					self.z0_log_sigma_sq = self._fcAfter_x_2_z_conv(self.x0_reduce, self.z_size, trainable=FLAGS.train_net)
					self.z0_log_sigma_sq = tf.clip_by_value(self.z0_log_sigma_sq, -50., 10.)
					eps0 = tf.random_normal(tf.pack([self.dyn_batch_size_x0, self.z_size]))
					self.z0 = self.z0_mean + eps0 * tf.sqrt(tf.exp(self.z0_log_sigma_sq))

		## Define encoder1 --recog p&s
		with tf.device('/gpu:1'):
			self.x2d_rgb = self.gen.x2d_rgb_batch # batch_size
			self.x2d_rgb_mean = tf.placeholder(tf.float32, [227, 227, 3])
			# self.x2d_rgb_norm = tf.sub(self.x2d_rgb, self.x2d_rgb_mean)
			self.x2d_rgb_norm = tf.sub(tf.div(self.x2d_rgb, 255.), 0.5)
			# self.x2d_rgb_norm = tf.div(self.x2d_rgb_norm, 127.5)

		## GND X
		with tf.device('/gpu:3'):
			self.x = self.gen.x_batch
			self.x = tf.transpose(self.x, perm=[0, 2, 3, 1, 4])
			self.x2d_gnd = self.gen.x2d_gnd_batch # batch_size
			self.x2d_gnd = tf.reverse(self.x2d_gnd, [False, True, False])
				
			## X with GND p
			self.x3d_gnd_rot = tf.reshape(tf.clip_by_value(self.gen._transform(self.x0, self.gen.R_s_flat, self.gen.out_size), 0., 1.), [-1, 30, 30, 30, 1])
			self.x2d_gnd_proj = self._proj(self.x3d_gnd_rot)
			
			self.dyn_batch_size_x = tf.shape(self.x)[0]
			self.x_flatten = tf.reshape(self.x, [-1, 27000])
		with tf.device('/gpu:1'):
			with tf.variable_scope("recog_s"):
				if FLAGS.if_gndS == False:
					if FLAGS.if_alex == False:
						x_reduce_size = 256
						_, self.x_reduce_s = self._x2d_2_z_conv(self.x2d_rgb_norm, x_reduce_size)
					else:
						self.x_reduce_s = self._x2d_2_z_conv_alex(self.x2d_rgb_norm, trainable=FLAGS.if_s_trainable)
					if FLAGS.if_alex == False:
						self.z_mean = slim.fully_connected(slim.fully_connected(self.x_reduce_s, x_reduce_size, activation_fn=self.transfer_fct_conv), self.z_size, activation_fn=None)
						# self.z_log_sigma_sq = slim.fully_connected(slim.fully_connected(self.x_reduce_s, x_reduce_size, activation_fn=self.transfer_fct_conv), self.z_size, activation_fn=None)
					else:
						self.z_mean = self._fcAfter_x2d_2_z_conv_alex(self.x_reduce_s, self.z_size, trainable=FLAGS.if_s_trainable)
						# self.z_log_sigma_sq = self._fcAfter_x2d_2_z_conv_alex(self.x_reduce_s, self.z_size)
					# eps = tf.random_normal(tf.pack([self.dyn_batch_size_x, self.z_size]))
					# self.z = self.z_mean + eps * tf.exp(self.z_log_sigma_sq)
					self.z = self.z_mean
				else:
					self.z = self.z0_mean
					self.z_mean = self.z0_mean
					# self.z_log_sigma_sq = self.z0_log_sigma_sq
		with tf.device('/gpu:1'):
			with tf.variable_scope("recog_p"):
				self.p_s = self.gen.p_s
				if FLAGS.if_gndP == False:
					if FLAGS.if_alex == False:
						x_reduce_size = 256
						_, self.x_reduce_p = self._x2d_2_z_conv(self.x2d_rgb_norm, x_reduce_size)
					else:
						self.x_reduce_p = self._x2d_2_z_conv_alex(self.x2d_rgb_norm, trainable=FLAGS.if_p_trainable)
					if FLAGS.if_alex == False:
						self.p_predict, _ = self._x2d_2_z_conv(self.x_reduce_p, 3)
					else:
						self.p_predict = self._fcAfter_x2d_2_z_conv_alex(self.x_reduce_p, 3, trainable=FLAGS.if_p_trainable)
					def euclidean_norm(tensor, reduction_indicies = 1, name = None):
						squareroot_tensor = tf.square(tensor)
						euclidean_norm = tf.reduce_sum(squareroot_tensor, reduction_indices = reduction_indicies, keep_dims = True)
						return tf.sqrt(euclidean_norm)
					self.p_l2norm = euclidean_norm(self.p_predict, reduction_indicies = 1) + 1e-10
					self.p_s_l2norm = euclidean_norm(self.p_s, reduction_indicies = 1) + 1e-10
					self.p_unit = tf.div(self.p_predict, self.p_l2norm)
					self.p_predict = np.pi * tf.mul(tf.nn.tanh(self.p_l2norm), self.p_unit)
				else:
					self.p_predict = self.p_s

		## Define decoder0
		with tf.device('/gpu:2'):
			z0_for_recon = tf.cond(self.train_net, lambda: self.z0, lambda: self.z)
			with tf.variable_scope("decoder0"):
				with slim.arg_scope([slim.fully_connected], trainable=(FLAGS.train_net or FLAGS.if_unlock_decoder0)):
					if FLAGS.if_conv == False:
						self.x0_recon_flatten = slim.fully_connected(s_2_x(z0_for_recon), 27000, activation_fn=tf.sigmoid, trainable=(FLAGS.train_net or FLAGS.if_unlock_decoder0))
						self.x0_recon = tf.reshape(self.x0_recon_flatten, [-1, 30, 30, 30, 1])
					else:
						self.x0_recon = self._z_2_x_conv(z0_for_recon, trainable=(FLAGS.train_net or FLAGS.if_unlock_decoder0))
						# self.x0_recon_flatten = tf.reshape(self.x0_recon, [-1, 27000])

		## Define decoder1
		with tf.device('/gpu:3'):
			with tf.variable_scope("decoder1"):
				# self.Rs_flat_pred = self.gen._tws_to_Rs_flat(self.p_s)
				# self.Rs_flat_pred = self.gen._tws_to_Rs_flat(tf.cond(self.is_training, lambda:self.p_s, lambda: self.p_predict))
				self.Rs_flat_pred = self.gen._tws_to_Rs_flat(self.p_predict)
				self.x_recon = tf.reshape(tf.clip_by_value(self.gen._transform(self.x0_recon, self.Rs_flat_pred, self.gen.out_size), 0., 1.), [-1, 30, 30, 30, 1])
				self.x_recon_flatten = tf.reshape(self.x_recon, [-1, 27000])
				self.x_proj = self._proj(self.x_recon)

	def _flatten_to_25d(self, X):
		def _do_with_single(X_3d_single):
			X_3d_single_sum = tf.scan(lambda a, x: a + x, X_3d_single, initializer=tf.zeros(tf.pack([30, 30, 1])))
			X_3d_single_25d = tf.select(\
				tf.logical_and(tf.greater(X_3d_single_sum, tf.constant(0.)), tf.less(X_3d_single_sum, tf.constant(1.5))), \
				X_3d_single, tf.to_float(tf.zeros_like(X_3d_single)))
			return X_3d_single_25d
		return tf.map_fn(_do_with_single, X)
	
	def _proj(self, X):
		X = tf.transpose(X, perm=[0, 3, 1, 2, 4])
		X25 = self._flatten_to_25d(X)
		return tf.reduce_max(self._flatten_to_25d(X25), 1)
		# return self._flatten_to_25d(X)

	def _create_loss_optimizer(self):
		with tf.device('/gpu:3'):
			epsilon = 1e-10
			self.recon_loss0 = tf.reduce_mean(tf.reduce_sum(-self.x0_flatten * tf.log(tf.reshape(self.x0_recon, [-1, 27000]) + epsilon) -
							 (1.0 - self.x0_flatten) * tf.log(1.0 - tf.reshape(self.x0_recon, [-1, 27000]) + epsilon), 1))
		with tf.device('/gpu:0'):
			## http://kvfrans.com/variational-autoencoders-explained/
			## tf.exp(self.z0_log_sigma_sq):= tf.square(z_stddev) --> self.z0_log_sigma_sq:= log(tf.square(z_stddev))
			self.latent_loss0 = tf.reduce_mean(tf.reduce_sum(0.5 * (tf.square(self.z0_mean) + tf.exp(self.z0_log_sigma_sq) -
									self.z0_log_sigma_sq- 1.0), 1))

		with tf.device('/gpu:1'):
			self.euc_loss_s = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.z_mean - self.z0_mean), 1)), 0)
			self.euc_loss_p = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.p_s - self.p_predict), 1)), 0) \
				# + 10 * tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.p_l2norm - self.p_s_l2norm), 1)), 0)
		with tf.device('/gpu:3'):
			# self.recon_loss_x = tf.reduce_mean(tf.reduce_sum(-self.x_flatten * tf.log(self.x_recon_flatten + epsilon) -
			# 				 (1.0 - self.x_flatten) * tf.log(1.0 - self.x_recon_flatten + epsilon), 1))
			# self.x_flatten = (self.x_flatten * 3.) - 1.
			# self.x_recon_flatten = (self.x_recon_flatten * 0.9) + 0.1
			self.recon_loss_x_1 = tf.reduce_mean(tf.reduce_sum(- self.x_flatten * tf.log(self.x_recon_flatten + epsilon), 1))
			self.recon_loss_x_2 = tf.reduce_mean(tf.reduce_sum(- (1.0 - self.x_flatten) * tf.log(1.0 - self.x_recon_flatten + epsilon), 1))
			self.recon_loss_x_reweight = 0.97 * self.recon_loss_x_1 + 0.03 * self.recon_loss_x_2
			self.recon_loss_x = self.recon_loss_x_1 + self.recon_loss_x_2
			# self.recon_loss_x = self.recon_loss_x_reweight

			x2d_gnd_flatten = tf.reshape(self.x2d_gnd, [-1, 30*30])
			x_proj_flatten = tf.reshape(self.x_proj, [-1, 30*30])
			self.reproj_loss_1 = tf.reduce_mean(tf.reduce_sum(- x2d_gnd_flatten * tf.log(x_proj_flatten + epsilon), 1))
			self.reproj_loss_2 = tf.reduce_mean(tf.reduce_sum(- (1.0 - x2d_gnd_flatten) * tf.log(1.0 - x_proj_flatten + epsilon),1))
			self.reproj_loss_reweight = 0.03 * self.reproj_loss_1 + 0.97 * self.reproj_loss_2
			self.reproj_loss = self.reproj_loss_1 + self.reproj_loss_2
			ones_2d = tf.ones_like(x_proj_flatten, dtype=tf.int8)
			zeros_2d = tf.zeros_like(x_proj_flatten, dtype=tf.int8)
			x2d_gnd_flatten_int = tf.cast(x2d_gnd_flatten, tf.int8)
			x_proj_flatten_int = tf.cast(tf.select(x_proj_flatten > 0.5, ones_2d, zeros_2d), tf.int8)
			GndP = tf.equal(x2d_gnd_flatten_int, ones_2d)
			GndN = tf.equal(x2d_gnd_flatten_int, zeros_2d)
			PredP = tf.equal(x_proj_flatten_int, ones_2d)
			PredN = tf.equal(x_proj_flatten_int, zeros_2d)
			# https://www.wikiwand.com/en/Precision_and_recall
			TP = tf.reduce_sum(tf.cast(tf.logical_and(GndP, PredP), tf.float32))
			TN = tf.reduce_sum(tf.cast(tf.logical_and(GndN, PredN), tf.float32))
			FP = tf.reduce_sum(tf.cast(tf.logical_and(GndN, PredP), tf.float32))
			FN = tf.reduce_sum(tf.cast(tf.logical_and(GndP, PredN), tf.float32))
			self.accuracy = tf.div((TP + TN), (TP + TN + FP + FN))
			self.precision = tf.div(TP, (TP + FP))
			self.recall = tf.div(TP, (TP + FN))

			if FLAGS.train_net:
					self.cost = FLAGS.reweight_recon * self.recon_loss0 + FLAGS.reweight_vae * self.latent_loss0
			else:
					self.cost = FLAGS.reweight_euc_s * self.euc_loss_s + FLAGS.reweight_euc_p * self.euc_loss_p \
						+ FLAGS.reweight_recon * self.recon_loss_x\
						+ FLAGS.reweight_reproj * self.reproj_loss
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, colocate_gradients_with_ops=True)
			# self.optimizer_def = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
			# gvs = self.optimizer_def.compute_gradients(self.cost, colocate_gradients_with_ops=True)
			# def clip_recog_p(grad, var):
			# 	if "recog_p" in var.op.name:
			# 		return (tf.clip_by_value(grad, -0.001, 0.001), var)
			# 	else:
			# 		return (grad, var)
			# capped_gvs = [clip_recog_p(grad, var) for grad, var in gvs ]
			# self.optimizer = self.optimizer_def.apply_gradients(capped_gvs)
			
			summary_loss0 = tf.scalar_summary('loss0/loss', self.cost)
			summary_loss_recon0 = tf.scalar_summary('loss0/rec_loss', self.recon_loss0 / self.x_size)
			summary_loss_latent0 = tf.scalar_summary('loss0/vae_loss', self.latent_loss0)
			summaries0 = [summary_loss0, summary_loss_recon0, summary_loss_latent0]
			self.merged_summaries0 = tf.merge_summary(summaries0)
			summary_loss = tf.scalar_summary('loss/loss', self.cost)
			summary_loss_recon = tf.scalar_summary('loss/rec_loss', self.recon_loss_x / self.x_size)
			summary_loss_recon_reweight = tf.scalar_summary('loss/rec_loss_reweight', self.recon_loss_x_reweight / self.x_size)
			summary_loss_recon_1 = tf.scalar_summary('loss/rec_loss_1', self.recon_loss_x_1 / self.x_size)
			summary_loss_recon_2 = tf.scalar_summary('loss/rec_loss_2', self.recon_loss_x_2 / self.x_size)
			summary_loss_reproj = tf.scalar_summary('loss/reproj_loss', self.reproj_loss / 900.)
			summary_loss_reproj_reweight = tf.scalar_summary('loss/reproj_loss_reweight', self.reproj_loss_reweight / 900.)
			summary_loss_reproj_1 = tf.scalar_summary('loss/reproj_loss_1', self.reproj_loss_1 / 900.)
			summary_loss_reproj_2 = tf.scalar_summary('loss/reproj_loss_2', self.reproj_loss_2 / 900.)
			summary_accuracy = tf.scalar_summary('loss/accuracy', self.accuracy)
			summary_precision = tf.scalar_summary('loss/precision', self.precision)
			summary_recall = tf.scalar_summary('loss/recall', self.recall)
			summary_loss_euc_s = tf.scalar_summary('loss/euc_loss_s', self.euc_loss_s)
			summary_loss_euc_p = tf.scalar_summary('loss/euc_loss_p', self.euc_loss_p)
			summaries = [summary_loss, summary_loss_recon, summary_loss_recon_reweight, summary_loss_recon_1, summary_loss_recon_2, \
				summary_loss_reproj, summary_loss_reproj_reweight, summary_loss_reproj_1, summary_loss_reproj_2, summary_loss_euc_s, summary_loss_euc_p, \
				summary_accuracy, summary_precision, summary_recall]
			self.merged_summaries = tf.merge_summary(summaries)
			summary_loss0_test = tf.scalar_summary('loss0_test/loss', self.cost)
			summary_loss0_recon_test = tf.scalar_summary('loss0_test/rec_loss', self.recon_loss0 / self.x_size)
			summary_loss0_latent_test = tf.scalar_summary('loss0_test/vae_loss', self.latent_loss0)
			summaries0_test = [summary_loss0_test, summary_loss0_recon_test, summary_loss0_latent_test]
			self.merged_summaries0_test = tf.merge_summary(summaries0_test)
			summary_loss_test = tf.scalar_summary('loss_test/loss', self.cost)
			summary_loss_recon_test = tf.scalar_summary('loss_test/rec_loss', self.recon_loss_x / self.x_size)
			summary_loss_reproj_test = tf.scalar_summary('loss_test/reproj_loss', self.reproj_loss / 900.)
			summary_accuracy_test = tf.scalar_summary('loss_test/accuracy', self.accuracy)
			summary_precision_test = tf.scalar_summary('loss_test/precision', self.precision)
			summary_recall_test = tf.scalar_summary('loss_test/recall', self.recall)
			summary_loss_euc_s_test = tf.scalar_summary('loss_test/euc_loss_s', self.euc_loss_s)
			summary_loss_euc_p_test = tf.scalar_summary('loss_test/euc_loss_p', self.euc_loss_p)
			summaries_test = [summary_loss_test, summary_loss_recon_test, summary_loss_reproj_test, summary_loss_euc_s_test, summary_loss_euc_p_test, \
				summary_accuracy_test, summary_precision_test, summary_recall_test]
			self.merged_summaries_test = tf.merge_summary(summaries_test)

	def BatchNorm(self, inputT, trainable, scope=None):
		# Note: is_training is tf.placeholder(tf.bool) type
		# return tf.cond(self.is_training,
		# 	lambda: batch_norm(inputT, is_training=True,
		# 		center=False, updates_collections=None, scope=scope),
		# 	lambda: batch_norm(inputT, is_training=False,
		# 		updates_collections=None, center=False, reuse = True, scope=scope))
		if trainable:
			print '########### BN trainable!!!'
		return tflearn.layers.normalization.batch_normalization(inputT, trainable=trainable)

	def _x2d_2_z_conv_alex(self, input_tensor, trainable=True):
		print '+++++ Defining alex net!'
		def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
			'''From https://github.com/ethereon/caffe-tensorflow
			'''
			c_i = input.get_shape()[-1]
			assert c_i%group==0
			assert c_o%group==0
			convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
			
			if group==1:
				conv = convolve(input, kernel)
			else:
				input_groups = tf.split(3, group, input)
				kernel_groups = tf.split(3, group, kernel)
				output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
				conv = tf.concat(3, output_groups)
			return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

		#conv1
		#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
		k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
		conv1W = tf.Variable(alexnet_data["conv1"][0], trainable=trainable)
		conv1b = tf.Variable(alexnet_data["conv1"][1], trainable=trainable)
		conv1_in = conv(input_tensor, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
		conv1 = tf.nn.relu(conv1_in)
		#lrn1
		#lrn(2, 2e-05, 0.75, name='norm1')
		radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
		lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
		#maxpool1
		#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
		k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
		maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
		#conv2
		#conv(5, 5, 256, 1, 1, group=2, name='conv2')
		k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
		conv2W = tf.Variable(alexnet_data["conv2"][0], trainable=trainable)
		conv2b = tf.Variable(alexnet_data["conv2"][1], trainable=trainable)
		conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
		conv2 = tf.nn.relu(conv2_in)
		#lrn2
		#lrn(2, 2e-05, 0.75, name='norm2')
		radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
		lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
		#maxpool2
		#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
		k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
		maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
		#conv3
		#conv(3, 3, 384, 1, 1, name='conv3')
		k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
		conv3W = tf.Variable(alexnet_data["conv3"][0], trainable=trainable)
		conv3b = tf.Variable(alexnet_data["conv3"][1], trainable=trainable)
		conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
		conv3 = tf.nn.relu(conv3_in)
		#conv4
		#conv(3, 3, 384, 1, 1, group=2, name='conv4')
		k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
		conv4W = tf.Variable(alexnet_data["conv4"][0], trainable=trainable)
		conv4b = tf.Variable(alexnet_data["conv4"][1], trainable=trainable)
		conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
		conv4 = tf.nn.relu(conv4_in)
		#conv5
		#conv(3, 3, 256, 1, 1, group=2, name='conv5')
		k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
		conv5W = tf.Variable(alexnet_data["conv5"][0], trainable=trainable)
		conv5b = tf.Variable(alexnet_data["conv5"][1], trainable=trainable)
		conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
		conv5 = tf.nn.relu(conv5_in)
		#maxpool5
		#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
		k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
		maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
		#fc6
		#fc(4096, name='fc6')
		fc6W = tf.Variable(alexnet_data["fc6"][0], trainable=trainable)
		fc6b = tf.Variable(alexnet_data["fc6"][1], trainable=trainable)
		fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
		#fc7
		#fc(4096, name='fc7')
		fc7W = tf.Variable(alexnet_data["fc7"][0], trainable=trainable)
		fc7b = tf.Variable(alexnet_data["fc7"][1], trainable=trainable)
		fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
		return fc7

	def _fcAfter_x2d_2_z_conv_alex(self, current_input, output_size, trainable=True):
		self.before_flatten_shape_2d = current_input.get_shape().as_list()
		self.flatten_shape_2d = tf.pack([-1, np.prod(current_input.get_shape().as_list() [1:])])
		# current_input = tf.nn.dropout(current_input, 0.8)
		flattened = tf.reshape(current_input, self.flatten_shape_2d)
		self.flatten_length_2d = flattened.get_shape().as_list()[1]
		print '---------- _x2d_2_z_conv: flatten length:', self.flatten_length_2d
		hidden_tensor = tf.contrib.layers.fully_connected(flattened, self.flatten_length_2d//4, activation_fn=self.transfer_fct_conv, trainable=trainable)
		hidden_tensor = tf.contrib.layers.fully_connected(hidden_tensor, self.flatten_length_2d//4, activation_fn=self.transfer_fct_conv, trainable=trainable)
		hidden_tensor = tf.contrib.layers.fully_connected(hidden_tensor, output_size, activation_fn=self.transfer_fct_conv, trainable=trainable)
		return hidden_tensor


	def _x2d_2_z_conv(self, input_tensor, output_size):
		current_input = input_tensor

		def conv_layer(current_input, kernel_shape, strides, scope, transfer_fct, is_training, if_batch_norm, padding):
			# kernel = tf.truncated_normal(kernel_shape, dtype=tf.float32, stddev=1e-3)
			kernel = tf.Variable(
				tf.random_uniform(kernel_shape, -1.0 / (math.sqrt(kernel_shape[3]) + 10), 1.0 / (math.sqrt(kernel_shape[3]) + 10)))
			biases = tf.Variable(tf.zeros(shape=[kernel_shape[-1]], dtype=tf.float32), trainable=True)
			if if_batch_norm:
				# current_output = self.BatchNorm(
				# 		transfer_fct(tf.add(tf.nn.conv2d(current_input, kernel, strides, padding), biases)), 
				# 		is_training=is_training, scope=scope)
				current_output = transfer_fct(
					self.BatchNorm(
						tf.add(tf.nn.conv2d(current_input, kernel, strides, padding), biases),
						trainable=trainable, scope=scope
						)
					)
			else:
				current_output = transfer_fct(
						tf.nn.bias_add(tf.nn.conv2d(current_input, kernel, strides, padding), biases),
					)
			return current_output
		def transfer_fct_none(x):
			return x

		current_input = conv_layer(current_input, [3, 3, 3, 64], [1, 2, 2, 1], 'BN-2d-0', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="SAME")
		print current_input.get_shape().as_list()
		current_input = conv_layer(current_input, [3, 3, 64, 128], [1, 2, 2, 1], 'BN-2d-1', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="SAME")
		print current_input.get_shape().as_list()
		current_input = conv_layer(current_input, [3, 3, 128, 256], [1, 2, 2, 1], 'BN-2d-2', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="SAME")
		print current_input.get_shape().as_list()
		current_input = conv_layer(current_input, [3, 3, 256, 256], [1, 2, 2, 1], 'BN-2d-3', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="SAME")
		print current_input.get_shape().as_list()

		self.before_flatten_shape_2d = current_input.get_shape().as_list()
		self.flatten_shape_2d = tf.pack([-1, np.prod(current_input.get_shape().as_list() [1:])])
		# current_input = tf.nn.dropout(current_input, 0.8)
		flattened = tf.reshape(current_input, self.flatten_shape_2d)
		self.flatten_length_2d = flattened.get_shape().as_list()[1]
		print '---------- _x2d_2_z_conv: flatten length:', self.flatten_length_2d
		# hidden_tensor = tf.contrib.layers.fully_connected(flattened, self.flatten_length_2d//4, activation_fn=None)
		# hidden_tensor = tf.contrib.layers.fully_connected(hidden_tensor, self.flatten_length_2d//4, activation_fn=None)
		# hidden_tensor = tf.contrib.layers.fully_connected(hidden_tensor, output_size, activation_fn=None)
		hidden_tensor = xavier_fully(flattened, self.flatten_length_2d//4)
		hidden_tensor = xavier_fully(hidden_tensor, self.flatten_length_2d//4)
		hidden_tensor = xavier_fully(hidden_tensor, output_size)
		return (hidden_tensor, flattened)

	def _x_2_z_conv(self, input_tensor, trainable):
		input_shape=[None, 27000]

		x_tensor = tf.reshape(input_tensor, [-1, 30, 30, 30, 1])
		current_input = x_tensor

		def conv_layer(current_input, kernel_shape, strides, scope, transfer_fct, is_training, if_batch_norm, padding, trainable):
			# kernel = tf.truncated_normal(kernel_shape, dtype=tf.float32, stddev=1e-3)
			kernel = tf.Variable(
				tf.random_uniform(kernel_shape, -1.0 / (math.sqrt(kernel_shape[3]) + 10), 1.0 / (math.sqrt(kernel_shape[3]) + 10)), 
				trainable=trainable)
			biases = tf.Variable(tf.zeros(shape=[kernel_shape[-1]], dtype=tf.float32), trainable=trainable)
			if if_batch_norm:
				current_output = transfer_fct(
					self.BatchNorm(
						tf.add(tf.nn.conv3d(current_input, kernel, strides, padding), biases),
						trainable=trainable, scope=scope
						)
					)
			else:
				current_output = transfer_fct(
						tf.nn.bias_add(tf.nn.conv3d(current_input, kernel, strides, padding), biases),
					)
			return current_output
		def transfer_fct_none(x):
			return x

		current_input = conv_layer(current_input, [3, 3, 3, 1, 32], [1, 1, 1, 1, 1], 'BN-0', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="VALID", trainable=trainable)
		print current_input.get_shape().as_list()
		current_input = conv_layer(current_input, [3, 3, 3, 32, 64], [1, 2, 2, 2, 1], 'BN-1', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="SAME", trainable=trainable)
		print current_input.get_shape().as_list()
		current_input = conv_layer(current_input, [5, 5, 5, 64, 64], [1, 1, 1, 1, 1], 'BN-2', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="VALID", trainable=trainable)
		print current_input.get_shape().as_list()
		current_input = conv_layer(current_input, [5, 5, 5, 64, 128], [1, 2, 2, 2, 1], 'BN-3', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="SAME", trainable=trainable)
		print current_input.get_shape().as_list()
		current_input = conv_layer(current_input, [5, 5, 5, 128, 256], [1, 1, 1, 1, 1], 'BN-4', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="VALID", trainable=trainable)
		print current_input.get_shape().as_list()
		# current_input = conv_layer(current_input, [3, 3, 3, 128, 256], [1, 2, 2, 2, 1], 'BN-6', transfer_fct_none, is_training=self.is_training, if_batch_norm=False, padding="SAME")
		# print current_input.get_shape().as_list()

		self.before_flatten_shape = current_input.get_shape().as_list()
		self.flatten_shape = tf.pack([-1, np.prod(current_input.get_shape().as_list() [1:])])
		flattened = tf.reshape(current_input, self.flatten_shape)
		self.flatten_length = flattened.get_shape().as_list()[1]
		return flattened

	def _fcAfter_x_2_z_conv(self, flattened, output_size, trainable):
		print '---------- _x_2_z_conv: flatten length:', self.flatten_length
		hidden_tensor = tf.contrib.layers.fully_connected(flattened, self.flatten_length//4, activation_fn=self.transfer_fct_conv, trainable=trainable)
		hidden_tensor = tf.contrib.layers.fully_connected(hidden_tensor, self.flatten_length//4, activation_fn=self.transfer_fct_conv, trainable=trainable)
		hidden_tensor = tf.contrib.layers.fully_connected(hidden_tensor, output_size, activation_fn=self.transfer_fct_conv, trainable=trainable)
		return hidden_tensor

	def _z_2_x_conv(self, input_sample, trainable):
		# self.flatten_length = 256
		dyn_batch_size = tf.shape(input_sample)[0]
		# hidden_tensor_inv = tf.contrib.layers.fully_connected(input_sample, self.flatten_length//4, activation_fn=None)
		hidden_tensor_inv = tf.contrib.layers.fully_connected(input_sample, self.flatten_length//2, activation_fn=None, trainable=trainable)
		hidden_tensor_inv = tf.contrib.layers.fully_connected(hidden_tensor_inv, self.flatten_length, activation_fn=None, trainable=trainable)

		# W_fc = tf.Variable(xavier_init(input_sample.get_shape().as_list()[1], self.flatten_length))
		# b_fc = tf.Variable(tf.zeros([self.flatten_length], dtype=tf.float32)),
		# hidden_tensor_inv = tf.matmul(input_sample, W_fc) + b_fc

		current_input = tf.reshape(hidden_tensor_inv, [-1, 1, 1, 1, 256])
		print 'current_input', current_input.get_shape().as_list()

		def deconv_layer(current_input, kernel_shape, strides, output_shape, scope, transfer_fct, is_training, if_batch_norm, padding, trainable):
			# kernel = tf.truncated_normal(kernel_shape, dtype=tf.float32, stddev=1e-1)
			kernel = tf.Variable(
				tf.random_uniform(kernel_shape, -1.0 / (math.sqrt(kernel_shape[3]) + 10), 1.0 / (math.sqrt(kernel_shape[3]) + 10)), 
				trainable=trainable)
			biases = tf.Variable(tf.zeros(shape=[kernel_shape[-2]], dtype=tf.float32), trainable=trainable)
			if if_batch_norm:
				current_output = transfer_fct(
					self.BatchNorm(tf.reshape(
						tf.add(tf.nn.conv3d_transpose(current_input, kernel,
							output_shape, strides, padding), biases),
						output_shape),
						trainable=trainable, scope=scope
						)
					)
			else:
				current_output = transfer_fct(
					tf.reshape(
						tf.nn.bias_add(tf.nn.conv3d_transpose(current_input, kernel,
							output_shape, strides, padding), biases),
						output_shape)
					)
			return current_output
		def transfer_fct_none(x):
			return x
		current_input = deconv_layer(current_input, [5, 5, 5, 128, 256], [1, 1, 1, 1, 1], tf.pack([dyn_batch_size, 5, 5, 5, 128]), 'BN-deconv-0', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="VALID", trainable=trainable)
		print current_input.get_shape().as_list()
		current_input = deconv_layer(current_input, [5, 5, 5, 64, 128], [1, 2, 2, 2, 1], tf.pack([dyn_batch_size, 10, 10, 10, 64]), 'BN-deconv-1', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding ="SAME", trainable=trainable)
		print current_input.get_shape().as_list()
		current_input = deconv_layer(current_input, [5, 5, 5, 64, 64], [1, 1, 1, 1, 1], tf.pack([dyn_batch_size, 14, 14, 14, 64]), 'BN-deconv-2', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="VALID", trainable=trainable)
		print current_input.get_shape().as_list()
		current_input = deconv_layer(current_input, [3, 3, 3, 32, 64], [1, 2, 2, 2, 1], tf.pack([dyn_batch_size, 28, 28, 28, 32]), 'BN-deconv-3', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="SAME", trainable=trainable)
		print current_input.get_shape().as_list()
		current_input = deconv_layer(current_input, [3, 3, 3, 1, 32], [1, 1, 1, 1, 1], tf.pack([dyn_batch_size, 30, 30, 30, 1]), 'BN-deconv-4', tf.sigmoid, is_training=self.is_training, if_batch_norm=FLAGS.if_BN_out, padding="VALID", trainable=trainable)
		print current_input.get_shape().as_list()
		return current_input


	def _train_align0(self, is_training):
		_, cost, cost_recon, cost_vae, merged, \
		x0_reduce, x, x_recon, x_idx, z_mean, z_log_sigma_sq, z, steps, is_training = self.sess.run(
			(self.optimizer, self.cost, self.recon_loss0, self.latent_loss0, self.merged_summaries0, \
				self.x0_reduce, self.x0, self.x0_recon, self.gen.x0_batch_idx, self.z0_mean, self.z0_log_sigma_sq, self.z0, self.global_step, self.is_training), \
			feed_dict={self.is_training: True, self.gen.is_training: True, self.is_queue: True, self.train_net: True})
		print x0_reduce[:1]
		print z_mean[:2], np.amax(z_mean), np.amin(z_mean)
		std_diev = np.sqrt(np.exp(z_log_sigma_sq[:2]))
		print std_diev, np.amax(std_diev), np.amin(std_diev)
		return (cost, cost_recon, cost_vae, merged, x, x_recon, x_idx, z_mean, z_log_sigma_sq, z, steps)

	def _test_align0(self, is_training):
		cost, cost_recon, cost_vae, merged, x, x_recon, x_idx, z_mean, z_log_sigma_sq, z, steps, is_training = self.sess.run(
			(self.cost, self.recon_loss0, self.latent_loss0, self.merged_summaries0_test, \
				self.x0, self.x0_recon, self.gen.x0_batch_idx, self.z0_mean, self.z0_log_sigma_sq, self.z0, self.global_step, self.is_training), \
			feed_dict={self.is_training: False, self.gen.is_training: False, self.is_queue: True, self.train_net: True})
		return (cost, cost_recon, cost_vae, merged, x, x_recon, x_idx, z_mean, z_log_sigma_sq, z, steps)

	def _train_align(self, is_training):
		_, cost, cost_recon, cost_reproj, cost_euc_s, cost_euc_p, merged, \
		x, x_recon, x_proj, x_idx, x2d_rgb, x2d_gnd, z_mean, z0, z0_mean, \
		p_s, p_predict, p_l2norm, p_unit, steps, is_training = self.sess.run(
			(self.optimizer, self.cost, self.recon_loss_x, self.reproj_loss, self.euc_loss_s, self.euc_loss_p, self.merged_summaries, \
				self.x, self.x_recon, self.x_proj, self.gen.x_batch_idx, self.x2d_rgb_norm, self.x2d_gnd, self.z_mean, self.z0, self.z0_mean, 
				self.gen.p_s, self.p_predict, self.p_l2norm, self.p_unit, self.global_step, self.is_training), \
			feed_dict={self.is_training: True, self.gen.is_training: True, self.is_queue: True, self.train_net: False})
		print p_s[:5, :]
		print p_predict[:5, :]
		print p_l2norm[:5, :]
		print p_unit[:5, :]
		print np.amax(np.absolute(p_predict))
		return (cost, cost_recon, cost_reproj, cost_euc_s, cost_euc_p, merged, \
			x, x_recon, x_proj, x_idx, x2d_rgb, x2d_gnd, z_mean, z0, z0_mean, \
			p_s, p_predict, steps)

	def _test_align(self, is_training):
		cost, cost_recon, cost_reproj, cost_euc_s, cost_euc_p, merged, \
		x, x_recon, x_proj, x_idx, x2d_rgb, x2d_gnd, z_mean, z0, z0_mean, \
		p_s, p_predict, steps, is_training = self.sess.run(
			(self.cost, self.recon_loss_x, self.reproj_loss, self.euc_loss_s, self.euc_loss_p, self.merged_summaries_test, \
				self.x, self.x_recon, self.x_proj, self.gen.x_batch_idx, self.x2d_rgb_norm, self.x2d_gnd, self.z_mean, self.z0, self.z0_mean, 
				self.gen.p_s, self.p_predict, self.global_step, self.is_training), \
			feed_dict={self.is_training: False, self.gen.is_training: False, self.is_queue: True, self.train_net: False})
		return (cost, cost_recon, cost_reproj, cost_euc_s, cost_euc_p, merged, \
			x, x_recon, x_proj, x_idx, x2d_rgb, x2d_gnd, z_mean, z0, z0_mean, \
			p_s, p_predict, steps)

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

def prepare_for_training(vae):
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
		if FLAGS.if_show == False:
			global pltfig_3d_recon
			pltfig_3d_recon = plt.figure(2, figsize=(45, 15))
			plt.show(block=False)
			global pltfig_2d
			pltfig_2d = plt.figure(5, figsize=(45, 15))
			plt.show(block=False)
			global pltfig_2d_reproj
			pltfig_2d_reproj = plt.figure(6, figsize=(45, 15))
			plt.show(block=False)
			global pltfig_2d_reproj_gnd
			pltfig_2d_reproj_gnd = plt.figure(7, figsize=(45, 15))
			plt.show(block=False)
	def _count_train_set(filename):
		count = 0
		options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
		for serialized_example in tf.python_io.tf_record_iterator(filename, options):
			count = count + 1
		if FLAGS.train_net:
			print("++++++++++ Total training samples: %d; %d batches in an epoch." % (count, int(count/FLAGS.models_in_batch)))
		else:
			print("++++++++++ Total training samples: %d; %d batches in an epoch." % (count, int(count/FLAGS.batch_size)))
		return count
	global num_samples
	global num_samples_test
	print '+++++ counting samples...'
	# if FLAGS.train_net:
		# num_samples = _count_train_set(FLAGS.data_train_net0)
		# num_samples_test = _count_train_set(FLAGS.data_test_net0)
	# else:
	# num_samples = _count_train_set(FLAGS.data_train_net1)
	# num_samples_test = _count_train_set(FLAGS.data_test_net1)
	# num_samples = 313870 # airplane
	# num_samples_test = 78630
	# num_samples = 313870 # car
	# num_samples_test = 145021
	num_samples = 541773 # chair
	num_samples_test = 136027

	if FLAGS.if_disp:
		global pltfig_z0
		pltfig_z0 = plt.figure(3, figsize=(20, 8))
		plt.show(block=False)
		if FLAGS.train_net == False:
			global pltfig_p0
			pltfig_p0 = plt.figure(4, figsize=(20, 8))
			plt.show(block=False)
	global tsne_model
	tsne_model = TSNE(n_components=2, random_state=0, init='pca')
	# global gen_per_model
	# gen_per_model = FLAGS.batch_size // FLAGS.models_in_batch
	print '+++++ prepare_for_training finished.'

def train(vae):
	prepare_for_training(vae)
	try:
		while not vae.coord.should_stop():
			start_time = time.time()

			if FLAGS.train_net:
				cost, cost_recon, cost_vae, merged, x, x_recon, x_idx, z0_mean, z0_log_sigma_sq, z0, step = vae._train_align0(is_training=True)
				epoch_show = math.floor(float(step) * FLAGS.models_in_batch / float(num_samples))
				batch_show = math.floor(step - epoch_show * (num_samples / FLAGS.models_in_batch))
			else:
				cost, cost_recon, cost_reproj, cost_euc_s, cost_euc_p, merged, x, x_recon, x_proj, x_idx, x2d_rgb, x2d_gnd, z_mean, z0, z0_mean, \
				p_s, p_predict, step = vae._train_align(is_training=True)
				epoch_show = math.floor(float(step) * FLAGS.batch_size / float(num_samples))
				batch_show = math.floor(step - epoch_show * (num_samples / FLAGS.batch_size))

			## print z_mean, z0_mean, p, p_predict
			# print x_recon
			# print p_predict[:10, :]
			if FLAGS.if_disp and step % FLAGS.disp_every_step == 0 and step != 0:
				print '----- Drawing latent space of s from training batch... size of z0_mean', z0_mean.shape
				plt.figure(3)
				plt.clf()
				ax_z0 = plt.subplot(121)
				if vae.z_size != 1:
					if vae.z_size != 2:
						# z0_mean = bh_sne(np.round(np.float64(z0_mean), 2))
						z0_mean = tsne_model.fit_transform(np.round(z0_mean, 2))
				# C = range(0, z0_mean.shape[0])
				# if FLAGS.train_net == False:
				C = x_idx
				plt.scatter(z0_mean[:, 0], z0_mean[:, 1], c=C, lw=0)
				for i, txt in enumerate(C):
					ax_z0.text(z0_mean[i, 0], z0_mean[i, 1], str(C[i][0]), fontsize=7, bbox={'facecolor':'white', 'alpha':0.7, 'pad':0.0, 'lw':0})
				ax_z0.set_title('z0 space (aligned models)', fontsize=14, fontweight='bold')

				# if FLAGS.train_net == False:
				# 	print '----- Drawing latent space of s from training batch... size of z_mean', z_mean.shape
				# 	ax_z = plt.subplot(122)
				# 	if vae.z_size != 1:
				# 		if vae.z_size != 2:
				# 			# z_mean = bh_sne(np.float64(z_mean))
				# 			z_mean = tsne_model.fit_transform(z_mean)
				# 	# C = range(0, z0_mean.shape[0])
				# 	C = x_idx
				# 	plt.scatter(z_mean[:, 0], z_mean[:, 1], c=C, lw=0)
				# 	# plt.colorbar()
				# 	for i, txt in enumerate(x_idx):
				# 		plt.text(z_mean[i, 0], z_mean[i, 1], str(C[i][0]), fontsize=7, bbox={'facecolor':'white', 'alpha':0.7, 'pad':0.0, 'lw':0})
				# 	ax_z.set_title('z space (all models)', fontsize=14, fontweight='bold')
				# pltfig_z0.canvas.draw()
				# pltfig_z0.savefig(params['summary_folder']+'/%d-pltfig_z.png'%step)

				if FLAGS.train_net == False:
					print '----- Drawing latent space of p from training batch...'
					plt.figure(4)
					plt.clf()
					ax_p0 = plt.subplot(121, projection='3d')
					ax_p0.scatter(p_s[:, 0], p_s[:, 1], p_s[:, 2], lw=0)
					ax_p0.set_title('ground truth pose', fontsize=14, fontweight='bold')

					ax_p = plt.subplot(122, projection='3d')
					ax_p.scatter(p_predict[:, 0], p_predict[:, 1], p_predict[:, 2], lw=0)
					ax_p.set_title('estimated pose', fontsize=14, fontweight='bold')
					pltfig_p0.canvas.draw()
					pltfig_p0.savefig(params['summary_folder']+'/%d-pltfig_p.png'%step)


			if FLAGS.if_summary:
				vae.train_writer.add_summary(merged, step)
				vae.train_writer.flush()
				if FLAGS.train_net:
					print "STEP", '%03d' % (step), "Epo", '%03d' % (epoch_show), "ba", '%03d' % (batch_show), \
					"cost =", "%.4f = %.4f + %.4f" % (\
						cost, cost_recon * FLAGS.reweight_recon, cost_vae * FLAGS.reweight_vae), \
					"-- recon = %.4f, vae = %.4f" % (cost_recon / vae.x_size, cost_vae)
				else:
					print "STEP", '%03d' % (step), "Epo", '%03d' % (epoch_show), "ba", '%03d' % (batch_show), \
					"cost =", "%.4f = %.4f + %.4f + %.4f + %.4f" % (\
						cost, cost_recon * FLAGS.reweight_recon, cost_reproj * FLAGS.reweight_reproj, cost_euc_s * FLAGS.reweight_euc_s, cost_euc_p * FLAGS.reweight_euc_p), \
					"-- recon = %.4f, reproj = %.4f, euc_s = %.4f, euc_p = %.4f" % (cost_recon / vae.x_size, cost_reproj / 900., cost_euc_s, cost_euc_p)

			if FLAGS.if_save and step != 0 and step % FLAGS.save_every_step == 0:
				save_vae(vae, step, epoch_show, batch_show)

			if FLAGS.if_test and step % FLAGS.test_every_step == 0 and step != 0:
				if FLAGS.train_net:
					cost, cost_recon, cost_vae, merged, x, x_recon, x_idx, z_mean, z_log_sigma_sq, z, step = vae._test_align0(is_training=False)
				else:
					cost, cost_recon, cost_reproj, cost_euc_s, cost_euc_p, merged, \
					x, x_recon, x_proj, x_idx, x2d_rgb, x2d_gnd, z_mean, z0, z0_mean, \
					p_s, p_predict, step = vae._test_align(is_training=False)
				if FLAGS.if_summary:
					vae.train_writer.add_summary(merged, step)
					vae.train_writer.flush()
					if FLAGS.train_net:
						print "TESTING net 0... "\
						"cost =", "%.4f = %.4f + %.4f" % (\
							cost, cost_recon * FLAGS.reweight_recon, cost_vae * FLAGS.reweight_vae), \
						"-- recon = %.4f, vae = %.4f" % (cost_recon / vae.x_size, cost_vae)
					else:
						print "TESTING net 1... "\
						"cost =", "%.4f = %.4f + %.4f + %.4f + %.4f" % (\
							cost, cost_recon * FLAGS.reweight_recon, cost_reproj * FLAGS.reweight_reproj, cost_euc_s * FLAGS.reweight_euc_s, cost_euc_p * FLAGS.reweight_euc_p), \
						"-- recon = %.4f, reproj = %.4f, euc_s = %.4f, euc_p = %.4f" % (cost_recon / vae.x_size, cost_reproj / 900., cost_euc_s, cost_euc_p)
				if FLAGS.if_draw and step % FLAGS.draw_every == 0:
					print 'Drawing reconstructed sample from testing batch...'
					plt.figure(1)
					for test_idx in range(15):
						im = draw_sample(figM, x[test_idx].reshape((30, 30, 30)), ms)
						plt.subplot(3, 5, test_idx+1)
						plt.imshow(im)
						plt.axis('off')
					pltfig_3d.suptitle('Target models at step %s of %s'%(step, FLAGS.folder_name_save_to), fontsize=20, fontweight='bold')
					pltfig_3d.canvas.draw()
					pltfig_3d.savefig(params['summary_folder']+'/%d-pltfig_3d_gnd.png'%step)
					plt.figure(2)
					for test_idx in range(15):
						im = draw_sample(figM, x_recon[test_idx].reshape((30, 30, 30)), ms)
						plt.subplot(3, 5, test_idx+1)
						plt.imshow(im)
						plt.axis('off')
					pltfig_3d_recon.suptitle('Reconstructed models at step %s of %s'%(step, FLAGS.folder_name_save_to), fontsize=20, fontweight='bold')
					pltfig_3d_recon.canvas.draw()
					pltfig_3d_recon.savefig(params['summary_folder']+'/%d-pltfig_3d_recon.png'%step)
					if FLAGS.train_net == False:
						plt.figure(5)
						for test_idx in range(15):
							plt.subplot(3, 5, test_idx+1)
							plt.imshow(((x2d_rgb[test_idx] + 0.5) * 255.).astype(np.uint8))
							plt.axis('off')
						pltfig_2d.suptitle('Target RGB image at step %s of %s'%(step, FLAGS.folder_name_save_to), fontsize=20, fontweight='bold')
						pltfig_2d.canvas.draw()
						pltfig_2d.savefig(params['summary_folder']+'/%d-pltfig_rgb.png'%step)
						plt.figure(6)
						for test_idx in range(15):
							im = draw_sample(figM_2d, x_proj[test_idx].reshape((30, 30, 1)), ms_2d, camera_view=True)
							plt.subplot(3, 5, test_idx+1)
							plt.imshow(im)
							plt.axis('off')
						pltfig_2d_reproj.suptitle('Reprojection at step %s of %s'%(step, FLAGS.folder_name_save_to), fontsize=20, fontweight='bold')
						pltfig_2d_reproj.canvas.draw()
						pltfig_2d_reproj.savefig(params['summary_folder']+'/%d-pltfig_2d_proj.png'%step)
						plt.figure(7)
						for test_idx in range(15):
							im = draw_sample(figM_2d, x2d_gnd[test_idx].reshape((30, 30, 1)), ms_2d, camera_view=True)
							plt.subplot(3, 5, test_idx+1)
							plt.imshow(im)
							plt.axis('off')
						pltfig_2d_reproj_gnd.suptitle('Gnd truth projection at step %s of %s'%(step, FLAGS.folder_name_save_to), fontsize=20, fontweight='bold')
						pltfig_2d_reproj_gnd.canvas.draw()
						pltfig_2d_reproj_gnd.savefig(params['summary_folder']+'/%d-pltfig_2d_gnd.png'%step)
			end_time = time.time()
			elapsed = end_time - start_time
			print "--- Time %f seconds."%elapsed
	except tf.errors.OutOfRangeError:
		print('Done training.')
	finally:
		# When done, ask the threads to stop.
		vae.coord.request_stop()
	# Wait for threads to finish.
	vae.coord.join(vae.threads)
	vae.sess.close()

def save_vae(vae, step, epoch, batch):
	if FLAGS.train_net:
		net_folder = vae.params['model_folder'] + '/net0'
	else:
		net_folder = vae.params['model_folder'] + '/net1'
	save_path = vae.saver_for_resume.save(vae.sess, \
		net_folder + '/%s-step%d-epoch%d-batch%d.ckpt' % (params['save_name'], step, epoch, batch), \
		global_step=step)
	print("-----> Model saved to file: %s; step = %d" % (save_path, step))

def restore_vae(vae, params):
	# if FLAGS.train_netvae.saver_for_restore_encoder0.restore(vae.sess, latest_checkpoint)
	if FLAGS.train_net:
		net_folder = vae.params['model_folder_restore'] + '/net0'
	else:
		net_folder = vae.params['model_folder_restore'] + '/net1'
	net0_folder = vae.params['model_folder_net0_restore'] + '/net0'

	if FLAGS.restore_encoder0:
		if "ckpt" not in net0_folder:
			latest_checkpoint = tf.train.latest_checkpoint(net0_folder)
		else:
			latest_checkpoint = FLAGS.folder_name_net0_restore_from
		print "+++++ Loading net0 from: %s" % (latest_checkpoint)
		vae.restorer_encoder0.restore(vae.sess, latest_checkpoint)
		# if FLAGS.resume_from == 1:
		# 	latest_checkpoint = tf.train.latest_checkpoint(net_folder + '/')
		# 	print "+++++ Loading excluding net0 from: %s" % (latest_checkpoint)
		# 	vae.restorer_exclude_encoder0.restore(vae.sess, latest_checkpoint)
	else:
		if FLAGS.resume_from == 1:
			if "ckpt" not in vae.params['model_folder_restore']:
				latest_checkpoint = tf.train.latest_checkpoint(vae.params['model_folder_restore'] + '/')
			else:
				latest_checkpoint = FLAGS.folder_name_restore_from
			print "+++++ Resuming: Model restored from folder: %s file: %s" % (vae.params['model_folder_restore'], latest_checkpoint)
			vae.restorer.restore(vae.sess, latest_checkpoint)
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
	raw_input("Press Enter to continue...")
	os.system('rm -rf %s/ %s/' % (params["summary_folder"], params["model_folder"]))

if FLAGS.if_alex == True:
	global alexnet_data
	alexnet_data = np.load("./alexnet/bvlc_alexnet.npy").item()
	print '===== Alexnet loaded.'
vae = VariationalAutoencoder(params)

global net_folder
if FLAGS.train_net:
	net_folder = vae.params['model_folder'] + '/net0'
else:
	net_folder = vae.params['model_folder'] + '/net1'
if FLAGS.resume_from == 1 and FLAGS.folder_name_restore_from == "":
	print "+++++ Setting restore folder name to the saving folder name..."
	params["model_folder_restore"] = net_folder
print '===== FLAGS.restore_encoder0:', FLAGS.restore_encoder0
print '===== FLAGS.folder_name_restore_from:', FLAGS.folder_name_restore_from
print '===== FLAGS.folder_name_net0_restore_from:', FLAGS.folder_name_net0_restore_from
if FLAGS.resume_from != -1 or FLAGS.restore_encoder0 == True:
	restore_vae(vae, params)

def show_1(vae):
	prepare_for_training(vae)
	try:
		while not vae.coord.should_stop():
			try:
				image_idx = 0

				view, R_s_flat, angleaxis_batch, x, x3d_gnd_rot, x_proj, x0, x_recon, x2d_rgb, x2d_gnd_proj, x2d_gnd, p_s, step, is_training = vae.sess.run(
				(vae.gen.view_batch, vae.gen.R_s_flat, vae.gen.angleaxis_batch, vae.x, vae.x3d_gnd_rot, vae.x_proj, vae.x0, vae.x_recon, vae.x2d_rgb_norm, vae.x2d_gnd_proj, vae.x2d_gnd, vae.p_s, vae.global_step, vae.is_training), \
				feed_dict={vae.is_training: True, vae.gen.is_training: True, vae.is_queue: True, vae.train_net: False})

				epoch_show = math.floor(float(step) * FLAGS.batch_size / float(num_samples))
				batch_show = math.floor(step - epoch_show * (num_samples / FLAGS.batch_size))
				print 'TESTING... Drawing reconstructed sample from testing batch... %d'%step
				# print R_s_flat[0]
				print p_s
				print angleaxis_batch
				print view

				# def do_with_row(a):
				# 	tw1 = np.array([[0.], [1.], [0.]])
				# 	def AxisAngle2Mat(axis, angle):
				# 		# return np.array([[0., -tw[2], tw[1]], [tw[2], 0, -tw[0]], [-tw[1], tw[0], 0.]])
				# 		c = np.cos(angle)
				# 		s = np.sin(angle)
				# 		t = 1 - c
				# 		axis = axis / np.linalg.norm(axis)
				# 		x = axis[0]
				# 		y = axis[1]
				# 		z = axis[2]
				# 		return np.array([[t*x*x+c, t*x*y-z*s, t*x*z+y*s], \
				# 			[t*x*y+z*s, t*y*y+c, t*y*z-x*s], \
				# 			[t*x*z-y*s, t*y*z+x*s, t*z*z+c]]).reshape((3, 3))
				# 	# R1 = scipy.linalg.expm(hat(tw1))
				# 	R1 = AxisAngle2Mat(tw1, a[0])
				# 	# print a[0].shape, R1.shape
				# 	new_x_azis = np.matmul(R1, np.array([[0.], [0.], [1.]]));
				# 	# tw2 = new_x_azis * a[1]
				# 	# R2 = scipy.linalg.expm(hat(tw2))
				# 	R2 = AxisAngle2Mat(new_x_azis, a[1])
				# 	R = np.matmul(R2, R1)
				# 	R_homo = np.row_stack((np.column_stack((R, np.zeros((3, 1)))), np.array([0., 0., 0., 1.])))
				# 	return R_homo.flatten()
				# Rs_flat_batch = do_with_row(az_el_batch[0] / 180.0 * np.pi)
				# print Rs_flat_batch

				for x_1, x_proj_1, x_recon_1, x0_1, x3d_gnd_rot_1, x2d_gnd_proj_1, x2d_gnd_1, x2d_rgb_1, p in zip(x, x_proj, x_recon, x0, x3d_gnd_rot, x2d_gnd_proj, x2d_gnd, x2d_rgb, p_s):
					plt.figure(1)
					
					im = draw_sample(figM, x_1.reshape((30, 30, 30)), ms)
					plt.subplot(2, 3, 2)
					plt.imshow(im)
					plt.axis('off')
					plt.title('voxelized model 3D (gnd X) with canonical camera', fontsize=10)

					im = draw_sample(figM, x3d_gnd_rot_1.reshape((30, 30, 30)), ms)
					plt.subplot(2, 3, 3)
					plt.imshow(im)
					plt.axis('off')
					plt.title('voxelized model 3D (gnd x0 + gnd p) with canonical camera', fontsize=10)

					# im = draw_sample(figM, x_recon_1.reshape((30, 30, 30)), ms)
					# plt.subplot(2, 3, 5)
					# plt.imshow(im)
					# plt.axis('off')
					# plt.title('voxelized model (recon) (not aligned) with canonical camera', fontsize=10)

					# im = draw_sample(figM_2d, np.asarray(x2d_gnd_proj_1>0.5, dtype=float).reshape((30, 30, 1)), ms_2d, camera_view=True)
					# plt.subplot(2, 3, 5)
					# plt.imshow(im)
					# plt.axis('off')
					# plt.title('voxelized model 2D proj (gnd x0 + gnd p, thres) from camera view', fontsize=10)

					im = draw_sample(figM_2d, x2d_gnd_1.reshape((30, 30, 1)), ms_2d, camera_view=True)
					plt.subplot(2, 3, 4)
					plt.imshow(im)
					plt.axis('off')
					plt.title('voxelized model 2D proj (gnd seg) from camera view', fontsize=10)

					# print 'x_proj_1.shape', x_proj_1.shape
					# im = draw_sample(figM_2d, x_proj_1.reshape((30, 30, 1)), ms_2d, camera_view=True)
					# plt.subplot(2, 3, 3)
					# plt.imshow(im)
					# plt.axis('off')
					# plt.title('voxelized model 2D proj (recon) from camera view', fontsize=10)

					im = draw_sample(figM_2d, x2d_gnd_proj_1.reshape((30, 30, 1)), ms_2d, camera_view=True)
					plt.subplot(2, 3, 6)
					plt.imshow(im)
					plt.axis('off')
					plt.title('voxelized model 2D proj (x0 + gnd p) from camera view', fontsize=10)
					
					im0 = draw_sample(figM, x0_1.reshape((30, 30, 30)), ms, p = p)
					plt.subplot(2, 3, 1)
					plt.imshow(im0)
					plt.axis('off')
					plt.title('voxelized model', fontsize=10)

					ax = plt.subplot(2, 3, 5)
					plt.imshow(((x2d_rgb_1 + 0.5) * 255.).astype(np.uint8))
					# (x2d_rgb[test_idx] + 0.5) * 255.)
					plt.axis('off')
					plt.title('RGB image of voxelized model from camera view', fontsize=10)

					# im = draw_sample(figM_2d, np.asarray(x_proj_1>0.5, dtype=float).reshape((30, 30, 1)), ms_2d, camera_view=True)
					# plt.subplot(2, 3, 2)
					# plt.imshow(im)
					# plt.axis('off')
					# plt.title('voxelized model 2D proj (recon thres) from camera view', fontsize=10)

					pltfig_3d.canvas.draw()
					# pltfig_3d.savefig('/home/rz1/Documents/tensorflowPlay/Imgs/camera-demo/camera-%d.png'%image_idx)
					
					image_idx = image_idx + 1
					time.sleep(3)  # do something here
					print '.',
			
			except KeyboardInterrupt:
				print '\nPausing...  (Hit ENTER to continue, type quit to exit.)'
				try:
					response = raw_input()
					if response == 'quit':
						break
					print 'Resuming...'
				except KeyboardInterrupt:
					print 'Resuming...'
					continue
	except tf.errors.OutOfRangeError:
		print('Done training.')
	finally:
		# When done, ask the threads to stop.
		vae.coord.request_stop()
	# Wait for threads to finish.
	vae.coord.join(vae.threads)
	vae.sess.close()

# def show_0(vae):
# 	prepare_for_training(vae)
# 	try:
# 		while not vae.coord.should_stop():
# 			try:
# 				# cost, cost_recon, cost_vae, merged, x, x_recon, z0_mean, z0_log_sigma_sq, z0, step = vae._train_align0(is_training=True)
# 				x1, x_recon1, z_mean1, _, _, step, is_training = vae.sess.run(
# 				(vae.x0, vae.x0_recon, vae.z0_mean, vae.z0_log_sigma_sq, vae.z0, vae.global_step, vae.is_training), \
# 				feed_dict={vae.is_training: True, vae.gen.is_training: True, vae.is_queue: True, vae.train_net: True})

# 				x2, x_recon2, z_mean2, _, _, step, is_training = vae.sess.run(
# 				(vae.x0, vae.x0_recon, vae.z0_mean, vae.z0_log_sigma_sq, vae.z0, vae.global_step, vae.is_training), \
# 				feed_dict={vae.is_training: True, vae.gen.is_training: True, vae.is_queue: True, vae.train_net: True})

# 				image_idx = 0
# 				x_recon_list = []

# 				for lam in range(0, 15):
# 					z = z_mean2 * lam * (1. / 15.) + z_mean1 * (1 - lam * (1. / 15.)) 
# 					x_recon = vae.sess.run(
# 					(vae.x0_recon), \
# 					feed_dict={vae.z0: z, vae.is_training: True, vae.gen.is_training: True, vae.is_queue: True, vae.train_net: True})
# 					x_recon_list.append(x_recon)

# 				num_samples = x1.shape[0]

# 				for i in range(num_samples):
# 					x1_1 = x1[i]
# 					x2_1 = x2[i]
# 					im1 = draw_sample(x1_1.reshape((30, 30, 30)), ms)
# 					im2 = draw_sample(x2_1.reshape((30, 30, 30)), ms)
# 					for j in range(15):
# 						print 'TESTING... Drawing reconstructed sample from testing batch... %d lam=%d'%(i, j)
# 						plt.figure(1)

# 						plt.subplot(1, 3, 1)
# 						plt.imshow(im1)
# 						plt.axis('off')
# 						plt.title('input 1')
						
# 						if j == 0:
# 							im = im1
# 						elif j == 14:
# 							im = im2
# 						else:
# 							x_recon_1 = x_recon_list[j-1][i]
# 							im = draw_sample(x_recon_1.reshape((30, 30, 30)), ms)
# 						plt.subplot(1, 3, 2)
# 						plt.imshow(im)
# 						plt.axis('off')
# 						if j != 0 and j != 14:
# 							plt.title('interpolation %d/15'%(j))
# 						else:
# 							plt.title('')

# 						plt.subplot(1, 3, 3)
# 						plt.imshow(im2)
# 						plt.axis('off')
# 						plt.title('input 2')

# 						pltfig_3d.canvas.draw()
# 						pltfig_3d.savefig('/home/rz1/Documents/tensorflowPlay/Imgs/inter-demo/inter-%d-%d.png'%(image_idx, j))
# 						# time.sleep(3)  # do something here
# 						print '.',
# 					image_idx = image_idx + 1
			
# 			except KeyboardInterrupt:
# 				print '\nPausing...  (Hit ENTER to continue, type quit to exit.)'
# 				try:
# 					response = raw_input()
# 					if response == 'quit':
# 						break
# 					print 'Resuming...'
# 				except KeyboardInterrupt:
# 					print 'Resuming...'
# 					continue

# 	except tf.errors.OutOfRangeError:
# 		print('Done training.')
# 	finally:
# 		# When done, ask the threads to stop.
# 		vae.coord.request_stop()
# 	# Wait for threads to finish.
# 	vae.coord.join(vae.threads)
# 	vae.sess.close()

# # def generate_rand_az_el(gen_size):
# # 	az_col = 30. * np.random.randint(low=0, high=12, size=(gen_size, 1))
# # 	el_col = 20. * (np.random.randint(low=0, high=6, size=(gen_size, 1)) - 3.)
# # 	return np.hstack((az_col, el_col))

# # def batch_az_el_2_point(az_el):
# # 	az_col = az_el[:, 0:1]
# # 	el_col = az_el[:, 1:2]
# # 	sin_az = np.sin(az_col)
# # 	cos_az = np.cos(az_col)
# # 	sin_el = np.sin(el_col)
# # 	cos_el = np.cos(el_col)
# # 	return np.hstack((np.multiply(cos_el, sin_az), np.multiply(cos_el, cos_az), sin_el))

# # def batch_az_el_2_R_flat(az_el_batch):
# # 	def do_with_row(a):
# # 		tw1 = np.array([0., a[0], 0.])
# # 		def hat(tw):
# # 			return np.array([[0., -tw[2], tw[1]], [tw[2], 0, -tw[0]], [-tw[1], tw[0], 0.]])
# # 		R1 = scipy.linalg.expm(hat(tw1))
# # 		new_x_azis = np.matmul(R1, np.array([0., 0., 1.]));
# # 		tw2 = new_x_azis * a[1]
# # 		R2 = scipy.linalg.expm(hat(tw2))
# # 		R = np.matmul(R2, R1)
# # 		R_homo = np.row_stack((np.column_stack((R, np.zeros((3, 1)))), np.array([0., 0., 0., 1.])))
# # 		return R_homo.flatten()
# # 	Rs_flat_batch = np.apply_along_axis(do_with_row, 1, az_el_batch)
# # 	return Rs_flat_batch

# # def generate_batch_rand_az_el(batch_size, models_in_batch):
# # 	idx_s = np.random.randint(low=0, high=models_in_batch, size=(batch_size, 1), dtype=int)
# # 	az_el_s = generate_rand_az_el(batch_size)
# # 	p_s = batch_az_el_2_point(az_el_s / 180.0 * np.pi)
# # 	Rs_flat_s = batch_az_el_2_R_flat(az_el_s / 180.0 * np.pi)
# # 	return (idx_s, p_s, Rs_flat_s, az_el_s)

if FLAGS.if_show == False:
	print("-----> Starting Training")
	print 'if_unlock_decoder0', FLAGS.if_unlock_decoder0
	print [var.op.name for var in tf.trainable_variables()]
	train(vae)
else:
	print("-----> Starting Showing")
	if FLAGS.train_net:
		show_0(vae)
	else:
		show_1(vae)	