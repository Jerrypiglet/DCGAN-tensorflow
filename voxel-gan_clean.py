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
flags.DEFINE_string("data_test_net0", "", "load test data from")
flags.DEFINE_string("data_train_net1", "", "load train data from")
flags.DEFINE_string("data_test_net1", "", "load test data from")
flags.DEFINE_boolean("if_BN", False, "if batch_norm")
flags.DEFINE_boolean("if_BN_out", False, "if batch_norm for x output layer")
flags.DEFINE_boolean("if_show", True, "if show mode")
flags.DEFINE_boolean("if_unlock_decoder0", False, "if unlock decoder0")
flags.DEFINE_boolean("if_gndS", False, "if_gndS")
flags.DEFINE_boolean("if_gndP", True, "if_gndP")
flags.DEFINE_boolean("if_p_trainable", True, "if recog p is trainable")
flags.DEFINE_boolean("if_s_trainable", True, "if recog s is trainable")
global FLAGS
FLAGS = flags.FLAGS

class VariationalAutoencoder(object):
	def __init__(self, params, transfer_fct=tf.sigmoid):
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

		self.restorer_encoder0 = tf.train.Saver(slim.get_variables_to_restore(include=["encoder0","decoder0","step"]), max_to_keep=2, write_version=tf.train.SaverDef.V2)
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
			self.x0 = tf.cond(self.train_net, lambda: self.gen.x0_batch, lambda: self.gen.x_gnd_batch) # aligned models
			self.x0 = tf.transpose(self.x0, perm=[0, 2, 3, 1, 4])
			self.x0_flatten = tf.reshape(self.x0, [-1, 27000])
			self.dyn_batch_size_x0 = tf.shape(self.x0)[0]
			with tf.variable_scope("encoder0"):
				with slim.arg_scope([slim.fully_connected], trainable=FLAGS.train_net):
					self.x0_reduce = self._x_2_z_conv(self.x0_flatten, trainable=FLAGS.train_net)
					self.z0_mean = self._fcAfter_x_2_z_conv(self.x0_reduce, self.z_size, trainable=FLAGS.train_net)
					self.z0_mean = tf.clip_by_value(self.z0_mean, -50., 50.)
					self.z0_log_sigma_sq = self._fcAfter_x_2_z_conv(self.x0_reduce, self.z_size, trainable=FLAGS.train_net)
					self.z0_log_sigma_sq = tf.clip_by_value(self.z0_log_sigma_sq, -50., 10.)
					eps0 = tf.random_normal(tf.pack([self.dyn_batch_size_x0, self.z_size]))
					self.z0 = self.z0_mean + eps0 * tf.sqrt(tf.exp(self.z0_log_sigma_sq))

		## Define decoder0
		with tf.device('/gpu:2'):
			# z0_for_recon = tf.cond(self.train_net, lambda: self.z0, lambda: self.z)
			z0_for_recon = self.z0
			with tf.variable_scope("decoder0"):
				with slim.arg_scope([slim.fully_connected], trainable=(FLAGS.train_net or FLAGS.if_unlock_decoder0)):
					self.x0_recon = self._z_2_x_conv(z0_for_recon, trainable=(FLAGS.train_net or FLAGS.if_unlock_decoder0))	

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

			self.cost = FLAGS.reweight_recon * self.recon_loss0 + FLAGS.reweight_vae * self.latent_loss0
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, colocate_gradients_with_ops=True)
			
			summary_loss0 = tf.scalar_summary('loss0/loss', self.cost)
			summary_loss_recon0 = tf.scalar_summary('loss0/rec_loss', self.recon_loss0 / self.x_size)
			summary_loss_latent0 = tf.scalar_summary('loss0/vae_loss', self.latent_loss0)
			summaries0 = [summary_loss0, summary_loss_recon0, summary_loss_latent0]
			self.merged_summaries0 = tf.merge_summary(summaries0)

			summary_loss0_test = tf.scalar_summary('loss0_test/loss', self.cost)
			summary_loss0_recon_test = tf.scalar_summary('loss0_test/rec_loss', self.recon_loss0 / self.x_size)
			summary_loss0_latent_test = tf.scalar_summary('loss0_test/vae_loss', self.latent_loss0)
			summaries0_test = [summary_loss0_test, summary_loss0_recon_test, summary_loss0_latent_test]
			self.merged_summaries0_test = tf.merge_summary(summaries0_test)

	def BatchNorm(self, inputT, trainable, scope=None):
		if trainable:
			print '########### BN trainable!!!'
		return tflearn.layers.normalization.batch_normalization(inputT, trainable=trainable)

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
		# if FLAGS.train_net == False:
		# 	global pltfig_p0
		# 	pltfig_p0 = plt.figure(4, figsize=(20, 8))
		# 	plt.show(block=False)
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
			if FLAGS.if_disp and step % FLAGS.disp_every_step == 0 and step != 0:
				print '----- Drawing latent space of s from training batch... size of z0_mean', z0_mean.shape
				plt.figure(3)
				plt.clf()
				ax_z0 = plt.subplot(121)
				if vae.z_size != 1:
					if vae.z_size != 2:
						# z0_mean = bh_sne(np.round(np.float64(z0_mean), 2))
						z0_mean = tsne_model.fit_transform(np.round(z0_mean, 2))
				C = x_idx
				plt.scatter(z0_mean[:, 0], z0_mean[:, 1], c=C, lw=0)
				for i, txt in enumerate(C):
					ax_z0.text(z0_mean[i, 0], z0_mean[i, 1], str(C[i][0]), fontsize=7, bbox={'facecolor':'white', 'alpha':0.7, 'pad':0.0, 'lw':0})
				ax_z0.set_title('z0 space (aligned models)', fontsize=14, fontweight='bold')

			if FLAGS.if_summary:
				vae.train_writer.add_summary(merged, step)
				vae.train_writer.flush()
				if FLAGS.train_net:
					print "STEP", '%03d' % (step), "Epo", '%03d' % (epoch_show), "ba", '%03d' % (batch_show), \
					"cost =", "%.4f = %.4f + %.4f" % (\
						cost, cost_recon * FLAGS.reweight_recon, cost_vae * FLAGS.reweight_vae), \
					"-- recon = %.4f, vae = %.4f" % (cost_recon / vae.x_size, cost_vae)

			if FLAGS.if_save and step != 0 and step % FLAGS.save_every_step == 0:
				save_vae(vae, step, epoch_show, batch_show)

			if FLAGS.if_test and step % FLAGS.test_every_step == 0 and step != 0:
				if FLAGS.train_net:
					cost, cost_recon, cost_vae, merged, x, x_recon, x_idx, z_mean, z_log_sigma_sq, z, step = vae._test_align0(is_training=False)
				if FLAGS.if_summary:
					vae.train_writer.add_summary(merged, step)
					vae.train_writer.flush()
					if FLAGS.train_net:
						print "TESTING net 0... "\
						"cost =", "%.4f = %.4f + %.4f" % (\
							cost, cost_recon * FLAGS.reweight_recon, cost_vae * FLAGS.reweight_vae), \
						"-- recon = %.4f, vae = %.4f" % (cost_recon / vae.x_size, cost_vae)
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
	# else:
	# 	net_folder = vae.params['model_folder'] + '/net1'
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

global alexnet_data
alexnet_data = np.load("./alexnet/bvlc_alexnet.npy").item()
print '===== Alexnet loaded.'
vae = VariationalAutoencoder(params)

global net_folder
if FLAGS.train_net:
	net_folder = vae.params['model_folder'] + '/net0'
# else:
# 	net_folder = vae.params['model_folder'] + '/net1'
if FLAGS.resume_from == 1 and FLAGS.folder_name_restore_from == "":
	print "+++++ Setting restore folder name to the saving folder name..."
	params["model_folder_restore"] = net_folder
print '===== FLAGS.restore_encoder0:', FLAGS.restore_encoder0
print '===== FLAGS.folder_name_restore_from:', FLAGS.folder_name_restore_from
print '===== FLAGS.folder_name_net0_restore_from:', FLAGS.folder_name_net0_restore_from
if FLAGS.resume_from != -1 or FLAGS.restore_encoder0 == True:
	restore_vae(vae, params)

if FLAGS.if_show == False:
	print("-----> Starting Training")
	print 'if_unlock_decoder0', FLAGS.if_unlock_decoder0
	print [var.op.name for var in tf.trainable_variables()]
	train(vae)
else:
	print("-----> Starting Showing")
	if FLAGS.train_net:
		show_0(vae)
	# else:
	# 	show_1(vae)	