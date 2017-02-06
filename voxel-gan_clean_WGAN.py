import numpy as np
import random
import tensorflow as tf
import math
# from libs.activations import lrelu
import tflearn
from sklearn.manifold import TSNE
from tsne import bh_sne
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as ly
import tensorpack
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
flags.DEFINE_float("learning_rate", 5e-5, "learning rate")
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
flags.DEFINE_boolean("if_BN_G", True, "if batch_norm")
flags.DEFINE_boolean("if_BN_D", True, "if batch_norm")
flags.DEFINE_boolean("if_BN_out", False, "if batch_norm for x output layer")
flags.DEFINE_boolean("if_show", True, "if show mode")
flags.DEFINE_boolean("if_unlock_decoder0", False, "if unlock decoder0")
# flags.DEFINE_boolean("if_gndS", False, "if_gndS")
# flags.DEFINE_boolean("if_gndP", True, "if_gndP")
# flags.DEFINE_boolean("if_p_trainable", True, "if recog p is trainable")
# flags.DEFINE_boolean("if_s_trainable", True, "if recog s is trainable")
global FLAGS
FLAGS = flags.FLAGS

def lrelu(x, leak=0.3, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

class VariationalAutoencoder(object):
	def __init__(self, params):
		self.params = params
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
		self.sess.run(tf.global_variables_initializer())
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
			self.x0 = 2 * self.gen.x0_batch - 1.# aligned models
			self.x0 = tf.transpose(self.x0, perm=[0, 2, 3, 1, 4])
			self.dyn_batch_size_x0 = tf.shape(self.x0)[0]

		with tf.device('/gpu:0'):
			self.z = tf.placeholder(tf.float32, [None, self.z_size], name='z')
			self.z_sum = tf.histogram_summary("z", self.z)

			# self.flatten_length = 2048

			self.D_logits = self._discriminator(self.x0)
			self.G, self.G_noGate = self._generator(self.z)
			self.G_sum = tf.histogram_summary("G", tf.reshape(self.G_noGate, [-1]))

		with tf.device('/gpu:0'):
			self.D_logits_ = self._discriminator(self.G, reuse=True)

	def _create_loss_optimizer(self):
		self.d_loss_real = tf.reduce_mean(
			- self.D_logits)
		self.d_loss_fake = tf.reduce_mean(
			self.D_logits_)
		self.d_loss = self.d_loss_real + self.d_loss_fake
		self.g_loss = tf.reduce_mean(
			- self.D_logits_)

		self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
		self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)
													
		self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
		self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

		t_vars = tf.trainable_variables()
		self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
		self.g_vars = [var for var in t_vars if 'generator' in var.name]

		print [var.name for var in t_vars if 'discriminator' in var.name]
		print [var.name for var in t_vars if 'generator' in var.name]

		learning_rate_ger = FLAGS.learning_rate
		learning_rate_dis = FLAGS.learning_rate
		# update Citers times of critic in one iter(unless i < 25 or i % 500 == 0, i is iterstep)
		self.Citers = 5
		# the upper bound and lower bound of parameters in critic
		clamp_lower = -0.01
		clamp_upper = 0.01
		# whether to use adam for parameter update, if the flag is set False, use tf.train.RMSPropOptimizer
		# as recommended in paper
		is_adam = False

		counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
		self.opt_g = ly.optimize_loss(loss=self.g_loss, learning_rate=learning_rate_ger,
						optimizer=tf.train.AdamOptimizer if is_adam is True else tf.train.RMSPropOptimizer, 
						variables=self.g_vars, global_step=counter_g,
						summaries = 'gradient_norm')

		counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
		self.opt_c = ly.optimize_loss(loss=self.d_loss, learning_rate=learning_rate_dis,
						optimizer=tf.train.AdamOptimizer if is_adam is True else tf.train.RMSPropOptimizer, 
						variables=self.d_vars, global_step=counter_c,
						summaries = 'gradient_norm')
		clipped_var_c = [tf.assign(var, tf.clip_by_value(var, clamp_lower, clamp_upper)) for var in self.d_vars]
		# merge the clip operations on critic variables
		with tf.control_dependencies([self.opt_c]):
			self.opt_c = tf.tuple(clipped_var_c)

		self.merged_summary = tf.merge_summary([self.g_loss_sum, self.d_loss_sum, self.z_sum, self.G_sum, self.d_loss_fake_sum, self.d_loss_real_sum])

	def BatchNorm(self, inputT, trainable=True, scope=None, reuse=None):
		if trainable:
			print '########### BN trainable!!!'
		return tflearn.layers.normalization.batch_normalization(inputT, trainable=trainable, scope=scope, reuse=False)
		# return tensorpack.models.BatchNorm(inputT)

	def _discriminator(self, input_tensor, trainable=True, reuse=False):
		with tf.variable_scope("discriminator") as scope:
			if reuse:
				scope.reuse_variables()

			input_shape=[None, 27000]
			x_tensor = tf.reshape(input_tensor, [-1, 30, 30, 30, 1])
			current_input = x_tensor

			def conv_layer(current_input, kernel_shape, strides, scope, transfer_fct, is_training, if_batch_norm, padding, trainable, reuse=False):
				# kernel = tf.truncated_normal(kernel_shape, dtype=tf.float32, stddev=1e-3)
				# kernel = tf.Variable(
				# 	tf.random_uniform(kernel_shape, -1.0 / (math.sqrt(kernel_shape[3]) + 10), 1.0 / (math.sqrt(kernel_shape[3]) + 10)), 
				# 	trainable=trainable)
				kernel = tf.get_variable(name=scope+'kernel', initializer=tf.random_normal(kernel_shape, stddev=0.2), trainable=trainable)
				biases = tf.get_variable(name=scope+'bias', initializer=tf.zeros(shape=[kernel_shape[-1]], dtype=tf.float32), trainable=trainable)
				if if_batch_norm:
					current_output = transfer_fct(
						self.BatchNorm(
							tf.add(tf.nn.conv3d(current_input, kernel, strides, padding), biases),
							trainable=trainable, scope=scope, reuse=None
							)
						)
				else:
					current_output = transfer_fct(
							tf.nn.bias_add(tf.nn.conv3d(current_input, kernel, strides, padding), biases),
						)
				return current_output
			def transfer_fct_none(x):
				return x

			current_input = conv_layer(current_input, [4, 4, 4, 1, 32], [1, 2, 2, 2, 1], 'BN-0', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN_D, padding="SAME", trainable=trainable, reuse=reuse)
			print current_input.get_shape().as_list()
			current_input = conv_layer(current_input, [4, 4, 4, 32, 64], [1, 2, 2, 2, 1], 'BN-1', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN_D, padding="SAME", trainable=trainable, reuse=reuse)
			print current_input.get_shape().as_list()
			current_input = conv_layer(current_input, [4, 4, 4, 64, 128], [1, 2, 2, 2, 1], 'BN-2', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN_D, padding="SAME", trainable=trainable, reuse=reuse)
			print current_input.get_shape().as_list()
			current_input = conv_layer(current_input, [4, 4, 4, 128, 256], [1, 2, 2, 2, 1], 'BN-3', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN_D, padding="SAME", trainable=trainable, reuse=reuse)
			print current_input.get_shape().as_list()
			# current_input = conv_layer(current_input, [4, 4, 4, 512, 1], [1, 1, 1, 1, 1], 'BN-4', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN, padding="SAME", trainable=trainable)
			# print current_input.get_shape().as_list()

			self.before_flatten_shape = current_input.get_shape().as_list()
			self.flatten_shape = tf.pack([-1, np.prod(current_input.get_shape().as_list() [1:])])
			flattened = tf.reshape(current_input, self.flatten_shape)
			self.flatten_length = flattened.get_shape().as_list()[1]

			print '---------- _>>> discriminator: flatten length:', self.flatten_length
			hidden_tensor = ly.fully_connected(flattened, self.flatten_length//2, activation_fn=self.transfer_fct_conv, trainable=trainable, scope='d_fc1')
			hidden_tensor = ly.fully_connected(hidden_tensor, self.flatten_length//4, activation_fn=self.transfer_fct_conv, trainable=trainable, scope='d_fc2')
			hidden_tensor = ly.fully_connected(hidden_tensor, 1, activation_fn=None, trainable=trainable, scope='d_fc3')
			return hidden_tensor

	def _generator(self, input_sample, trainable=True):
		with tf.variable_scope("generator") as scope:
			dyn_batch_size = tf.shape(input_sample)[0]
			hidden_tensor_inv = ly.fully_connected(input_sample, self.flatten_length//4, activation_fn=self.transfer_fct_conv, trainable=trainable, scope='g_fc1')
			hidden_tensor_inv = ly.fully_connected(hidden_tensor_inv, self.flatten_length//2, activation_fn=self.transfer_fct_conv, trainable=trainable, scope='g_fc2')
			hidden_tensor_inv = ly.fully_connected(hidden_tensor_inv, self.flatten_length, activation_fn=self.transfer_fct_conv, trainable=trainable, scope='g_fc3')

			current_input = tf.reshape(hidden_tensor_inv, [-1, 2, 2, 2, 256])
			print 'current_input', current_input.get_shape().as_list()

			def deconv_layer(current_input, kernel_shape, strides, output_shape, scope, transfer_fct, is_training, if_batch_norm, padding, trainable):
				# kernel = tf.truncated_normal(kernel_shape, dtype=tf.float32, stddev=1e-1)
				# kernel = tf.Variable(
				# 	tf.random_uniform(kernel_shape, -1.0 / (math.sqrt(kernel_shape[3]) + 10), 1.0 / (math.sqrt(kernel_shape[3]) + 10)), 
				# 	trainable=trainable)
				kernel = tf.Variable(tf.random_normal(kernel_shape, stddev=0.2), name=scope+'kernel')
				biases = tf.Variable(tf.zeros(shape=[kernel_shape[-2]], dtype=tf.float32), trainable=trainable, name=scope+'bias')
				if if_batch_norm:
					current_output = transfer_fct(
						self.BatchNorm(tf.reshape(
							tf.add(tf.nn.conv3d_transpose(current_input, kernel,
								output_shape, strides, padding), biases),
							output_shape),
							trainable=trainable, scope=scope, reuse=None
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
			current_input = deconv_layer(current_input, [4, 4, 4, 128, 256], [1, 2, 2, 2, 1], tf.pack([dyn_batch_size, 4, 4, 4, 128]), 'BN-deconv-0', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN_G, padding="SAME", trainable=trainable)
			print current_input.get_shape().as_list()
			current_input = deconv_layer(current_input, [4, 4, 4, 64, 128], [1, 2, 2, 2, 1], tf.pack([dyn_batch_size, 8, 8, 8, 64]), 'BN-deconv-1', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN_G, padding ="SAME", trainable=trainable)
			print current_input.get_shape().as_list()
			current_input = deconv_layer(current_input, [4, 4, 4, 32, 64], [1, 2, 2, 2, 1], tf.pack([dyn_batch_size, 15, 15, 15, 32]), 'BN-deconv-2', self.transfer_fct_conv, is_training=self.is_training, if_batch_norm=FLAGS.if_BN_G, padding="SAME", trainable=trainable)
			print current_input.get_shape().as_list()
			current_input = deconv_layer(current_input, [4, 4, 4, 1, 32], [1, 2, 2, 2, 1], tf.pack([dyn_batch_size, 30, 30, 30, 1]), 'BN-deconv-3', transfer_fct_none, is_training=self.is_training, if_batch_norm=False, padding="SAME", trainable=trainable)
			print current_input.get_shape().as_list()
			print '---------- _<<< generator: flatten length:', self.flatten_length
			return (tf.nn.tahn(current_input), current_input)

def train(gan):
	prepare_for_training(gan)
	# sample_z = np.random.uniform(-1, 1, size=(gan.batch_size, gan.z_size))
	i = 0
	try:
		while not gan.coord.should_stop():
			def next_feed_dict():
				batch_z = np.random.normal(0., 1., [gan.batch_size, gan.z_size]) \
								.astype(np.float32)
				feed_dict={gan.z: batch_z, gan.is_training: True, gan.gen.is_training: True, gan.is_queue: True, gan.train_net: True}
				return feed_dict

			def write_to_screen(merged):
				start_time = time.time()
				feed_dict = next_feed_dict()
				G, d_loss, g_loss, step = gan.sess.run([gan.G, gan.d_loss, gan.g_loss, gan.global_step], feed_dict=feed_dict)

				gan.train_writer.add_summary(merged, step)

				epoch_show = math.floor(float(step) * FLAGS.models_in_batch / float(num_samples))
				batch_show = math.floor(step - epoch_show * (num_samples / FLAGS.models_in_batch))

				if FLAGS.if_summary:
					gan.train_writer.flush()
					if FLAGS.train_net:
						print "i", '%03d' % (i), "STEP", '%03d' % (step), "Epo", '%03d' % (epoch_show), "ba", '%03d' % (batch_show), \
						"d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)

				if FLAGS.if_save and step != 0 and step % FLAGS.save_every_step == 0:
					save_gan(gan, step, epoch_show, batch_show)

				if FLAGS.if_draw and step % FLAGS.draw_every == 0:
					print 'Drawing reconstructed sample from testing batch...'
					plt.figure(1)
					print G.shape
					for test_idx in range(2):
						im = draw_sample(figM, (G[test_idx].reshape((30, 30, 30)))*0.5+0.5, ms)
						plt.subplot(1, 2, test_idx+1)
						plt.imshow(im)
						plt.axis('off')
					pltfig_3d.suptitle('Reconstructed models at step %s of %s'%(step, FLAGS.folder_name_save_to), fontsize=20, fontweight='bold')
					pltfig_3d.canvas.draw()
					pltfig_3d.savefig('./saved_images/%d-pltfig_3d_recon.png'%step)
				end_time = time.time()
				elapsed = end_time - start_time
				print "--- Time %f seconds."%elapsed

			if i < 25 or i % 500 == 0:
				citers = 100
			else:
				citers = gan.Citers
			for j in range(citers):
				feed_dict = next_feed_dict()
				# run_options = tf.RunOptions(
				# 	trace_level=tf.RunOptions.FULL_TRACE)
				# run_metadata = tf.RunMetadata()
				_, merged = gan.sess.run([gan.opt_c, gan.merged_summary], feed_dict=feed_dict)
									 # options=run_options, run_metadata=run_metadata)
				# gan.train_writer.add_summary(merged, i)
				# gan.train_writer.add_run_metadata(
				# 	run_metadata, 'critic_metadata {}'.format(i), i)
				write_to_screen(merged)

			feed_dict = next_feed_dict()
			_, merged = gan.sess.run([gan.opt_g, gan.merged_summary], feed_dict=feed_dict)
				 # options=run_options, run_metadata=run_metadata)
			# gan.train_writer.add_summary(merged, i)
			# gan.train_writer.add_run_metadata(
			# 	run_metadata, 'generator_metadata {}'.format(i), i)
			write_to_screen(merged)

			i = i + 1
	except tf.errors.OutOfRangeError:
		print('Done training.')
	finally:
		# When done, ask the threads to stop.
		gan.coord.request_stop()
	# Wait for threads to finish.
	gan.coord.join(gan.threads)
	gan.sess.close()

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