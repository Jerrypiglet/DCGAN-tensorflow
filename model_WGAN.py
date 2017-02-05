from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

class DCGAN(object):
	def __init__(self, sess, input_height=108, input_width=108, is_crop=True,
				 batch_size=64, sample_num = 64, output_height=64, output_width=64,
				 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
				 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
				 input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
		"""

		Args:
			sess: TensorFlow session
			batch_size: The size of batch. Should be specified before training.
			y_dim: (optional) Dimension of dim for y. [None]
			z_dim: (optional) Dimension of dim for Z. [100]
			gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
			df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
			gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
			dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
			c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
		"""
		self.sess = sess
		self.is_crop = is_crop
		self.is_grayscale = (c_dim == 1)

		self.batch_size = batch_size
		self.sample_num = sample_num

		self.input_height = input_height
		self.input_width = input_width
		self.output_height = output_height
		self.output_width = output_width

		self.y_dim = y_dim
		self.z_dim = z_dim

		self.gf_dim = gf_dim
		self.df_dim = df_dim

		self.gfc_dim = gfc_dim
		self.dfc_dim = dfc_dim

		self.c_dim = c_dim

		# batch normalization : deals with poor initialization helps gradient flow
		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')

		if not self.y_dim:
			self.d_bn3 = batch_norm(name='d_bn3')

		self.g_bn0 = batch_norm(name='g_bn0')
		self.g_bn1 = batch_norm(name='g_bn1')
		self.g_bn2 = batch_norm(name='g_bn2')

		if not self.y_dim:
			self.g_bn3 = batch_norm(name='g_bn3')

		self.dataset_name = dataset_name
		self.input_fname_pattern = input_fname_pattern
		self.checkpoint_dir = checkpoint_dir
		self.build_model()

	def build_model(self):
		if self.y_dim:
			self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

		if self.is_crop:
			image_dims = [self.output_height, self.output_width, self.c_dim]
		else:
			image_dims = [self.input_height, self.input_height, self.c_dim]

		self.inputs = tf.placeholder(
			tf.float32, [self.batch_size] + image_dims, name='real_images')
		self.sample_inputs = tf.placeholder(
			tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

		inputs = self.inputs
		sample_inputs = self.sample_inputs

		self.z = tf.placeholder(
			tf.float32, [None, self.z_dim], name='z')
		self.z_sum = histogram_summary("z", self.z)

		if self.y_dim:
			self.G = self.generator(self.z, self.y)
			self.D_logits = \
					self.discriminator(inputs, self.y, reuse=False)

			self.sampler = self.sampler(self.z, self.y)
			self.D_logits_ = \
					self.discriminator(self.G, self.y, reuse=True)
		else:
			self.G = self.generator(self.z)
			self.D_logits = self.discriminator(inputs)

			self.sampler = self.sampler(self.z)
			self.D_logits_ = self.discriminator(self.G, reuse=True)

		self.d_sum = histogram_summary("d", self.D_logits)
		self.d__sum = histogram_summary("d_", self.D_logits_)
		self.G_sum = image_summary("G", self.G)

		self.d_loss_real = tf.reduce_mean(
			# tf.nn.sigmoid_cross_entropy_with_logits(
			# 	logits=self.D_logits, targets=tf.ones_like(self.D)))
			- self.D_logits)
		self.d_loss_fake = tf.reduce_mean(
			# tf.nn.sigmoid_cross_entropy_with_logits(
			# 	logits=self.D_logits_, targets=tf.zeros_like(self.D_)))
			self.D_logits_)
		self.g_loss = tf.reduce_mean(
			# tf.nn.sigmoid_cross_entropy_with_logits(
			# 	logits=self.D_logits_, targets=tf.ones_like(self.D_)))
			- self.D_logits_)

		self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
		self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
													
		self.d_loss = self.d_loss_real + self.d_loss_fake

		self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
		self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

		# t_vars = tf.trainable_variables()

		# self.d_vars = [var for var in t_vars if 'd_' in var.name]
		# self.g_vars = [var for var in t_vars if 'g_' in var.name]

		self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d_')
		self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_')

		self.saver = tf.train.Saver()

	def train(self, config):
		"""Train DCGAN"""
		if config.dataset == 'mnist':
			data_X, data_y = self.load_mnist()
		else:
			data = glob(os.path.join("./data", config.dataset, self.input_fname_pattern))
		#np.random.shuffle(data)

		learning_rate_ger = 5e-5
		learning_rate_dis = 5e-5
		# update Citers times of critic in one iter(unless i < 25 or i % 500 == 0, i is iterstep)
		Citers = 5
		# the upper bound and lower bound of parameters in critic
		clamp_lower = -0.01
		clamp_upper = 0.01
		# whether to use adam for parameter update, if the flag is set False, use tf.train.RMSPropOptimizer
		# as recommended in paper
		is_adam = False

		counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
		opt_g = ly.optimize_loss(loss=self.g_loss, learning_rate=learning_rate_ger,
						optimizer=tf.train.AdamOptimizer if is_adam is True else tf.train.RMSPropOptimizer, 
						variables=self.g_vars, global_step=counter_g,
						summaries = 'gradient_norm')

		counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
		opt_c = ly.optimize_loss(loss=self.c_loss, learning_rate=learning_rate_dis,
						optimizer=tf.train.AdamOptimizer if is_adam is True else tf.train.RMSPropOptimizer, 
						variables=theta_c, global_step=counter_c,
						summaries = 'gradient_norm')
		clipped_var_c = [tf.assign(var, tf.clip_by_value(var, clamp_lower, clamp_upper)) for var in self.d_vars]
		# merge the clip operations on critic variables
		with tf.control_dependencies([opt_c]):
			opt_c = tf.tuple(clipped_var_c)

		# d_optim = tf.train.AdamOptimizer(learning_rate_dis, beta1=config.beta1) \
		# 					.minimize(self.d_loss, var_list=self.d_vars)
		# g_optim = tf.train.AdamOptimizer(learning_rate_ger, beta1=config.beta1) \
		# 					.minimize(self.g_loss, var_list=self.g_vars)
		try:
			tf.global_variables_initializer().run()
		except:
			tf.initialize_all_variables().run()

		self.g_sum = merge_summary([self.z_sum, self.d__sum,
			self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
		self.d_sum = merge_summary(
				[self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
		self.writer = SummaryWriter("./logs", self.sess.graph)

		sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
		
		if config.dataset == 'mnist':
			sample_inputs = data_X[0:self.sample_num]
			sample_labels = data_y[0:self.sample_num]
		else:
			sample_files = data[0:self.sample_num]
			sample = [
					get_image(sample_file,
										input_height=self.input_height,
										input_width=self.input_width,
										resize_height=self.output_height,
										resize_width=self.output_width,
										is_crop=self.is_crop,
										is_grayscale=self.is_grayscale) for sample_file in sample_files]
			if (self.is_grayscale):
				sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
			else:
				sample_inputs = np.array(sample).astype(np.float32)
	
		counter = 1
		start_time = time.time()

		if self.load(self.checkpoint_dir):
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		def next_feed_dict_mnist():
			train_img = dataset.train.next_batch(batch_size)[0]
			train_img = 2*train_img-1
			if is_svhn is not True:
				train_img = np.reshape(train_img, (-1, 28, 28))
				train_img = np.pad(train_img, pad_width=npad,
								   mode='constant', constant_values=-1)
				train_img = np.expand_dims(train_img, -1)
			batch_z = np.random.normal(0, 1, [batch_size, z_dim]) \
				.astype(np.float32)
			feed_dict = {real_data: train_img, z: batch_z}
			return feed_dict

		for epoch in xrange(config.epoch):
			if config.dataset == 'mnist':
				batch_idxs = min(len(data_X), config.train_size) // config.batch_size
			else:      
				data = glob(os.path.join(
					"./data", config.dataset, self.input_fname_pattern))
				batch_idxs = min(len(data), config.train_size) // config.batch_size

			for idx in xrange(0, batch_idxs):
				if config.dataset == 'mnist':
					batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
					batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]
				else:
					batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
					batch = [
							get_image(batch_file,
												input_height=self.input_height,
												input_width=self.input_width,
												resize_height=self.output_height,
												resize_width=self.output_width,
												is_crop=self.is_crop,
												is_grayscale=self.is_grayscale) for batch_file in batch_files]
					if (self.is_grayscale):
						batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
					else:
						batch_images = np.array(batch).astype(np.float32)

				batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
							.astype(np.float32)

				if config.dataset == 'mnist':
					# # Update D network
					# _, summary_str = self.sess.run([d_optim, self.d_sum],
					# 	feed_dict={ 
					# 		self.inputs: batch_images,
					# 		self.z: batch_z,
					# 		self.y:batch_labels,
					# 	})
					# self.writer.add_summary(summary_str, counter)

					# # Update G network
					# _, summary_str = self.sess.run([g_optim, self.g_sum],
					# 	feed_dict={
					# 		self.z: batch_z, 
					# 		self.y:batch_labels,
					# 	})
					# self.writer.add_summary(summary_str, counter)

					# # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
					# _, summary_str = self.sess.run([g_optim, self.g_sum],
					# 	feed_dict={ self.z: batch_z, self.y:batch_labels })
					# self.writer.add_summary(summary_str, counter)
					
					# errD_fake = self.d_loss_fake.eval({
					# 		self.z: batch_z, 
					# 		self.y:batch_labels
					# })
					# errD_real = self.d_loss_real.eval({
					# 		self.inputs: batch_images,
					# 		self.y:batch_labels
					# })
					# errG = self.g_loss.eval({
					# 		self.z: batch_z,
					# 		self.y: batch_labels
					# })

					i = idx
					if i < 25 or i % 500 == 0:
						citers = 100
					else:
						citers = Citers
					for j in range(citers):
						feed_dict = next_feed_dict()
						if i % 100 == 99 and j == 0:
							run_options = tf.RunOptions(
								trace_level=tf.RunOptions.FULL_TRACE)
							run_metadata = tf.RunMetadata()
							_, merged = sess.run([opt_c, merged_all], feed_dict=feed_dict,
												 options=run_options, run_metadata=run_metadata)
							summary_writer.add_summary(merged, i)
							summary_writer.add_run_metadata(
								run_metadata, 'critic_metadata {}'.format(i), i)
						else:
							sess.run(opt_c, feed_dict=feed_dict)                
					feed_dict = next_feed_dict()
					if i % 100 == 99:
						_, merged = sess.run([opt_g, merged_all], feed_dict=feed_dict,
							 options=run_options, run_metadata=run_metadata)
						summary_writer.add_summary(merged, i)
						summary_writer.add_run_metadata(
							run_metadata, 'generator_metadata {}'.format(i), i)
					else:
						sess.run(opt_g, feed_dict=feed_dict)    
				else:
					# Update D network
					_, summary_str = self.sess.run([d_optim, self.d_sum],
						feed_dict={ self.inputs: batch_images, self.z: batch_z })
					self.writer.add_summary(summary_str, counter)

					# Update G network
					_, summary_str = self.sess.run([g_optim, self.g_sum],
						feed_dict={ self.z: batch_z })
					self.writer.add_summary(summary_str, counter)

					# Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
					_, summary_str = self.sess.run([g_optim, self.g_sum],
						feed_dict={ self.z: batch_z })
					self.writer.add_summary(summary_str, counter)
					
					errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
					errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
					errG = self.g_loss.eval({self.z: batch_z})

				counter += 1
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
					% (epoch, idx, batch_idxs,
						time.time() - start_time, errD_fake+errD_real, errG))

				if np.mod(counter, 100) == 1:
					if config.dataset == 'mnist':
						samples, d_loss, g_loss = self.sess.run(
							[self.sampler, self.d_loss, self.g_loss],
							feed_dict={
									self.z: sample_z,
									self.inputs: sample_inputs,
									self.y:sample_labels,
							}
						)
						save_images(samples, [8, 8],
									'./{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
						print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
					else:
						try:
							samples, d_loss, g_loss = self.sess.run(
								[self.sampler, self.d_loss, self.g_loss],
								feed_dict={
										self.z: sample_z,
										self.inputs: sample_inputs,
								},
							)
							save_images(samples, [8, 8],
										'./{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
							print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
						except:
							print("one pic error!...")

				if np.mod(counter, 500) == 2:
					self.save(config.checkpoint_dir, counter)

	def discriminator(self, image, y=None, reuse=False):
		with tf.variable_scope("discriminator") as scope:
			if reuse:
				scope.reuse_variables()

			if not self.y_dim:
				print "DISCRIMINAROT: not self.y_dim"
				h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
				h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
				h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
				h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
				h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

				return h4
			else:
				print "DISCRIMINAROT: self.y_dim"
				yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
				x = conv_cond_concat(image, yb)

				h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
				h0 = conv_cond_concat(h0, yb)

				h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
				h1 = tf.reshape(h1, [self.batch_size, -1])      
				h1 = tf.concat_v2([h1, y], 1)
				
				h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
				h2 = tf.concat_v2([h2, y], 1)

				h3 = linear(h2, 1, 'd_h3_lin')
				
				return h3

	def generator(self, z, y=None):
		with tf.variable_scope("generator") as scope:
			if not self.y_dim:
				s_h, s_w = self.output_height, self.output_width
				s_h2, s_h4, s_h8, s_h16 = \
						int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
				s_w2, s_w4, s_w8, s_w16 = \
						int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

				# project `z` and reshape
				self.z_, self.h0_w, self.h0_b = linear(
						z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

				self.h0 = tf.reshape(
						self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
				h0 = tf.nn.relu(self.g_bn0(self.h0))

				self.h1, self.h1_w, self.h1_b = deconv2d(
						h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
				h1 = tf.nn.relu(self.g_bn1(self.h1))

				h2, self.h2_w, self.h2_b = deconv2d(
						h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
				h2 = tf.nn.relu(self.g_bn2(h2))

				h3, self.h3_w, self.h3_b = deconv2d(
						h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
				h3 = tf.nn.relu(self.g_bn3(h3))

				h4, self.h4_w, self.h4_b = deconv2d(
						h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

				return tf.nn.tanh(h4)
			else:
				s_h, s_w = self.output_height, self.output_width
				s_h2, s_h4 = int(s_h/2), int(s_h/4)
				s_w2, s_w4 = int(s_w/2), int(s_w/4)

				# yb = tf.expand_dims(tf.expand_dims(y, 1),2)
				yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
				z = tf.concat_v2([z, y], 1)

				h0 = tf.nn.relu(
						self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
				h0 = tf.concat_v2([h0, y], 1)

				h1 = tf.nn.relu(self.g_bn1(
						linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
				h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

				h1 = conv_cond_concat(h1, yb)

				h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
						[self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
				h2 = conv_cond_concat(h2, yb)

				return tf.nn.sigmoid(
						deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

	def sampler(self, z, y=None):
		with tf.variable_scope("generator") as scope:
			scope.reuse_variables()

			if not self.y_dim:
				
				s_h, s_w = self.output_height, self.output_width
				s_h2, s_h4, s_h8, s_h16 = \
						int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
				s_w2, s_w4, s_w8, s_w16 = \
						int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

				# project `z` and reshape
				h0 = tf.reshape(
						linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
						[-1, s_h16, s_w16, self.gf_dim * 8])
				h0 = tf.nn.relu(self.g_bn0(h0, train=False))

				h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
				h1 = tf.nn.relu(self.g_bn1(h1, train=False))

				h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
				h2 = tf.nn.relu(self.g_bn2(h2, train=False))

				h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
				h3 = tf.nn.relu(self.g_bn3(h3, train=False))

				h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

				return tf.nn.tanh(h4)
			else:
				s_h, s_w = self.output_height, self.output_width
				s_h2, s_h4 = int(s_h/2), int(s_h/4)
				s_w2, s_w4 = int(s_w/2), int(s_w/4)

				# yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
				yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
				z = tf.concat_v2([z, y], 1)

				h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
				h0 = tf.concat_v2([h0, y], 1)

				h1 = tf.nn.relu(self.g_bn1(
						linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
				h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
				h1 = conv_cond_concat(h1, yb)

				h2 = tf.nn.relu(self.g_bn2(
						deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
				h2 = conv_cond_concat(h2, yb)

				return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

	def load_mnist(self):
		data_dir = os.path.join("./data", self.dataset_name)
		
		fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
		loaded = np.fromfile(file=fd,dtype=np.uint8)
		trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

		fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
		loaded = np.fromfile(file=fd,dtype=np.uint8)
		trY = loaded[8:].reshape((60000)).astype(np.float)

		fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
		loaded = np.fromfile(file=fd,dtype=np.uint8)
		teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

		fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
		loaded = np.fromfile(file=fd,dtype=np.uint8)
		teY = loaded[8:].reshape((10000)).astype(np.float)

		trY = np.asarray(trY)
		teY = np.asarray(teY)
		
		X = np.concatenate((trX, teX), axis=0)
		y = np.concatenate((trY, teY), axis=0).astype(np.int)
		
		seed = 547
		np.random.seed(seed)
		np.random.shuffle(X)
		np.random.seed(seed)
		np.random.shuffle(y)
		
		y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
		for i, label in enumerate(y):
			y_vec[i,y[i]] = 1.0
		
		return X/255.,y_vec

	@property
	def model_dir(self):
		return "{}_{}_{}_{}".format(
				self.dataset_name, self.batch_size,
				self.output_height, self.output_width)
			
	def save(self, checkpoint_dir, step):
		model_name = "DCGAN.model"
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		print(" [*] Reading checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			print(" [*] Success to read {}".format(ckpt_name))
			return True
		else:
			print(" [*] Failed to find a checkpoint")
			return False
