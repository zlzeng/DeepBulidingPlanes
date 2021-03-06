import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import os

seed = 8964	# random seed
# tf.set_random_seed(seed)
tf.compat.v1.set_random_seed(seed)

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
GPU_ID=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
os.system('rm tmp')
print('GPU: {} will be used'.format(GPU_ID))

# os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore warning messages

class NET(object):
	"""
	Plane segmentation
	"""

	def __init__(self, dtype=tf.float32):
		print('Initial network object...')
		self.dtype = dtype
		self.line_num_classes = 2
		self.planar_num_classes = 6
		# self.pre_trained_vgg_model = './vgg16/vgg_16.ckpt'
		self.pre_trained_vgg_model = None
		self.num_iterations = 0

	# basic layer
	def _he_uniform(self, shape, regularizer=None, trainable=None, name=None):
		name = 'weights' if name is None else name + '/weights'

		# size = (k_h, k_w, in_dim, out_dim)
		kernel_size = np.prod(shape[:2])	# k_h*k_w
		fan_in = shape[-2] * kernel_size	# fan_out = shape[-1]*kernel_size

		# compute the scale value
		s = np.sqrt(1. / fan_in)

		# create variable and specific GPU device
		with tf.device('/device:GPU:' + GPU_ID):
			w = tf.get_variable(name, shape, dtype=self.dtype,
								initializer=tf.random_uniform_initializer(minval=-s, maxval=s),
								regularizer=regularizer, trainable=trainable)

		return w

	def _constant(self, shape, value=0, regularizer=None, trainable=None, name=None):
		name = 'biases' if name is None else name + '/biases'

		with tf.device('/device:GPU:' + GPU_ID):
			b = tf.get_variable(name, shape, dtype=self.dtype,
								initializer=tf.constant_initializer(value=value),
								regularizer=regularizer, trainable=trainable)

		return b

	def _conv2d(self, tensor, dim, size=3, stride=1, rate=1, pad='SAME', act='relu', norm='gn', G=8, bias=True, name='conv'):
		"""pre activate => norm => conv
		"""
		in_dim = tensor.shape.as_list()[-1]
		size = size if isinstance(size, (tuple, list)) else [size, size]
		stride = stride if isinstance(stride, (tuple, list)) else [1, stride, stride, 1]
		rate = rate if isinstance(rate, (tuple, list)) else [1, rate, rate, 1]
		kernel_shape = [size[0], size[1], in_dim, dim]

		w = self._he_uniform(kernel_shape, name=name)
		b = self._constant(dim, name=name) if bias else 0

		if act == 'relu':
			tensor = tf.nn.relu(tensor, name=name + '/relu')
		elif act == 'sigmoid':
			tensor = tf.nn.sigmoid(tensor, name=name + '/sigmoid')
		elif act == 'softplus':
			tensor = tf.nn.softplus(tensor, name=name + '/softplus')
		elif act == 'leaky_relu':
			tensor = tf.nn.leaky_relu(tensor, name=name + '/leaky_relu')
		else:
			norm = 'none'

		if norm == 'gn':	# group normalization after acitvation
			# normalize
			# tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
			x = tf.transpose(tensor, [0, 3, 1, 2])
			N, C, H, W = x.get_shape().as_list()
			G = min(G, C)
			x = tf.reshape(x, [-1, G, C // G, H, W])
			mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
			x = (x - mean) / tf.sqrt(var + 1e-6)

			# per channel gamma and beta
			with tf.device('/device:GPU:' + GPU_ID):
				gamma = tf.get_variable(name + '/gamma', [C], dtype=self.dtype, initializer=tf.constant_initializer(1.0))
				beta = tf.get_variable(name + '/beta', [C], dtype=self.dtype, initializer=tf.constant_initializer(0.0))
				gamma = tf.reshape(gamma, [1, C, 1, 1])
				beta = tf.reshape(beta, [1, C, 1, 1])

			tensor = tf.reshape(x, [-1, C, H, W]) * gamma + beta
			# tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
			tensor = tf.transpose(tensor, [0, 2, 3, 1])

		out = tf.nn.conv2d(tensor, w, strides=stride, padding=pad, dilations=rate, name=name) + b	# default no bias

		return out

	def _max_pool2d(self, tensor, size=2, stride=2, pad='VALID'):
		size = size if isinstance(size, (tuple, list)) else [1, size, size, 1]
		stride = stride if isinstance(stride, (tuple, list)) else [1, stride, stride, 1]
		#
		size = [1, size[0], size[1], 1] if len(size) == 2 else size
		stride = [1, stride[0], stride[1], 1] if len(stride) == 2 else stride

		out = tf.nn.max_pool(tensor, size, stride, pad)

		return out

	def _up_bilinear(self, tensor, dim, shape, name='upsample'):
		out = tf.image.resize_bilinear(tensor, shape)
		out = self._conv2d(out, dim=dim, size=1, name=name + '/1x1_conv')
		return out

	def _conv_down(self, tensor, dim=24, name='conv_down'):
		out = self._conv2d(tensor, dim=dim, size=1, act='linear', name=name)
		return out

	def _res_block_v2(self, tensor, name='res'):
		shortcut = tensor

		# not change the depth 
		dim = tensor.shape.as_list()[-1]

		# residual building block using our conv2d
		out = self._conv2d(tensor, dim=dim, name=name+'/3x3_conv1')
		out = self._conv2d(out, dim=dim, name=name+'/3x3_conv2')

		return out+shortcut

	def _gated_conv2d(self, s1, s2, name='gc'):
		# not change the depth
		dim = s1.shape.as_list()[-1]

		alpha = tf.nn.sigmoid(s1)

		out = alpha * s2 + s2

		out = self._conv2d(out, dim=dim, name=name)

		return out

	def encode(self, x, scope='vgg_16'):

		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

			# features = self._conv2d(x, dim=32, norm='none', name='conv0')

			# stage 1
			with tf.variable_scope('conv1'):
				self.conv1_1 = self._conv2d(x, dim=64, name='conv1_1')
				self.conv1_2 = self._conv2d(self.conv1_1, dim=64, name='conv1_2')
				self.pool1 = self._max_pool2d(self.conv1_2)	# /2
			with tf.variable_scope('conv2'):
				self.conv2_1 = self._conv2d(self.pool1, dim=128, name='conv2_1')
				self.conv2_2 = self._conv2d(self.conv2_1, dim=128, name='conv2_2')
				self.pool2 = self._max_pool2d(self.conv2_2)	# /4
			with tf.variable_scope('conv3'):
				self.conv3_1 = self._conv2d(self.pool2, dim=256, name='conv3_1')
				self.conv3_2 = self._conv2d(self.conv3_1, dim=256, name='conv3_2')
				self.conv3_3 = self._conv2d(self.conv3_2, dim=256, name='conv3_3')
				self.pool3 = self._max_pool2d(self.conv3_3)	# /8
			with tf.variable_scope('conv4'):
				self.conv4_1 = self._conv2d(self.pool3, dim=512, name='conv4_1')
				self.conv4_2 = self._conv2d(self.conv4_1, dim=512, name='conv4_2')
				self.conv4_3 = self._conv2d(self.conv4_2, dim=512, name='conv4_3')
				self.pool4 = self._max_pool2d(self.conv4_3)	# /16
			with tf.variable_scope('conv5'):
				self.conv5_1 = self._conv2d(self.pool4, rate=2, dim=512, name='conv5_1')
				self.conv5_2 = self._conv2d(self.conv5_1, rate=2, dim=512, name='conv5_2')
				self.conv5_3 = self._conv2d(self.conv5_2, rate=2, dim=512, name='conv5_3')
				# self.pool5 = self._max_pool2d(self.conv5_3)	# /32

	def dsn(self, geo, dim, shape, scope='dsn_decode'):

		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			# dsn1
			conv1_1_down = self._conv_down(self.conv1_1, name='conv1_1_down')
			conv1_2_down = self._conv_down(self.conv1_2, name='conv1_2_down')
			conv1_norm = self._conv_down(geo.conv1_norm, name='conv1_norm_down')
			score_fuse1 = conv1_1_down + conv1_2_down
			score_dsn1 = self._up_bilinear(score_fuse1, dim=24, \
											shape=shape, name='score_dsn1')
			score_norm1 = self._up_bilinear(conv1_norm, dim=24, \
											shape=shape, name='score_norm1')
			# score_dsn1 = score_dsn1 + score_norm1
			score_dsn1 = tf.concat([score_dsn1, score_norm1], axis=3)

			# res
			res1 = self._res_block_v2(score_dsn1, name='res1')

			# res logits
			res_dsn1 = self._conv_down(res1, dim=dim, name='res_dsn1')

			# dsn2
			conv2_1_down = self._conv_down(self.conv2_1, name='conv2_1_down')
			conv2_2_down = self._conv_down(self.conv2_2, name='conv2_2_down')
			conv2_norm = self._conv_down(geo.conv2_norm, name='conv2_norm_down')
			score_fuse2 = conv2_1_down + conv2_2_down
			score_dsn2 = self._up_bilinear(score_fuse2, dim=24, \
											shape=shape, name='score_dsn2')
			score_norm2 = self._up_bilinear(conv2_norm, dim=24, \
											shape=shape, name='score_norm2')
			# score_dsn2 = score_dsn2 + score_norm2
			score_dsn2 = tf.concat([score_dsn2, score_norm2], axis=3)

			# gated conv layer
			gated_score_dsn2 = self._gated_conv2d(res1, score_dsn2, name='gc2')

			# res
			res2 = self._res_block_v2(gated_score_dsn2, name='res2')	

			# res logits
			res_dsn2 = self._conv_down(res2, dim=dim, name='res_dsn2')					

			# dsn3
			conv3_1_down = self._conv_down(self.conv3_1, name='conv3_1_down')
			conv3_2_down = self._conv_down(self.conv3_2, name='conv3_2_down')
			conv3_3_down = self._conv_down(self.conv3_3, name='conv3_3_down')
			conv3_norm = self._conv_down(geo.conv3_norm, name='conv3_norm_down')
			score_fuse3 = conv3_1_down + conv3_2_down + conv3_3_down
			score_dsn3 = self._up_bilinear(score_fuse3, dim=24, \
											shape=shape, name='score_dsn3')
			score_norm3 = self._up_bilinear(conv3_norm, dim=24, \
											shape=shape, name='score_norm3')
			# score_dsn3 = score_dsn3 + score_norm3
			score_dsn3 = tf.concat([score_dsn3, score_norm3], axis=3)

			# gated conv layer
			gated_score_dsn3 = self._gated_conv2d(res2, score_dsn3, name='gc3')

			# res
			res3 = self._res_block_v2(gated_score_dsn3, name='res3')

			# res logits
			res_dsn3 = self._conv_down(res3, dim=dim, name='res_dsn3')			

			# dsn4
			conv4_1_down = self._conv_down(self.conv4_1, name='conv4_1_down')
			conv4_2_down = self._conv_down(self.conv4_2, name='conv4_2_down')
			conv4_3_down = self._conv_down(self.conv4_3, name='conv4_3_down')
			conv4_norm = self._conv_down(geo.conv4_norm, name='conv4_norm_down')			
			score_fuse4 = conv4_1_down + conv4_2_down + conv4_3_down
			score_dsn4 = self._up_bilinear(score_fuse4, dim=24, \
											shape=shape, name='score_dsn4')
			score_norm4 = self._up_bilinear(conv4_norm, dim=24, \
											shape=shape, name='score_norm4')
			# score_dsn4 = score_dsn4 + score_norm4
			score_dsn4 = tf.concat([score_dsn4, score_norm4], axis=3)

			# gated conv layer
			gated_score_dsn4 = self._gated_conv2d(res3, score_dsn4, name='gc4')

			# res
			res4 = self._res_block_v2(gated_score_dsn4, name='res4')

			# res logits
			res_dsn4 = self._conv_down(res4, dim=dim, name='res_dsn4')			

			# dsn5
			conv5_1_down = self._conv_down(self.conv5_1, name='conv5_1_down')
			conv5_2_down = self._conv_down(self.conv5_2, name='conv5_2_down')
			conv5_3_down = self._conv_down(self.conv5_3, name='conv5_3_down')
			conv5_norm = self._conv_down(geo.conv5_norm, name='conv5_norm_down')	
			score_fuse5 = conv5_1_down + conv5_2_down + conv5_3_down
			score_dsn5 = self._up_bilinear(score_fuse5, dim=24, \
											shape=shape, name='score_dsn5')
			score_norm5 = self._up_bilinear(conv5_norm, dim=24, \
											shape=shape, name='score_norm5')
			# score_dsn5 = score_dsn5 + score_norm5
			score_dsn5 = tf.concat([score_dsn5, score_norm5], axis=3)

			# gated conv layer
			gated_score_dsn5 = self._gated_conv2d(res4, score_dsn5, name='gc5')

			# res
			res5 = self._res_block_v2(gated_score_dsn5, name='res5')

			# res logits
			res_dsn5 = self._conv_down(res5, dim=dim, name='res_dsn5')

			# fusion
			score_fusions = tf.concat([res1, res2, \
										res3, res4, res5], axis=3)

			res_dsn6 = self._conv_down(score_fusions, dim=dim, name='score_dsn6')

		return [res_dsn1, res_dsn2, res_dsn3, res_dsn4, res_dsn5, res_dsn6]

	def forward(self, image_inputs):

		shape = tf.shape(image_inputs)[1:3]
		[n, h, w, c] = image_inputs.shape.as_list()

		# geo-network
		self.geo = GeoNetNorm()
		self.geo.forward(image_inputs*255)

		# planar-network
		self.encode(image_inputs)
		self.logits_b1 = self.dsn(self.geo, dim=self.planar_num_classes, shape=shape, scope='planar')

		return self.logits_b1

	# def init_from_pretrained_vgg(self):
	#	 # restore pre-trained parameters
	#	 print 'Init from checkpoint: {}'.format(self.pre_trained_vgg_model)
	#	 variables_to_restore = [v for v in tf.trainable_variables() if v.name.startswith('vgg_16')]
	#	 tf.train.init_from_checkpoint(self.pre_trained_vgg_model,
	#									 {v.name.split(':')[0]: v for v in variables_to_restore})


class GeoNetNorm(object):
	"""
	GeoNetNorm

	"""
	def __init__(self):
		self.crop_size_h = 360
		self.crop_size_w = 480		
		self.batch_size = 1		
		self.mean_BGR = [104.008, 116.669, 122.675]
		self.pre_trained_model = '../pretrained/GeoNet/checkpoints'
		self.k = 9
		self.rate = 4
		self.clip_norm = 20.0
		self.thresh = 0.95		

	def forward(self, inputs, is_training=False):
		def preprocessing(inputs):
			dims = inputs.get_shape()
			if len(dims) == 3:
				inputs = tf.expand_dims(inputs, dim=0)
			mean_BGR = tf.reshape(self.mean_BGR, [1, 1, 1, 3])
			inputs = inputs[:, :, :, ::-1] + mean_BGR
			return inputs		
		
		# inputs = preprocessing(inputs)
		with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.relu, stride=1,
								padding='SAME'):
			
			with tf.variable_scope('fcn', reuse=tf.AUTO_REUSE):

				## ----------------- vgg norm---------------------------------------------------------------
				self.conv1_norm = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1_norm')
				self.pool1_norm = slim.max_pool2d(self.conv1_norm, [3, 3], stride=2, padding='SAME', scope='pool1_norm')

				self.conv2_norm = slim.repeat(self.pool1_norm, 2, slim.conv2d, 128, [3, 3], scope='conv2_norm')
				self.pool2_norm = slim.max_pool2d(self.conv2_norm, [3, 3], stride=2, padding='SAME', scope='pool2_norm')

				self.conv3_norm = slim.repeat(self.pool2_norm, 3, slim.conv2d, 256, [3, 3], scope='conv3_norm')
				self.pool3_norm = slim.max_pool2d(self.conv3_norm, [3, 3], stride=2, padding='SAME', scope='pool3_norm')

				self.conv4_norm = slim.repeat(self.pool3_norm, 3, slim.conv2d, 512, [3, 3], scope='conv4_norm')
				self.pool4_norm = slim.max_pool2d(self.conv4_norm, [3, 3], stride=1, padding='SAME', scope='pool4_norm')

				self.conv5_norm = slim.repeat(self.pool4_norm, 3, slim.conv2d, 512, [3, 3], rate=2, scope='conv5_norm')
				self.pool5_norm = slim.max_pool2d(self.conv5_norm, [3, 3], stride=1, padding='SAME', scope='pool5_norm')


