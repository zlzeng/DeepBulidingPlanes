from loader import *
from net import *

class MODEL(object):
	"""docstring for Baseline"""
	def __init__(self):
		self.logdir = '../pretrained/LOG'
		self.eval_file = 'eval_log.txt'
		self.eps = 1e-6
		self.base_lr = 1e-3
		self.use_poly_lr = True # turn on/off poly decay learning rate
		self.batch_size = 2
		# self.max_ep = 100
		self.max_steps = 100000
		self.power = 0.9
		self.save_ep = 2
		self.weight_decay = 1e-4
		self.momentum = 0.9
		self.summary_step = 500 # save summaries every 'summary_step' steps
		# self.keep_n_model = self.max_ep // self.save_ep
		self.keep_n_model = 5 # default
		self.optimizers = 'Momentum' # avaiable optimizer ['Nadam', default:'Adam', 'Adagrad', 'RMSProp', 'Momentum']
		# self.optim_id = 1 

	def preprocessing(self, im, gt, scale=None):

		[h, w] = im.shape.as_list()[1:3]

		# random rescale
		if scale is None:
			scale = tf.random_uniform([], minval=1.0, maxval=2.0, dtype=tf.float32)
		new_height = tf.to_int32(h * scale)
		new_width = tf.to_int32(w * scale)
		im = tf.image.resize_images(im, [new_height, new_width],
										method=tf.image.ResizeMethod.BILINEAR)
		# Since label classes are integers, nearest neighbor need to be used.
		gt = tf.image.resize_images(gt, [new_height, new_width],
										method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

		# crop to original shape
		im = tf.image.resize_image_with_crop_or_pad(im, h, w)	
		gt = tf.image.resize_image_with_crop_or_pad(gt, h, w)	

		# random flip left-right
		# uniform_random = tf.random_uniform([], 0, 1.0)
		# mirror_cond = tf.less(uniform_random, .5)
		# im = tf.cond(mirror_cond, lambda: tf.reverse(im, [2]), lambda: im)
		# gt = tf.cond(mirror_cond, lambda: tf.reverse(gt, [2]), lambda: gt)

		# random flip up-down
		# uniform_random = tf.random_uniform([], 0, 1.0)
		# mirror_cond = tf.less(uniform_random, .5)
		# im = tf.cond(mirror_cond, lambda: tf.reverse(im, [1]), lambda: im)
		# gt = tf.cond(mirror_cond, lambda: tf.reverse(gt, [1]), lambda: gt)

		return im, gt

	def setup_infer_pharse(self, output_ind=True):
		"""
		setup_xxx_pharse function will construct new computation graph, 
		so invoke once is enough, dont mixup and invoke two setup_xxx_pharse
		"""
		## build up static graph
		model_graph = tf.Graph()
		with model_graph.as_default():		
			self.net = NET()
			# input placeholders
			self.image = tf.placeholder(shape=[1, 360, 480, 3], dtype=self.net.dtype)

			self.net.forward(self.image)

			if output_ind:
				self.predict1 = tf.argmax(tf.nn.softmax(self.net.logits_b1[-1]), axis=3)
			else:
				self.predict1 = tf.nn.softmax(self.net.logits_b1[-1], axis=3)

		## initialize session
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config, graph=model_graph)
		# K.set_session(self.sess)

		# handle model in separately session
		with self.sess.as_default():
			with model_graph.as_default():		
				self.sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
				## restore parameters from checkpoints
				var_list = [v for v in tf.trainable_variables() if 'fcn' not in v.name]
				saver = tf.train.Saver(var_list)
				saver.restore(self.sess, save_path=tf.train.latest_checkpoint(self.logdir))

				var_list = [v for v in tf.trainable_variables() if 'fcn' in v.name]
				geo_saver = tf.train.Saver(var_list)
				geo_saver.restore(self.sess, save_path=tf.train.latest_checkpoint(self.net.geo.pre_trained_model))

		## need to use outside: {self.x(in), self.predict(out), self.sess(executor)}

	def setup_train_pharse(self):
		# get inputs
		self.paths = open(train_file, 'r').read().splitlines()
		
		random.shuffle(self.paths)
		total_images = len(self.paths)
		num_batch = total_images // self.batch_size

		# initial network 
		self.net = NET()

		# input placeholders
		self.image = tf.placeholder(shape=[self.batch_size, 360, 480, 3], dtype=self.net.dtype)

		self.label_p = tf.placeholder(shape=[self.batch_size, 360, 480, 1], dtype=tf.uint8) # for one-hot format

		# pre-processing 
		# images, labels = self.preprocessing(self.image, self.label_p)

		images = self.image
		labels = self.label_p

		# initial network architecture
		self.net.forward(images)	

		# use one-hot label
		y = tf.reshape(labels, [self.batch_size, 360, 480, ])
		y_p = tf.one_hot(y, self.net.planar_num_classes)

		# compute loss with multiple side outputs
		entropy_loss = 0.
		for i in range(len(self.net.logits_b1)):
			logits = self.net.logits_b1[i]
			entropy_loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_p, name='ce_stage'+str(i+1)))
		
		# compute loss with mean outputs and weight decay
		# entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.net.logits_b1, labels=y_p, name='ce'))
		l2_losses = [self.weight_decay*tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('weights' in v.name)]
		# l2_losses = [self.weight_decay*tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('weights' in v.name) and ('fcn' not in v.name)]
		self.loss1 = entropy_loss + tf.add_n(l2_losses)

		# poly learning rate
		base_lr = tf.constant(self.base_lr)
		self.step_ph = tf.placeholder(dtype=tf.float32, shape=())
		if self.use_poly_lr:
			# learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - self.step_ph / self.max_steps), self.power))
			learning_rate = tf.train.polynomial_decay(base_lr, 
				tf.cast(self.step_ph, tf.int32) - 0,
				self.max_steps, 1e-6, power=0.9)	
		else:
			learning_rate = base_lr

		# create update op 
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # set to none if no normalization applied

		# initial optimizer and train op
		with tf.control_dependencies(update_ops):
			var_list = [v for v in tf.trainable_variables()]
			# var_list = [v for v in tf.trainable_variables() if 'fcn' not in v.name]
			if self.optimizers == 'Nadam':
				optimizers= tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate)
			elif self.optimizers == 'Adam':
				optimizers = tf.train.AdamOptimizer(learning_rate=learning_rate)
			elif self.optimizers == 'Adagrad':
				optimizers = tf.train.AdagradOptimizer(learning_rate=learning_rate)
			elif self.optimizers == 'RMSProp':
				optimizers = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
			elif self.optimizers == 'Momentum':
				optimizers = tf.train.MomentumOptimizer(learning_rate, self.momentum)
			else:
				print(self.optim_id, ' is not a valid ID, default Adam optimizer will be used.')
				optimizers = tf.train.AdamOptimizer(learning_rate=learning_rate)

			self.train_op = optimizers.minimize(self.loss1, var_list=var_list, \
				colocate_gradients_with_ops=True)


		# collect summaries
		self.overall_acc = tf.placeholder(dtype=tf.float32, shape=())
		self.mean_acc = tf.placeholder(dtype=tf.float32, shape=())
		if self.logdir is not None:
			# scalar loss summary
			tf.summary.scalar('eval/loss1', self.loss1)
			tf.summary.scalar('eval/overall_acc', self.overall_acc)	
			tf.summary.scalar('eval/mean_acc', self.mean_acc)	

			# input images summary
			tf.summary.image('in/image', images)
			tf.summary.image('in/label', tf.cast(labels, dtype=tf.float32))
			# tf.summary.image('in/label', tf.cast(tf.reshape(self.label_p, [1, 360, 480, 1]), dtype=tf.float32))
			
			summary_out = tf.cast(tf.reshape(tf.argmax(tf.nn.softmax(self.net.logits_b1[-1]), axis=3), [self.batch_size, 360, 480, 1]), dtype=tf.float32)
			tf.summary.image('out/dsn6', summary_out)
		
		return num_batch, total_images

	def train(self):
		num_batch, total_images = self.setup_train_pharse()
		max_ep = self.max_steps//num_batch
		print("max_ep = {}, num_batch = {}, total steps = {}".format(max_ep, num_batch, self.max_steps))
		
		# create session and start train session
		config = tf.ConfigProto(allow_soft_placement=True) 
		config.gpu_options.allow_growth=True # prevent the program occupies all GPU memory
		with tf.Session(config=config) as sess:
			# register current session to Keras
			# K.set_session(sess)
			# init all variables in graph
			sess.run(tf.group(tf.global_variables_initializer(),
							tf.local_variables_initializer()))

			# restore geonet
			geo_saver = tf.train.Saver([v for v in tf.trainable_variables() if 'fcn' in v.name])
			geo_saver.restore(sess, save_path=tf.train.latest_checkpoint(self.net.geo.pre_trained_model))

			if self.logdir is not None: # save or not
				# saver 
				saver = tf.train.Saver([v for v in tf.trainable_variables() if 'fcn' not in v.name], max_to_keep=self.keep_n_model)
				# filewriter for log info
				log_dir = self.logdir+'/run-%02d%02d-%02d%02d' % tuple(time.localtime(time.time()))[1:5]
				writer = tf.summary.FileWriter(log_dir)
				merged = tf.summary.merge_all()
			
			total_times = 0
			pre_metric_sum = 0.0

			print("Evaluating using random variables...")
			pre_metric_sum, oacc, macc = self.online_evaluate_acc(sess=sess, epoch=-1)

			print("Start Training!")
			for ep in range(max_ep): # epoch loop
				random.shuffle(self.paths)
				i = 0
				for n in range(num_batch): # batch loop (32 examples at each batch)
					step = int(ep*(num_batch) + n)
					# im, line_mask, plane_gt, line_gt = load_data(self.paths[n])
					# im, plane_gt = load_data2(self.paths[n])
					paths = self.paths[i:i+self.batch_size]
					i += self.batch_size
					i = total_images-self.batch_size if i > total_images-self.batch_size else i # limit index size

					im, plane_gt = load_batch(paths)
					tic = time.time()
					if self.logdir is not None:
						# infer loss & update parameters & summaries log				
						[loss_value1, _, summaries] = sess.run([self.loss1, \
							self.train_op, merged], \
							feed_dict={self.image:im, self.label_p:plane_gt, \
							self.overall_acc:oacc, self.mean_acc:macc, self.step_ph:step})
					else:
						# infer loss & update parameters
						[loss_value, update_value] = sess.run([self.loss1, self.train_op],
							feed_dict={self.image:im, self.label_p:plane_gt, \
							self.overall_acc:oacc, self.mean_acc:macc, self.step_ph:step})
					duration = time.time() - tic
					# save summaries every 1000 steps(iterations)
					if (step % self.summary_step == 0) and (self.logdir is not None): 
						writer.add_summary(summaries, global_step=step)

					total_times += duration

					# print out log 
					print('step {}: loss1 = {:.3}; {:.2} data/sec, excuted {} minutes'.format(step,
							loss_value1, 1.0/duration, int(total_times/60)))

				if ep % self.save_ep == 0:
					cur_metric_sum, oacc, macc = self.online_evaluate_acc(sess=sess, epoch=ep)
					# cur_metric_sum = sum_acc
					if cur_metric_sum > pre_metric_sum:
						pre_metric_sum = cur_metric_sum
						if self.logdir is not None:
							saver.save(sess, self.logdir + '/best_model', global_step=None) # best precision model
			# last epoch		
			# self.online_evaluate_acc(sess=sess, epoch=self.max_ep)
			# if self.logdir is not None:
			# 	saver.save(sess, self.logdir+'/model', global_step=self.max_ep)				
			
			# close session	
			sess.close()			

	# evaluation method
	def fast_hist(self, im, gt, n=8):
		"""
		n is num_of_classes
		"""
		k = (gt >= 0) & (gt < n)
		return np.bincount(n * gt[k].astype(int) + im[k], minlength=n**2).reshape(n, n)

	def cal_precision_recall_mae(self, prediction, gt):
		# input should be np array with data type uint8
		hard_gt = gt.astype(np.uint8)

		eps = 1e-4

		mae = np.mean(np.abs(prediction - gt))

		t = np.sum(hard_gt)

		precision, recall = [], []
		# calculating precision and recall at 255 different binarizing thresholds
		for threshold in range(256):
			threshold = threshold / 255.

			hard_prediction = np.zeros(prediction.shape, dtype=np.uint8)
			hard_prediction[prediction > threshold] = 1

			tp = np.sum(hard_prediction * hard_gt)
			p = np.sum(hard_prediction)

			precision.append((tp + eps) / (p + eps))
			recall.append((tp + eps) / (t + eps))

		return np.mean(precision), np.mean(recall), mae		

	def online_evaluate_acc(self, sess, epoch, th=0.5):
		# paths = glob.glob(test_paths)
		paths = open(test_file, 'r').read().splitlines()

		predict1 = tf.argmax(tf.nn.softmax(self.net.logits_b1[-1]), axis=3) # prediction last one

		num_of_classes1 = self.net.planar_num_classes

		hist1 = np.zeros((num_of_classes1, num_of_classes1))		

		total = len(paths)
		for n in range(0, total-self.batch_size, self.batch_size): # no batch during evaluation
			im, plane_gt = load_batch(paths[n:n+self.batch_size])
			
			[pred1] = sess.run([predict1], feed_dict={self.image:im})

			for i in range(self.batch_size):
				hist1 += self.fast_hist(np.squeeze(pred1[i,...]).flatten(), np.squeeze(plane_gt[i,...]).flatten(), n=num_of_classes1)

		overall_acc1 = np.diag(hist1).sum() / hist1.sum()		
		mean_acc1 = np.diag(hist1) / (hist1.sum(1) + 1e-6)
		iu = np.diag(hist1) / (hist1.sum(1) + 1e-6 + hist1.sum(0) - np.diag(hist1))

		sum_acc = (overall_acc1+np.nanmean(mean_acc1[1:]))#+(overall_acc2+np.nanmean(mean_acc2[1:]))

		file = open(self.eval_file, 'a')
		print('Model at epoch {}: \n \
		overall_acc1={:.4}, mean_acc1={:.4}, mean_iu1={:.4}, \n \
		(include background) mean_acc1={:.4}, mean_iu1={:.4}, \n \
		sum_acc = {}'.format(\
			epoch, overall_acc1, np.nanmean(mean_acc1[1:]), np.nanmean(iu[1:]), \
			np.nanmean(mean_acc1), np.nanmean(iu), \
			sum_acc), file=file)

		file.close()	

		return sum_acc, overall_acc1, np.nanmean(mean_acc1)

