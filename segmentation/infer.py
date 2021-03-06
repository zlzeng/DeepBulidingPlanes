from model import *
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

logging.disable(logging.WARNING) # ignoring deprecated warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore warning messages

OUT_DIR = '../outputs/seg/demo'

OUTPUT_INDEX = True
OUTPUT_FUSE_RESULTS = True

COLOR_MAP = infer_plane_map

used_test_file = '../dataset/demo.txt'

if __name__ == '__main__':
	# seed = 8964
	tf.set_random_seed(seed)
	np.random.seed(seed)	
	random.seed(seed)

	# create output folder
	if not os.path.isdir(OUT_DIR):
		os.makedirs(OUT_DIR)

	# initial & restore trained model
	baseline = MODEL()
	baseline.setup_infer_pharse(OUTPUT_INDEX)

	# infer 
	paths = open(used_test_file, 'r').read().splitlines()
	total = len(paths)
	for n in range(total): # no batch during evaluation
		path = paths[n:n+1]
		im, _ = load_batch(path)

		feed_dict = {baseline.image:im}
		predict = baseline.sess.run([baseline.predict1], feed_dict=feed_dict)[0]

		if OUTPUT_INDEX:
			predict_rgb = ind2rgb(np.squeeze(predict), color_map=COLOR_MAP)

			save_path = os.path.join(OUT_DIR, path[0].split('\t')[0].split('/')[-1].split('.jpg')[0]+'.png')
			# imsave(save_path, predict_rgb)
			if OUTPUT_FUSE_RESULTS:
				out = (np.squeeze(im)*255)
				drawPlaneInd(out, np.squeeze(predict))
				Image.fromarray(out.astype(np.uint8)).save(save_path)
			else:
				Image.fromarray(predict_rgb.astype(np.uint8)).save(save_path)
			print('{}th image, save to {} [OK]'.format(n+1, save_path))
		else:
			predict_prob = np.squeeze(predict)

			save_path = os.path.join(OUT_DIR, path[0].split('\t')[0].split('/')[-1].split('.jpg')[0]+'.npy')
			np.save(save_path, predict_prob)
			print('{}th probabilities map, save to {} [OK]'.format(n+1, save_path))


	baseline.sess.close()
