import os
import sys
import glob
import time
import random

import json
from utils import *

src_dir = '../dataset'

train_file = '../dataset/train.txt'
test_file = '../dataset/test.txt' # all data set

seed = 8964 # random seed
random.seed(seed)

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

def normalize_meanstd(a, axis=(1,2)): 
	# axis param denotes axes along which mean & std reductions are to be performed
	mean = np.mean(a, axis=axis, keepdims=True)
	std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
	return (a - mean) / std

def gen_line_gt(json_path, thickness=2):
	f = open(json_path, 'r')
	data = json.load(f)

	h, w = data['imageHeight'], data['imageWidth']
	line_ind = np.zeros((h, w), dtype=np.uint8)
	# line_mask = np.zeros(im.shape[:2])

	# coords = []
	for sh in data['shapes']:
		if sh['label'] not in ['0', '1']:
			continue
		points = []
		for s in sh['points']:
			points.append([int(t) for t in s])

		ep = [tuple(s) for s in points]
		if len(ep)<2:
			continue
		# coords.append([list(s) for s in sh['points']])
		ind = int(sh['label'])
		cv2.line(line_ind, ep[0], ep[1], ind, thickness)
		# cv2.line(line_mask, ep[0], ep[1], 1, thickness)

	return line_ind

def gen_line_mask(im, thickness=2):
	points = findLSDCand(im)

	line_mask = np.zeros(im.shape[:2])
	drawLines(line_mask, points, 1, thickness)

	return line_mask, points


def gen_height_mask(json_path, thickness=2):
	f = open(json_path, 'r')
	data = json.load(f)

	h, w = data['imageHeight'], data['imageWidth']
	height_ind = np.zeros((h, w), dtype=np.uint8)
	height_mask = np.zeros((h, w))

	# coords = []
	for sh in data['shapes']:
		if sh['label'] not in ['0', '1', '2', '3', '4', '5']:
			continue
		ep = [tuple(s) for s in sh['points']]
		if len(ep) < 2:
			continue
		# coords.append([list(s) for s in sh['points']])
		ind = int(sh['label'])
		cv2.line(height_ind, ep[0], ep[1], ind, thickness)
		# cv2.line(height_mask, ep[0], ep[1], 1, thickness)
	height_mask[height_ind!=0] = 1

	return height_mask, height_ind

def load_data(file_path, infer=False, dtype=np.float32):
	im_path = file_path.split('\t')[0]
	plane_path = file_path.split('\t')[1]

	name = im_path.split('/')[-1].split('.jpg')[0]
	line_path = os.path.join(plane_path.split('_seg')[0] +'.json')

	# load raw data
	# im = imread(im_path, mode='RGB')
	# plane_gt = imread(plane_path, mode='L')
	im = np.array(Image.open(im_path).convert('RGB'))
	plane_gt = np.array(Image.open(plane_path).convert('L'))

	line, endpoints = gen_line_mask(im)
	line_gt = gen_line_gt(line_path)

	# force to specific data dtype
	im_norm = im / 255.
	im_norm = im_norm.astype(dtype)
	line = line.astype(dtype)
	line_gt = line_gt.astype(np.uint8)
	plane_gt = plane_gt.astype(np.uint8)

	# force to specific shape
	[h, w, c] = im.shape
	im_norm = np.reshape(im_norm, (1, h, w, c))
	line = np.reshape(line, (1, h, w, 1))
	line_gt = np.reshape(line_gt, (1, h, w, ))
	plane_gt = np.reshape(plane_gt, (1, h, w, ))

	# return values
	if infer:
		return im, im_norm, line, endpoints
	else:
		return im_norm, line, plane_gt, line_gt

def load_data2(file_path, hi_res=False, dtype=np.float32):
	im_path = file_path.split('\t')[0]
	gt_path = file_path.split('\t')[1].split('_seg.png')[0]+'.png'

	# load raw data
	# im = imread(im_path, mode='RGB')
	# gt = imread(gt_path, mode='RGB')
	# gt = imresize(gt, im.shape)
	im = np.array(Image.open(im_path).convert('RGB'))
	gt = np.array(Image.open(gt_path).convert('RGB').resize((im.shape[1], im.shape[0]), Image.NEAREST))

	# resize and normalize
	if hi_res:
		# im = imresize(im, (500,500,3))
		# gt = imresize(gt, (500,500,1))
		im = np.array(Image.fromarray(im.astype(np.uint8)).resize((500, 500)), np.float32)
		gt = np.array(Image.fromarray(gt.astype(np.uint8)).resize((500, 500), Image.NEAREST), np.float32)

	im_norm = im / 255.	
	gt = rgb2ind(gt)

	# set dtype
	im_norm = im_norm.astype(dtype)
	gt = gt.astype(np.uint8)

	# set shape
	[h, w, c] = im.shape
	im_norm = np.reshape(im_norm, (1, h, w, c))
	gt = np.reshape(gt, (1, h, w, ))

	return im_norm, gt

def load_batch(file_paths, dtype=np.float32):

	images, labels = [], []
	for file_path in file_paths:
		im_path = os.path.join(src_dir, file_path.split('\t')[0])
		gt_path = os.path.join(src_dir, file_path.split('\t')[1])
		
		im = np.array(Image.open(im_path).convert('RGB'))
		gt = np.array(Image.open(gt_path).convert('L'))		

		# im_norm = normalize_meanstd(im)
		im_norm = im / 255.
		# gt = rgb2ind(gt)

		# set dtype
		im_norm = im_norm.astype(dtype)
		gt = gt.astype(np.uint8)

		# set shape
		[h, w, c] = im.shape
		im_norm = np.reshape(im_norm, (1, h, w, c))
		gt = np.reshape(gt, (1, h, w, 1))
		images.append(im_norm)
		labels.append(gt)

	images = np.concatenate(images)
	labels = np.concatenate(labels)

	return images, labels
