import cv2
import numpy as np
from pylsd.lsd import lsd # for line segment detection
import matplotlib
matplotlib.use('agg') # without tkinter backend and plt.show() no effect
from matplotlib import pyplot as plt
from scipy import ndimage
# from scipy.misc import imread, imsave, imresize
from PIL import Image

infer_plane_map = {
	1: [255,   0,   0], # left plane
	2: [  0, 101, 255], # right plane
	3: [  0, 255, 101], # front plane
	4: [205, 255,   0], # ground
	5: [205, 255,   0], # sidewalk
}

def randomRGBColors(num=6):
	rgbs = []
	for i in xrange(num):
		rgbs.append(list((np.random.random(size=3)*256).astype(int)))
	return rgbs

def rgb2ind(im, color_map=infer_plane_map):
	"""
	Convert rgb plane map into index image based on the index<->rgb value mapping 
		defined in infer_plane_map

	Args:
		im: input image with shape = [h, w, 3]
		color_map: mapping of the index and rgb value, default use infer_plane_map
	Returns:
		index image with shape = [h, w]
	"""

	ind = np.zeros((im.shape[0], im.shape[1]))

	for i, rgb in color_map.items(): # use iteritems() instead when using Python2
		ind[(im==rgb).all(2)] = i	

	return ind.astype(np.uint8)

def ind2rgb(im, color_map=infer_plane_map):
	"""
	Convert ind image into rgb image based on given color map

	Args:
		im: input index image with shape = [h, w]
		color_map: mapping of the index and rgb value, default use infer_plane_map
	Returns:
		rgb image with shape = [h, w, 3]
	"""
	rgb_im = np.zeros((im.shape[0], im.shape[1], 3))

	for i, rgb in color_map.items():
		rgb_im[(im==i)] = rgb

	return rgb_im

def findLSDCand(im):
	"""
	Use LSD algorithm to extract candidate line segment 

	Args:
		im: input image with shape [h, w, 3]
	Returns:
		cand: a list of all candidate lines, endpoint format, [x1, x2, y1, y2, r], r is not used in this project
	"""
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	points = lsd(gray)
	return points


####################################################################################################
# drawing function for visulizations
####################################################################################################

def drawPlane(im, plane):
	"""
	Merge the plane mask & raw image based on the default index<->color mapping(infer_plane_map)

	Args:
		im: input image
		plane: input rgb plane
	Returns:
		None
	"""
	plane_ind = rgb2ind(plane)
	for i, rgb in infer_plane_map.items():
		for c in range(3):
			im[:,:,c] = np.where(plane_ind==i,
						im[:,:,c]*0.4+rgb[c]*0.6,
						im[:,:,c])

def drawPlaneInd(im, plane_ind, color_map=infer_plane_map):
	"""
	Merge the plane mask & raw image based on the default index<->color mapping(infer_plane_map)

	Args:
		im: input image
		plane_ind: input plane indexed image
	Returns:
		None
	"""
	for i, rgb in color_map.items():
		for c in range(3):
			im[:,:,c] = np.where(plane_ind==i,
						im[:,:,c]*0.4+rgb[c]*0.6,
						im[:,:,c])

def drawPlaneIndMergeGround(im, plane_ind, color_map=infer_plane_map):
	"""
	Merge the plane mask & raw image based on the default index<->color mapping(infer_plane_map)

	Args:
		im: input image
		plane_ind: input plane indexed image
	Returns:
		None
	"""
	for i, rgb in color_map.items():
		if i==4:
			rgb = color_map[5]
		for c in range(3):
			im[:,:,c] = np.where(plane_ind==i,
						im[:,:,c]*0.4+rgb[c]*0.6,
						im[:,:,c])

def drawPlaneMask(im, plane_mask, rgb):
	"""
	Merge one plane_mask & raw image based on the rgb args

	Args:
		im: input image, [h, w, 3]
		plane_mask: binary map of plane
		rgb: [r, g, b], a list of single rgb color value 
	Returns:
		None
	"""
	for c in range(3):
		im[:,:,c] = np.where(plane_mask!=0,
					im[:,:,c]*0.4+rgb[c]*0.6,
					im[:,:,c])

def drawLines(im, points, rgb=[255,255,0], thickness=1):
	"""
	Draw line on raw image by given rgb & thickness

	Args:
		im: input image with shape [h, w, 3]
		points: endpoint format of a list of lines
		rgb: a list of single rgb value of the line
		thickness: define the width of line plot on image
	Returns:
		None		
	"""
	for p in points:
		p1 = (int(p[0]), int(p[1]))
		p2 = (int(p[2]), int(p[3]))
		cv2.line(im, p1, p2, rgb, thickness)

####################################################################################################


