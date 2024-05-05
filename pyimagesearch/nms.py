# import the necessary packages
import numpy as np

# Malisiewicz et al.
def non_max_suppression(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes list
 
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# print("Shapes before deletion:")
		# print("idxs shape:", idxs.shape)
		# # Check the shapes of the arrays
		# print("Overlap shape:", overlap.shape)
		# print("OverlapThresh shape:", overlapThresh.shape)

		# # If overlapThresh has more elements, trim it
		# if overlapThresh.shape[0] > overlap.shape[0]:
		# 	overlapThresh = overlapThresh[:overlap.shape[0]]

		# # If overlap has more elements, pad it
		# elif overlap.shape[0] > overlapThresh.shape[0]:
		# 	# Dummy value to add to overlap
		# 	dummy_value = 0.0  # Adjust this as needed
		# 	overlap = np.append(overlap, dummy_value)

		# # Now, both arrays should have the same shape
		# print("Adjusted Overlap shape:", overlap.shape)
		# print("Adjusted OverlapThresh shape:", overlapThresh.shape)

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using theinteger data type
	return boxes[pick].astype("int")