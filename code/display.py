import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util

def DisplayDetections( image, boxes_list, boxes_list_1, boxes_list_2, det_time=None):
		if not boxes_list: return image  # input list is empty
		img = image.copy()
		# if not boxes_list_1: return image_1  # input list is empty
		# img_1 = image_1.copy()
		# if not boxes_list_2: return image_2  # input list is empty
		# img_2 = image_2.copy()
		for idx in range(len(boxes_list)):
			x_min = boxes_list[idx][0]
			y_min = boxes_list[idx][1]
			x_max = boxes_list[idx][2]
			y_max = boxes_list[idx][3]
			cls =  str(boxes_list[idx][4])
			score = str(np.round(boxes_list[idx][-1], 2))

			text = cls + ": " + score
			cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
			cv2.rectangle(img, (x_min, y_min - 20), (x_min, y_min), (255, 255, 255), -1)
			cv2.putText(img, text, (x_min + 5, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

		for idx in range(len(boxes_list_1)):
			x_min = boxes_list_1[idx][0]
			y_min = boxes_list_1[idx][1]
			x_max = boxes_list_1[idx][2]
			y_max = boxes_list_1[idx][3]
			cls =  str(boxes_list_1[idx][4])
			score = str(np.round(boxes_list_1[idx][-1], 2))

			text = cls + ": " + score
			cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)
			cv2.rectangle(img, (x_min, y_min - 20), (x_min, y_min), (255, 255, 255), -1)
			cv2.putText(img, text, (x_min + 5, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

		for idx in range(len(boxes_list_2)):
			x_min = boxes_list_2[idx][0]
			y_min = boxes_list_2[idx][1]
			x_max = boxes_list_2[idx][2]
			y_max = boxes_list_2[idx][3]
			cls =  str(boxes_list_2[idx][4])
			score = str(np.round(boxes_list_2[idx][-1], 2))

			text = cls + ": " + score
			cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
			cv2.rectangle(img, (x_min, y_min - 20), (x_min, y_min), (255, 255, 255), -1)
			cv2.putText(img, text, (x_min + 5, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

		if det_time != None:
			fps = round(1000. / det_time, 1)
			fps_txt = str(fps) + " FPS"
			cv2.putText(img, fps_txt, (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
			# cv2.putText(img_1, fps_txt, (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
			# cv2.putText(img_2, fps_txt, (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

		return img
