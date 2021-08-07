import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util

class DetectorTF1:
		
	def __init__(self, saved_all, label_all, class_id=None, threshold=0.7):
		# class_id is list of ids for desired classes, or None for all classes in the labelmap
		self.class_id = class_id
		# print("___________________________________________________")	
		# print(self.class_id)
		# print("_____________________________________________________")
		self.Threshold = threshold
		# Loading label map
		#container
		label_map = label_map_util.load_labelmap(label_all)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
		self.category_index = label_map_util.create_category_index(categories)
	
		tf.keras.backend.clear_session()
		self.detect_fn = tf.saved_model.load(saved_all)


	def DetectFromImage(self, img):
		im_height, im_width, _ = img.shape
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		input_tensor = np.expand_dims(img, 0)
		detections= self.detect_fn(input_tensor)
		# detections_behind =self.detect_fn_1(input_tensor)
		# detections_infor = self.detect_fn_2(input_tensor)

		bboxes = detections['detection_boxes'][0].numpy()
		bclasses = detections['detection_classes'][0].numpy().astype(np.int32)
		bscores = detections['detection_scores'][0].numpy()
		det_boxes = self.ExtractBBoxes(bboxes, bclasses, bscores, im_width, im_height)
		# crop= self.Cropped(bboxes, bclasses, bscores, im_width, im_height, img)

		return det_boxes
		


	def ExtractBBoxes(self, bboxes, bclasses, bscores, im_width, im_height):
		bbox = []
		for idx in range(len(bboxes)):
			if self.class_id is None or bclasses[idx] in self.class_id:
				if bscores[idx] >= self.Threshold:
					y_min = int(bboxes[idx][0] * im_height)
					x_min = int(bboxes[idx][1] * im_width)
					y_max = int(bboxes[idx][2] * im_height)
					x_max = int(bboxes[idx][3] * im_width)
					class_label = self.category_index[int(bclasses[idx])]['name']
					bbox.append([x_min, y_min, x_max, y_max, class_label, float(bscores[idx])])

		return bbox


