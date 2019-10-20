import cv2
import numpy as np



def detect(img):
	#passing yolo weights and cfg#
	NN=cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")
	layer_name=NN.getLayerNames()

	output_layer = [layer_name[i[0] - 1] for i in NN.getUnconnectedOutLayers()]

	#reading the dataset#
	file=open("coco.names")
	classes=[]
	for ch in file.readlines():
		classes.append(ch.strip('\n'))

	#img=cv2.resize(img,(416,416),interpolation=cv2.INTER_AREA)

	#detecting the objects present in image#
	blobs = cv2.dnn.blobFromImage(img,1 / 255.0, (416, 416),swapRB=True, crop=False)
	#passing the detected blobs to neural network#
	NN.setInput(blobs)
	#getting output#
	output_neuron=NN.forward(output_layer)
	(H, W) = img.shape[:2]
	boxes = []
	confidences = []
	classIDs = []
	for out in output_neuron:
		for detection in out:
			scores=detection[5:]
			class_id=np.argmax(scores)
			confidence=scores[class_id]
			if confidence>=0.2 and (classes[class_id] in ['car','bus','motorbike','truck','bicycle']):
				#print(detection[1],img.shape[1])
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(class_id)

	#Remove the overlapped rectangle through threshold#
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2,0.3)
	count=0
	for i in range(len(boxes)):
		if i in indexes.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the image
			#color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
			count=count+1

	return count