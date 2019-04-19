import numpy as np
import imutils
import time
import cv2
import os

# load the COCO class labels our Mask R-CNN was trained on
labelsPath = os.path.sep.join(["mask-rcnn-coco","object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join(["mask-rcnn-coco","frozen_inference_graph.pb"])
configPath = os.path.sep.join(["mask-rcnn-coco","mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# load our Mask R-CNN trained on the COCO dataset (90 classes)
# from disk
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# initialize the video stream and pointer to output video file
#vs = cv2.VideoCapture("videos/Dog rides her Horse.mp4")
vs = cv2.VideoCapture(0)
writer = None

while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # construct a blob from the input frame and then perform a
    # forward pass of the Mask R-CNN, giving us (1) the bounding box
    # coordinates of the objects in the image along with (2) the
    # pixel-wise segmentation for each specific object
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    (boxes, masks) = net.forward(["detection_out_final",
        "detection_masks"])
    end = time.time()

    # loop over the number of detected objects
    for i in range(0, boxes.shape[2]):
        # extract the class ID of the detection along with the
        # confidence (i.e., probability) associated with the
        # prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > 0.5:
            # scale the bounding box coordinates back relative to the
            # size of the frame and then compute the width and the
            # height of the bounding box
            (H, W) = frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            # extract the pixel-wise segmentation for the object,
            # resize the mask such that it's the same dimensions of
            # the bounding box, and then finally threshold to create
            # a *binary* mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH),
                interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.3)

            # extract the ROI of the image but *only* extracted the
            # masked region of the ROI
            roi = frame[startY:endY, startX:endX][mask]

            # grab the color used to visualize this particular class,
            # then create a transparent overlay by blending the color
            # with the ROI
            color = COLORS[classID]
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            # store the blended ROI in the original frame
            frame[startY:endY, startX:endX][mask] = blended

            # draw the bounding box of the instance on the frame
            color = [int(c) for c in color]
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                color, 2)

            # draw the predicted label and associated probability of
            # the instance segmentation on the frame
            text = "{}: {:.4f}".format(LABELS[classID], confidence)
            cv2.putText(frame, text, (startX, startY - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()
