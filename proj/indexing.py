#python3 indexing.py --path=album/
import cv2 as cv
import face_recognition
import argparse
import sys
import numpy as np
import os.path
import glob
import pickle
import nltk
from utils import serialize_obj, get_output_names, draw_pred, salience_score
from tqdm import tqdm
from text_recognizer import verifyText
import collections
import re

# Initialize the parameters and general config
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image
person_class_id = 0
# Load names of classes
classesFile = "coco.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

modelConfigurationTree = "yolov3-tiny.cfg"
modelWeightsTree = "yolov3-tiny_final.weights"


def add_to_index(index, indexText,imgs_info, img_info, img_text,file):
    for key, value in img_info.items():
        if key in index:
            lst = index[key]
            lst.append(file)
            index[key] = lst
        else:
            index[key] = [file]

    for key, value in img_text.items():
        if key in indexText:
            lst = indexText[key]
            lst.append((file,value))
            indexText[key] = lst
        else:
            indexText[key] = [(file,value)]

    imgs_info[file] = img_info


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(index, indexText, imgs_info, face_encodings, image, file, outs, outsTree,sno, debug = False):
    image_height = image.shape[0]
    image_width = image.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    class_ids = []
    confidences = []
    boxes = []
    # has count and salience score
    img_info = {}

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confThreshold:
                center_x = int(detection[0] * image_width)
                center_y = int(detection[1] * image_height)
                width = int(detection[2] * image_width)
                height = int(detection[3] * image_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    for out in outsTree:
        for detection in out:
            scores = detection[5:]
            class_id = -1
            confidence = scores[0]
            if confidence > confThreshold:
                center_x = int(detection[0] * image_width)
                center_y = int(detection[1] * image_height)
                width = int(detection[2] * image_width)
                height = int(detection[3] * image_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    saliency = cv.saliency.StaticSaliencySpectralResidual_create()
    _, saliency_map = saliency.computeSaliency(image)
    if debug:
        saliency_map_u = (saliency_map * 255).astype("uint8")
    # convert to int if indexing takes too long
    # saliency_map = (saliency_map * 255).astype("uint8")

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    # what actually gets added
    for i in indices:
        i = i[0]
        class_id = class_ids[i]
        box = boxes[i]

        # (count, salience_score)
        class_tup = img_info.get(class_id, (0, 0))

        img_info[class_id] = (class_tup[0] + 1, class_tup[1] + salience_score(saliency_map, box))

        if debug:
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            draw_pred(image, classes, class_ids[i], confidences[i], left, top, left + width, top + height)
            draw_pred(saliency_map_u, classes, class_ids[i], confidences[i], left, top, left + width, top + height)


    if person_class_id in img_info:
        if debug:
            print("encoding person: " + file)
        # switch from BGR to RGB
        picture = image[:, :, ::-1]
        pic_encodings = face_recognition.face_encodings(picture)
        if len(pic_encodings) > 0:
            face_encodings[file] = pic_encodings


    resultText = verifyText(image)
    img_text = {}
    if len(resultText)!= 0:
        textReged = list(map(lambda word: re.sub('[^A-Za-z0-9]+', '', word),resultText))
        textStemmed = list(map(lambda word: sno.stem(word), textReged))
        img_text=collections.Counter(textStemmed)
    if debug:
        cv.imshow("img", image)
        cv.imshow("saliency_map", saliency_map_u)
        # cv.imwrite("../report/Figures/salience_image_example_bbox.jpg", image)
        # cv.imwrite("../report/Figures/salience_map_bbox.jpg", saliency_map_u)
        cv.waitKey(0)
        print(img_info)



    add_to_index(index, indexText,imgs_info, img_info, img_text, file)
    return img_info


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
    parser.add_argument('--path', help='Path to image album')
    parser.add_argument('-d', action='store_true', help="show debug info")

    args = parser.parse_args()

    debug = False
    if args.d:
        debug = True

    imagesName = [file for file in glob.glob(args.path + "/*")] + glob.glob(args.path)

    sno = nltk.stem.SnowballStemmer('english')

    index = {}
    indexText = {}
    imgs_info = {}
    face_encodings = {}
    #model bears
#    netBears = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
#    netBears.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
#    netBears.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    tqdm.write("Indexing...")

    netTree = cv.dnn.readNetFromDarknet(modelConfigurationTree, modelWeightsTree)
    netTree.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    netTree.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    loop_range = range(len(imagesName)) if debug else tqdm(range(len(imagesName)))

    for i in loop_range:

        file = imagesName[i]
        if os.path.isdir(file):
            continue

        image = cv.imread(file)

        if image is None:
            tqdm.write("Error in reading file {}".format(file))
            continue

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(image, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        # Sets the input to the network
        net.setInput(blob)
        netTree.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_output_names(net))
        outsTree = netTree.forward(get_output_names(netTree))
        # Remove the bounding boxes with low confidence

        img_info = postprocess(index, indexText, imgs_info, face_encodings, image, file, outs, outsTree,sno, debug)
        if debug:
            print(img_info)

    serialize_obj(index, "index")
    serialize_obj(indexText, "indexText")
    serialize_obj(imgs_info, "imgs_info")
    serialize_obj(face_encodings, "face_encodings")

    tqdm.write("Done!")
    if debug:
        print("Index:", index)
        print("Images info:", imgs_info)
