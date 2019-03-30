#python3 search.py --path=car.jpg
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import glob
from utils import salience_score, draw_pred, get_output_names, deserialize_obj
import math
import collections
import re
import nltk
import operator
import face_recognition

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image
classesFile = "coco.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
person_class_id = 0

modelConfigurationTree = "yolov3-tiny.cfg"
modelWeightsTree = "yolov3-tiny_final.weights"


def search_by_text(query, indexText):
    result = {}

    query = query.split()
    sno = nltk.stem.SnowballStemmer('english')
    queryReged = list(map(lambda word: re.sub('[^A-Za-z0-9]+', '', word),query))
    queryStemmed = list(map(lambda word: sno.stem(word), queryReged))

    totalToNormalize = 0
    for term in queryStemmed:
        if term in indexText:
            lst = indexText[term]
            for x,y in lst:
                if x in result:
                    result[x]+=y
                else:
                    result[x]=y
                totalToNormalize+=y
    for key in result:
        result[key] = result[key]/totalToNormalize
    print(indexText)

    return sorted(result.items(), key=operator.itemgetter(1))



def search_index(img_info, imgs_info, pic_encodings, face_encodings, index):
    # union of all matches
    rel_images = []
    for obj, _ in img_info.items():
        if obj in index:
            rel_images += index[obj]

    rel_images = set(rel_images)

    face_disc = {}

    for img in face_encodings:
        if len(pic_encodings) > 0:
            face_disc[img] = 0
            for pic_encoding in pic_encodings:
                face_disc[img] = min([face_discount(dist) for dist in
                                      face_recognition.face_distance(face_encodings[img], pic_encoding)])
    return sorted([(img, obj_dist(img_info, imgs_info[img]) + face_disc.get(img, 0))\
                   for img in rel_images], key=lambda res_tup: res_tup[1])


def obj_dist(img_info_query, img_info_doc):
    cost = 0
    for obj in set(img_info_query.keys()) | set(img_info_doc.keys()):
        y, y_hat, y_imp, y_hat_imp = 0, 0, 0, 0
        if obj in img_info_query:
            y, y_imp = img_info_query.get(obj)
        if obj in img_info_doc:
            y_hat, y_hat_imp = img_info_doc.get(obj)

        cost += obj_cost(y, y_imp, y_hat, y_hat_imp)

    return cost


# y is original vector
def obj_cost(y, y_salience, y_hat, y_hat_salience):
    if y == 0:
        if y_hat == 0:
            return 0
        else:
            return math.log(1 + y_hat)*math.e**y_salience
    else:
        if y_hat == 0:
            return (1 + math.log(y)) * math.e ** y_salience
        else:
            return abs(math.log(y) - math.log(y_hat))*abs(y_salience - y_hat_salience)


def face_discount(face_dist):
    if face_dist <= 0.6:
        return -10
    elif 0.6 < face_dist < 1:
        # make it so out is [0, 1] and then apply log to it
        return max(math.log((face_dist - 0.6)*2.5), -10)
    else:
        return 0


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(image, outs, outsTree,debug=False):
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    image_height = image.shape[0]
    image_width = image.shape[1]

    class_ids = []
    confidences = []
    boxes = []
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

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

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

    if debug:
        cv.imshow("img", image)
        cv.imshow("saliency_map", saliency_map)
        cv.waitKey(0)
        print(img_info)

    return img_info


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
    parser.add_argument('--image', help='Path to images file.')
    parser.add_argument('--text', help='Text to search for image')
    parser.add_argument('-d', action='store_true', help="show debug info")

    args = parser.parse_args()

    debug = False
    if args.d:
        debug = True

    if args.text == None:

        file = args.image
        image = cv.imread(file)

        index = deserialize_obj("index")
        imgs_info = deserialize_obj("imgs_info")
        face_encodings = deserialize_obj("face_encodings")

        net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


        netTree = cv.dnn.readNetFromDarknet(modelConfigurationTree, modelWeightsTree)
        netTree.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        netTree.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(image, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        # Sets the input to the network
        net.setInput(blob)
        netTree.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_output_names(net))
        outsTree = netTree.forward(get_output_names(netTree))
        # Remove the bounding boxes with low confidence
        img_info = postprocess(image, outs, outsTree, debug)

        # BGR to RGB
        unk_image = image[:, :, ::-1]
        pic_locations = face_recognition.face_locations(unk_image)
        pic_encodings = face_recognition.face_encodings(unk_image, pic_locations)
        res = search_index(img_info, imgs_info, pic_encodings, face_encodings, index)
        print(res)
        i = 0
        for r in res:
            i+=1
            if(i>5):
                break
            cv.imshow("result {}".format(i), cv.imread(r[0]))
            cv.waitKey(0)

    else:
        indexText = deserialize_obj("indexText")
        res = search_by_text(args.text, indexText)
        print(res)
        i = 0
        for r in res:
            i += 1
            if (i > 5):
                break
            cv.imshow("result {}".format(i), cv.imread(r[0]))
            cv.waitKey(0)
