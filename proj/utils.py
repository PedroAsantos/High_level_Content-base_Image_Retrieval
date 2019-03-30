import cv2 as cv
import pickle


# Get the names of the output layers
def get_output_names(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def deserialize_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def serialize_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# Draw the predicted bounding box
def draw_pred(frame, classes, classId, conf, left, top, right, bottom, draw_label = False):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        print(classes[classId])
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    if draw_label:
        label_size, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - round(0.5*label_size[1])), (left + round(1.5*label_size[0]), top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


# salience score given by percentage of attention it gets
def salience_score(saliency_map, bbox):
    total = saliency_map.sum()

    left = bbox[0]
    top = bbox[1]
    width = bbox[2]
    height = bbox[3]

    roi = saliency_map[left:left+width, top:top+height]
    return roi.sum() / total
