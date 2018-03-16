import cv2
import os
import numpy as np
from preprocessing import parse_annotation
from frontend import YOLO
import json
from utils import draw_boxes


def _main_():
    config_path = 'tanisorn_config.json'

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations
    ###############################

    # parse annotations of the training set
    train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'],
                                                config['train']['train_image_folder'],
                                                config['model']['labels'])

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(config['valid']['valid_annot_folder']):
        valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'],
                                                    config['valid']['valid_image_folder'],
                                                    config['model']['labels'])
    else:
        train_valid_split = int(0.8 * len(train_imgs))
        np.random.shuffle(train_imgs)

        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]

    overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

    print 'Seen labels:\t', train_labels
    print 'Given labels:\t', config['model']['labels']
    print 'Overlap labels:\t', overlap_labels

    if len(overlap_labels) < len(config['model']['labels']):
        print 'Some labels have no images! Please revise the list of labels in the config.json file!'
        return

    ###############################
    #   Construct the model
    ###############################

    yolo = YOLO(architecture=config['model']['architecture'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    ###############################
    #   Load the pretrained weights (if any)
    ###############################

    if os.path.exists(config['train']['pretrained_weights']):
        print "Loading pre-trained weights in", config['train']['pretrained_weights']
        yolo.load_weights(config['train']['pretrained_weights'])
        #########################

        ###############################
        #   Start the training process
        ###############################
    image = cv2.imread('/home/palmy/PycharmProjects/DATA/Tanisorn/imgCarResize/CAR0043.jpg')
    image = cv2.resize(image, (416, 416))

    boxes = yolo.predict(image)
    sc = 0
    for box in boxes:
        if box.score > sc:
            sc = box.score

    print sc
    image = draw_boxes(image, boxes, config['model']['labels'])
    cv2.imwrite('car' + '_detected' + '.jpg', image)


if __name__ == '__main__':
    _main_()
