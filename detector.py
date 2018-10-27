import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
from glob import glob
import os
from keras.models import load_model
from enum import Enum
from moviepy.editor import *

class TL(Enum):
    UNKNOWN = 0
    RED = 1
    GREEN = 2
    YELLOW = 3

class Detector(object):
    def __init__(self):

        cwd = os.path.dirname(os.path.realpath(__file__))

        # load keras Lenet style model from file
        self.class_model = load_model( cwd+'/model.h5' )
        self.class_graph = tf.get_default_graph()

        # detection graph
        self.dg = tf.Graph()

        #model files
        rcnn_inception_resnet_v2 = "/models/faster_rcnn_inception_resnet_v2_atrous_coco/frozen_inference_graph.pb"
        rcnn_inception_resnet_101_faster = "/models/faster_rcnn_resnet101_coco/frozen_inference_graph.pb"
        ssd_mobilenet_v1 = "/models/frozen_inference_graph.pb"
        mfiles = [rcnn_inception_resnet_v2, rcnn_inception_resnet_101_faster, ssd_mobilenet_v1]

        # load
        with self.dg.as_default():
            gdef = tf.GraphDef()
            with open(cwd+ mfiles[1], 'rb') as f:
                gdef.ParseFromString( f.read() )
                tf.import_graph_def( gdef, name="" )

            # get names of nodes. 
            # from https://www.activestate.com/blog/2017/08/using-pre-trained-models-tensorflow-go
            self.session = tf.Session(graph=self.dg )
            self.image_tensor = self.dg.get_tensor_by_name('image_tensor:0')
            self.detection_boxes =  self.dg.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.dg.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.dg.get_tensor_by_name('detection_classes:0')
            self.num_detections    = self.dg.get_tensor_by_name('num_detections:0')

        self.tlclasses = [ TL.RED, TL.YELLOW, TL.GREEN ]
        self.tlclasses_d = { TL.RED : "RED", TL.YELLOW:"YELLOW", TL.GREEN:"GREEN", TL.UNKNOWN:"UNKNOWN" }

    def detect(self, image):
        """
        """
        ret = []

        boxes = self.locate( image )


        for box in boxes:
            class_image = cv2.resize( image[box[0]:box[2], box[1]:box[3]], (32,32) )
            status = self.classify( class_image )
            if status != TL.UNKNOWN:
                ret.append((status, box));

        return ret






    def locate(self, image):
        """ Find bounding boxes for lights using pretrained tensorflow model
            input format is  BGR8 image

            Returns: a box where the most likely location for the traffic signal is located. None if not found
        """

        ret = []
        with self.dg.as_default():
            #switch from BGR to RGB. Important otherwise detection won't work
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            tf_image_input = np.expand_dims(image,axis=0)
            
            #run detection model
            (detection_boxes, detection_scores, detection_classes, num_detections) = self.session.run(
                    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: tf_image_input})

            detection_boxes = np.squeeze(detection_boxes)
            detection_classes = np.squeeze(detection_classes)
            detection_scores = np.squeeze(detection_scores)


            #print(detection_classes)
            #print(detection_scores)
            #print(detection_boxes)

            detection_threshold = 0.8

            # Find first detection of signal. It's labeled with number 10
            tlboxes = []
            for i, cl in enumerate(detection_classes.tolist()):
                if cl == 10:
                    tlboxes.append(i)

            print("found {} boxes".format(len(tlboxes)))
            for idx in tlboxes:
                if idx == -1:
                    pass  # no signals detected
                elif detection_scores[idx] < detection_threshold:
                    pass # we are not confident of detection
                else:
                    dim = image.shape[0:2]
                    box = self.from_normalized_dims__to_pixel(detection_boxes[idx], dim)
                    box_h, box_w  = (box[2] - box[0], box[3]-box[1] )
                    if (box_h <20) or (box_w<20):
                        pass    # box too small
                    elif ( box_h/box_w <1.6):
                        pass    # wrong ratio
                    else:
                        print('detected bounding box: {} conf: {}'.format(box, detection_scores[idx]))
                        ret.append( box )

        return ret
        
    def from_normalized_dims__to_pixel(self, box, dim):
            height, width = dim[0], dim[1]
            box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
            return np.array(box_pixel)


    def draw_box(self, img, box, text=""):
        cv2.rectangle(img, (box[1],box[0]), (box[3],box[2]), (255,0,0), 5)
        cv2.putText(img, text, (box[1],box[0]-10), cv2.FONT_HERSHEY_SIMPLEX,  1, (255,0,0), 2  )
        return img


    def classify(self, image):
        """ Given a 32x32x3 image classifies it as red, greed or yellow
            Expects images in BGR format. Important otherwide won't classify correctly
            
        """
        status = TL.UNKNOWN
        img_resize = np.expand_dims(image, axis=0).astype('float32')
        with self.class_graph.as_default():
            predict = self.class_model.predict(img_resize)
            status  = self.tlclasses[ np.argmax(predict) ]

        return status








if __name__ == '__main__':
    detector = Detector()

    # enable for image detection
    # parses all images under images/
    if True:
        paths = glob(os.path.join('images/', '*.jpg'))
        for path in paths:
            img = cv2.imread(path)
            detections = detector.detect( img )

            for d in detections:
                color = detector.tlclasses_d[d[0]]
                print ( color, d[1], path)
                detector.draw_box(img, d[1], color)

            if len(detections):
                cv2.imwrite("out/"+path, img)

    #enable for video detection.
    # parses the videos speciried in vfiles
    if True:

        vfiles = ['castro3.mp4', 'castro4.mp4' ]
        for vf in vfiles:
            #videos
            clip = VideoFileClip('videos/'+vf)

            def process_frame(img):
                detections = detector.detect( img )

                for d in detections:
                    color = detector.tlclasses_d[d[0]]
                    detector.draw_box(img, d[1], color)

                return img

            outclip = clip.fl_image(process_frame)
            outclip.write_videofile('out/video/'+vf)





