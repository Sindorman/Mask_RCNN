#!/usr/bin/env python3

import os
import sys
import random
import math
from time import time
from os import path
import uuid

import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from werkzeug import secure_filename

def build_model():
    # Root directory of the project
    ROOT_DIR = os.path.abspath("../")

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    from mrcnn import utils
    import mrcnn.model as modellib
    from mrcnn import visualize
    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
    import coco

    #%matplotlib inline 

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()


    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    print("Loading weights for model...")
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    return model


app = Flask(__name__)
USER_IMG_DIR = "user_imgs/"

class MaskGenerator():
    def __init__(self):
        self.membervar = 5
        app.run(debug = True, use_reloader=False, host="0.0.0.0", port=80) # port 80 means sudo only :/


    @app.route('/upload_page')
    def upload_page(self):
        print("/upload_page called in A CLASS")
        return render_template('upload_page.html')
        
    @app.route('/uploader', methods = ['POST'])
    def upload_file(self):
        print("/uploader called")
        if request.method == 'POST':
            f = request.files['file']
            rand_uuid = uuid.uuid4().hex
            ofname = path.join(USER_IMG_DIR, rand_uuid + "_" + secure_filename(f.filename))
            print("writing to file '{}' ...".format(ofname))
            f.save(ofname)
            return 'file uploaded successfully'
		
if __name__ == '__main__':
    #model = build_model()

    # use_reloader on makes us load the model twice (this is slow and bad)
    maskGenerator = MaskGenerator()
'''

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']



# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
print("detecting objects...")
start_time = time()
results = model.detect([image], verbose=1)
end_time = time()
secs_elapsed = end_time - start_time
print("detection took {:0.2f} seconds.".format(secs_elapsed))

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])

print("rois (shape={}): {}".format(r['rois'].shape, r['rois']))
print("masks (shape={}): {}".format(r['masks'].shape, r['masks']))
print("class_ids (shape={}): {}".format(r['class_ids'].shape, r['class_ids']))
print("scores (shape={}): {}".format(r['scores'].shape, r['scores']))
print("STATIC class_names (len={}): {}".format(len(class_names), class_names))

'''