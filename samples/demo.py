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
#from skimage.transform import resize
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
#from scipy import misc
from PIL import Image


from flask import Flask, render_template, request, jsonify
from werkzeug import secure_filename
from scipy.ndimage.morphology import binary_dilation

from flask import after_this_request, request
#from cStringIO import StringIO as IO
#from io import StringIO as IO
from io import BytesIO as IO

import gzip
import functools 

def gzipped(f):
    @functools.wraps(f)
    def view_func(*args, **kwargs):
        @after_this_request
        def zipper(response):
            accept_encoding = request.headers.get('Accept-Encoding', '')

            if 'gzip' not in accept_encoding.lower():
                return response

            response.direct_passthrough = False

            if (response.status_code < 200 or
                response.status_code >= 300 or
                'Content-Encoding' in response.headers):
                return response
            gzip_buffer = IO()
            gzip_file = gzip.GzipFile(mode='wb', 
                                      fileobj=gzip_buffer)
            gzip_file.write(response.data)
            gzip_file.close()

            response.data = gzip_buffer.getvalue()
            response.headers['Content-Encoding'] = 'gzip'
            response.headers['Vary'] = 'Accept-Encoding'
            response.headers['Content-Length'] = len(response.data)

            return response

        return f(*args, **kwargs)

    return view_func

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

def get_masks(ifname):
    img = Image.open(ifname)
    print("image.shape before resize:", img.size)
    img = img.resize((512,512))
    print("image.shape after resize:", img.size)

    img = np.array(img)

    # Run detection
    print("detecting objects...")
    start_time = time()
    with graph.as_default():
        results = model.detect([img], verbose=1)
    end_time = time()
    secs_elapsed = end_time - start_time
    print("detection took {:0.2f} seconds.".format(secs_elapsed))

    r = results[0]
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
    #                        class_names, r['scores'])
    NUM_DILATIONS = 10
    print("dilating masks {} time(s)...".format(NUM_DILATIONS))
    masks = r['masks']
    for i in range(r['scores'].shape[0]):
        masks[:, :, i] = binary_dilation(masks[:, :, i], iterations=NUM_DILATIONS).astype(masks.dtype)

    print("rois (shape={}): {}".format(r['rois'].shape, r['rois']))
    print("masks (shape={}): {}".format(r['masks'].shape, r['masks']))
    print("class_ids (shape={}): {}".format(r['class_ids'].shape, r['class_ids']))
    print("scores (shape={}): {}".format(r['scores'].shape, r['scores']))
    print("STATIC class_names (len={}): {}".format(len(class_names), class_names))
    return r['rois'], r['masks'], r['class_ids'], r['scores'], class_names

@app.route('/')
def upload_page():
    print("/upload_page called")
    return render_template('index4.html')
    
@app.route('/uploader', methods = ['POST'])
@gzipped
def upload_file():
    print("/uploader called")
    if request.method == 'POST':
        print("request.files:", request.files)
        if not 'file'  in request.files:
            return jsonify({"error":"no file supplied"})
        f = request.files['file']
        rand_uuid = uuid.uuid4().hex
        ofname = path.join(USER_IMG_DIR, rand_uuid + "_" + secure_filename(f.filename))
        print("writing to file '{}' ...".format(ofname))
        f.save(ofname)

        rois, masks, class_ids, scores, class_names = get_masks(ofname)
        masks = masks.astype(int)
        print("masks final shape (before tolist():", masks.shape)
        ret_dict = {
            'rois': rois.tolist(),
            'masks':masks.tolist(),
            'class_ids':class_ids.tolist(),
            'scores':scores.tolist(),
            'class_names': class_names
        }
        return jsonify(ret_dict)


if __name__ == '__main__':
    model = build_model()
    global graph
    graph = tf.get_default_graph() 
    # use_reloader on makes us load the model twice (this is slow and bad)
    app.run(debug = True, use_reloader=False, host="0.0.0.0", port=80) # port 80 means sudo only :/
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
