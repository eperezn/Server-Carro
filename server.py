from flask import Flask,render_template,jsonify,request
import base64
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from utils import ops as utils_ops

from utils import label_map_util

from utils import visualization_utils as vis_util


app = Flask(__name__)
cont = 0
MODEL_NAME = 'inference_graph'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('inference_graph', 'labelmap.pbtxt')
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
@app.route('/')
def hello():
    return "Esta es la pagina principal del server, para poder interactuar se deben enviar images a la ruta /img"

@app.route('/img',methods=['GET','POST'])
def img():
    #En este momento esta con un ejemplo puesto en la ruta, pero la parte comentada debajo hace parte del envio de la imagen
    #Y luego la codificacion para envio al modelo y devolucion de resultados
    a  = run_inference_for_single_image('./imagenes/frame59.jpg', detection_graph)
    return jsonify(results=a)
    # global cont
    # data = request.get_json()
    # if cont > 4:
    #     with open("imagenes/image"+str(cont-4)+".jpg", "wb") as fh:
    #         fh.write(base64.b64decode(data["img"]))
    # cont+=1
    # return jsonify(data="OK") 

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(images, graph):
    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            image = Image.open(images)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            objects = []
            for index, value in enumerate(classes[0]):
                object_dict = {}
                if scores[0, index] > 0.5:
                    objects.append(str((category_index.get(value)).get('name')))
            return objects
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)