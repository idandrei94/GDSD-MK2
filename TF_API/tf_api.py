from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from node_lookup import NodeLookup

def load_model(model_path):
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def infer_image(image_path):
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    load_model("inception/classify_image_graph_def.pb")
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        # Creates node ID --> English string lookup.
        node_lookup = NodeLookup()

        top_k = predictions.argsort()[-5:][::-1]
        for k in top_k:
            print(node_lookup.id_to_string(k))
        return node_lookup.id_to_string(top_k[0]) + ', ' + node_lookup.id_to_string(top_k[1])+ ', ' + node_lookup.id_to_string(top_k[2])