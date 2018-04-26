# coding: utf-8

import os
import math
import six
import requests
from tqdm import tqdm
import numpy as np
import tensorflow as tf

DEFAULT_BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap'

def download(raw_data_dir, label, base_url):
    res = requests.get('{}/{}.npy'.format(base_url, label), stream=True)
    total_size = int(res.headers.get('content-length', 0)) 
    block_size = 1024
    wrote = 0
    file = os.path.join(raw_data_dir, '{}.npy'.format(label))
    if os.path.exists(file):
        print('Found "{}". Skip to download.'.format(file))
        return
    with open(file, 'wb') as f:
        for data in tqdm(
                res.iter_content(block_size),
                desc=file,
                total=math.ceil(total_size//block_size),
                unit='KB',
                unit_scale=True):
            wrote = wrote  + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")  

def downloads(raw_data_dir, labels, base_url):
    for label in labels:
        download(raw_data_dir, label, base_url)

def encode_tfrecord(data_dir, raw_data_dir, labels, train_file, test_file):
    raw_data = {label: np.load(os.path.join(raw_data_dir,'{}.npy'.format(label))) for label in labels}
    for k, v in six.iteritems(raw_data):
        print('{:10}: {}'.format(k,len(v)))
    train_data = []
    test_data = []
    for label_name, value in six.iteritems(raw_data):
        label_index = labels.index(label_name)
        print('proccessing label {}: "{}"'.format(label_index, label_name))
        value = np.asarray(value) / 255.
        tr = value[:70000]
        te = value[70000:100000]
        train_data.extend(zip(tr, np.full(70000, label_index)))
        test_data.extend(zip(te, np.full(30000, label_index)))
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    train_file = os.path.join(data_dir, train_file)
    test_file = os.path.join(data_dir, test_file)
    tfr_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    with tf.python_io.TFRecordWriter(train_file, tfr_options) as train_tfr, \
         tf.python_io.TFRecordWriter(test_file, tfr_options) as test_tfr:
        print('writing train tfrecord files...')
        for data, label in tqdm(train_data, desc=train_file):
            train_tfr.write(get_example_proto(data, [label]))
        for data, label in tqdm(test_data, desc=test_file):
            test_tfr.write(get_example_proto(data, [label]))

def get_example_proto(image, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'image' : tf.train.Feature(float_list=tf.train.FloatList(value=image)),
        'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
    })).SerializeToString()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--raw-data-dir', default='./raw_data')
    parser.add_argument('-d', '--data-dir', default='./data')
    parser.add_argument('-l','--labels', nargs='+', default=[])
    parser.add_argument('-b','--base-url', default=DEFAULT_BASE_URL)
    parser.add_argument('-t','--train-file', default='train.tfr')
    parser.add_argument('-e','--test-file', default='test.tfr')
    args = parser.parse_args()
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    if not os.path.exists(args.raw_data_dir):
        os.makedirs(args.raw_data_dir)
    
    if len(args.labels) == 0:
        print('Label not specified. terminate.')
    else:
        downloads(args.raw_data_dir, args.labels, args.base_url)
        encode_tfrecord(args.data_dir, args.raw_data_dir,
                        args.labels, args.train_file, args.test_file)
