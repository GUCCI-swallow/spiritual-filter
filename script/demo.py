import matplotlib
matplotlib.use('Agg')

import caffe
from caffe.proto import caffe_pb2
import numpy as np

head_path='/home/ubuntu/spiritual/caffe/examples/cifar10/'

mean_blob = caffe_pb2.BlobProto()
with open(head_path +'mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(
mean_blob.data,
dtype=np.float32).reshape(
    (mean_blob.channels,
    mean_blob.height,
    mean_blob.width))
classifier = caffe.Classifier(
    head_path +'cifar10_quick.prototxt',
    head_path +'cifar10_quick_iter_4000.caffemodel',
    mean=mean_array,
    raw_scale=255)

image = caffe.io.load_image('nozomi.jpg')
predictions = classifier.predict([image], oversample=False)
pred = np.argmax(predictions)
print(predictions)
print(pred)
