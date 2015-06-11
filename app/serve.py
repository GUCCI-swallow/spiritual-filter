#!/usr/bin/python
#-*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

import caffe
from caffe.proto import caffe_pb2
import numpy as np


from flask import Flask,render_template,request,redirect,url_for

app = Flask(__name__)


head_path='/home/ubuntu/spiritual/caffe/examples/cifar10/'

def spiritual_filter(check_data):
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

	image = caffe.io.load_image(check_data)
	predictions = classifier.predict([image], oversample=False)
	pred = np.argmax(predictions)
	return pred

@app.route('/')
def index():
	message = u"うちの画像をあげてみ"
	return render_template('index.html',message=message)


@app.route('/api',methods=['GET','POST'])
def api():
	message = u"なんにもないで"
	if request.method == 'POST':
		img = request.files['check_img']
		if spiritual_filter(img) == 0:			
			message = u"ご褒美にワシワシやで"
		else:
			message = u"ねぇ、この子はだれなん？"

	return render_template('index.html',message=message)
	

if __name__=='__main__':
	app.debug = True
	app.run(host='0.0.0.0')
