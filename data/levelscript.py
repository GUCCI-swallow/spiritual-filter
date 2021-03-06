#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2014 SIG2D
# Licensed under the Apache License, Version 2.0.
# changed by kivantium

import matplotlib
matplotlib.use('Agg')
import os
import shutil
import subprocess
import sys

import leveldb
import numpy as np
from PIL import Image
import random
from caffe.proto import caffe_pb2


THUMBNAIL_SIZE = 32


def make_thumbnail(image):
  image = image.convert('RGB')
  square_size = min(image.size)
  offset_x = (image.size[0] - square_size) / 2
  offset_y = (image.size[1] - square_size) / 2
  image = image.crop((offset_x, offset_y,
                      offset_x + square_size, offset_y + square_size))
  image.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.ANTIALIAS)
  return image


def make_datum(thumbnail, label):
  return caffe_pb2.Datum(
      channels=3,
      width=THUMBNAIL_SIZE,
      height=THUMBNAIL_SIZE,
      label=label,
      data=np.rollaxis(np.asarray(thumbnail), 2).tostring())


def create_leveldb(name):
  path = os.path.join(os.environ['HOME'], 'spiritual/caffe/examples/cifar10', name)
  try:
    shutil.rmtree(path)
  except OSError:
    pass
  print 'opening', path
  return leveldb.LevelDB(
      path, create_if_missing=True, error_if_exists=True, paranoid_checks=True)


def main():
  train_db = create_leveldb('cifar10_train_leveldb')
  test_db = create_leveldb('cifar10_test_leveldb')

  filepath_and_label = []
  for dirpath, _, filenames in os.walk('.'):
    try:
      label = int(dirpath.split('/')[1])
    except Exception:
      continue
    for filename in filenames:
      if filename.endswith(('.png', '.jpg')):
        filepath_and_label.append((os.path.join(dirpath, filename), label))

  random.shuffle(filepath_and_label)

  for seq, (filepath, label) in enumerate(filepath_and_label):
    print seq, label, filepath
    #読み込みエラーがあると止まるため改変
    #image = PIL.Image.open(filepath)
    try:
        image = Image.open(filepath)
    except:
        continue
    thumbnail = make_thumbnail(image)
    datum = make_datum(thumbnail, label)
    db = test_db if seq % 10 == 0 else train_db
    db.Put('%08d' % seq, datum.SerializeToString())


if __name__ == '__main__':
  sys.exit(main())
