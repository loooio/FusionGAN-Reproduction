# -*- coding: utf-8 -*-
from model import CGAN
from utils import input_setup

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto

import pprint
import os

# 使用TF2兼容的方式定义flags
flags = tf.compat.v1.flags
flags.DEFINE_integer("epoch", 10, "Number of epoch [10]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 132, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 120, "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_string("summary_dir", "log", "Name of log directory [log]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()


def main(_):
    pp.pprint(FLAGS.flag_values_dict())  # 更新flags打印方式

    # 使用更安全的目录创建方式
    os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)
    os.makedirs(FLAGS.sample_dir, exist_ok=True)

    # 配置TF2兼容的Session
    config = ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        tf.compat.v1.disable_eager_execution()

        srcnn = CGAN(sess,
                     image_size=FLAGS.image_size,
                     label_size=FLAGS.label_size,
                     batch_size=FLAGS.batch_size,
                     c_dim=FLAGS.c_dim,
                     checkpoint_dir=FLAGS.checkpoint_dir,
                     sample_dir=FLAGS.sample_dir)

        srcnn.train(FLAGS)

if __name__ == '__main__':
    tf.compat.v1.app.run()