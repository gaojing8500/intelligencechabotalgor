
# -*- coding: utf-8 -*-
# @Time    : 2020/6/11
# @Author  : jing.gao01
# @Email   : gaojing850063636@sina.com
# @File    : tokenizer.py
# @description :  检测是否有GPU资源
# @Dev tool: Vs code/Pycharm


from logging import getLogger
import tensorflow as tf
from tensorflow.python.client import device_lib

log = getLogger(__name__)

_gpu_available = None


def check_gpu_existence():
    r"""Return True if at least one GPU is available"""
    global _gpu_available
    if _gpu_available is None:
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        try:
            with tf.Session(config=sess_config):
                device_list = device_lib.list_local_devices()
                ##检测GPU资源
                _gpu_available = any(device.device_type == 'GPU' for device in device_list)
                print(_gpu_available)
        except AttributeError as e:
            log.warning(f'Got an AttributeError `{e}`, assuming documentation building')
            _gpu_available = False
    return _gpu_available


check_gpu_existence()