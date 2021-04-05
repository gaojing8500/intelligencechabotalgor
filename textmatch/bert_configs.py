# -*- coding: utf-8 -*-
# @Time    : 2020/5/20 
# @Author  : jing.gao01
# @Email   : gaojing850063636@sina.com
# @File    : configs.py
# @description :  模型输出输入路径进行配置
# @Dev tool: Vs code/Pycharm

import os
import tensorflow as tf

##TF的日志系统
tf.logging.set_verbosity(tf.logging.INFO)
file_path = os.path.dirname(__file__)

pre_model_dir = os.path.join(file_path, '../../models/pre-trained/bert')
bert_name = os.path.join(pre_model_dir, 'bert_config.json')
ckpt_name = os.path.join(pre_model_dir, '../../fine-tune/bert/baseline/model.ckpt')
fine_tune_dir = os.path.join(pre_model_dir, '../../fine-tune/bert/baseline/') ##如果重新训练可这样版本命名
vocab_file = os.path.join(pre_model_dir, 'vocab.txt')

num_train_epochs = 5
batch_size = 128
learning_rate = 0.00005

# gpu使用率
gpu_memory_fraction = 0.8

# 默认取倒数第二层的输出值作为句向量
layer_indexes = [-2]

# 序列的最大程度，单文本建议把该值调小
max_seq_len = 50

# graph名字
graph_file = 'model/tmp/result/graph'
