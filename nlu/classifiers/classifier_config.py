# -*- coding: utf-8 -*-
# @Time    : 2020/6/15
# @Author  : jing.gao01
# @Email   : gaojing850063636@sina.com
# @File    : classifier_config.py
# @description :  意图识别分类器配置文件，模型路径 训练参数等
# @Dev tool: Vs code/Pycharm
import os

file_path = os.path.dirname(__file__)

svm_model = os.path.join(file_path,"../../../models/classifier_model/classifier_svm_model_04.m")
tfidf_vocabulary = os.path.join(file_path,"../../../models/classifier_model/tfidf_vocabulary_04.m")