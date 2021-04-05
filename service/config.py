# -*- coding: utf-8 -*-
# @Time    : 2020/7/10 
# @Author  : jing.gao01
# @Email   : gaojing850063636@sina.com
# @File    : config.py  
# @description :  参数设置，模型配置文件
# @Dev tool: Vs code/Pycharm

import os

file_path = os.path.dirname(__file__)

class Config(object):
    def __init__(self):
        self.name = "参数配置 模型路径配置"
        self.qa_data_path = os.path.join(file_path,'../../data/lcqmc/test.txt')
        self.stopword_path = os.path.join(file_path,'../../data/stopwords/baidu_stopwords.txt')
        
