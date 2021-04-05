# -*- coding: utf-8 -*-
# @Time    : 2020/6/18
# @Author  : jing.gao01
# @Email   : gaojing850063636@sina.com
# @File    : config.py
# @description :  测试样例模型和参数配置文件
# @Dev tool: Vs code/Pycharm


import os

file_path = os.path.dirname(__file__)

extract_word_embedding = os.path.join(file_path,"../../models/word_embedding/tencent_word_embedding/400million_chinese_embedding.bin")
tencent_word_embedding = os.path.join(file_path,"../../models/word_embedding/tencent_word_embedding/Tencent_AILab_ChineseEmbedding.txt")