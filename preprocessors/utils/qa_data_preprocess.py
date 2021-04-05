# -*- coding: utf-8 -*-
# @Time    : 2020/7/10 
# @Author  : jing.gao01
# @Email   : gaojing850063636@sina.com
# @File    : qa_data_preprocess.py  
# @description :  处理QA数据 对句子进行分词处理并提取词汇表
# @Dev tool: Vs code/Pycharm

import sys
sys.path.append('../../')
from nlu.tokenizer import JiebaTokenizer

class QADataPreprocess(object):
    def __init__(self,data_path,stopword_path):
        self.data_path = data_path
        self.stopword_path = stopword_path
        #self.document_list = []
        self.q_text_list = []
        self.a_text_list = []
        self.read_file()
        self.read_stop_words()
        self.q_text_list_tokenizer()       

    def build_vocab(self,out_path):

        return "构建词汇表"

    def read_stop_words(self):
        stop_words = set()
        with open(self.stopword_path,'r',encoding='utf-8') as f:
            for line in f:
                words = line.strip('\r\n')
                if not words:
                    continue
                stop_words.add(words)
        self.stop_words = stop_words   
    def read_file(self):
        with open(self.data_path,'r',encoding = 'utf-8') as f:
            for line in f:
                q_text,a_text,_ = line.strip().split("\t")
                self.q_text_list.append(q_text)
                self.a_text_list.append(a_text)

    def q_text_list_tokenizer(self,stopwords=True):
        tokenizer = JiebaTokenizer()
        document_list = []
        for sent in self.q_text_list:
            sentence_token_list = []
            ##jieba分词
            sentence_token = tokenizer.tokenize(sent)
            for token in sentence_token:
                if stopwords and token[0] in self.stop_words:
                    continue
                sentence_token_list.append(token[0])
            document_list.append(sentence_token_list)
        return document_list

    def q_text_list_tokenizer_input(self,q_text,stopwords=True):
        tokenizer = JiebaTokenizer()
        document_list = []
        for sent in q_text:
            sentence_token_list = []
            ##jieba分词
            sentence_token = tokenizer.tokenize(sent)
            for token in sentence_token:
                if stopwords and token[0] in self.stop_words:
                    continue
                sentence_token_list.append(token[0])
            document_list.append(sentence_token_list)
        return document_list
