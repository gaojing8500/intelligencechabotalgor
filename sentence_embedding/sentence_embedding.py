# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 
# @Author  : jing.gao01
# @Email   : gaojing850063636@sina.com
# @File    : load_embedding.py
# @description :  加载词向量
# @Dev tool: Vs code/Pycharm
import sys
sys.path.append("../")
import numpy as np
import gensim
import test.config as config
from nlu.tokenizer import JiebaTokenizer
import tensorflow as tf


class SentenceEmbedding(object):
    def __init__(self,word_embedding_path,embedding_size,corpus_path):
        ##词向量路径
        self.word_embedding_path = word_embedding_path
        ##词向量维度
        self.embedding_size = embedding_size
        ##分词后语料
        self.corpus_path = corpus_path

    def build_vocabulary(self):
        return '构建词汇表'

    def load_word_embedding(self):
        return "加载词向量"

    def compute_embedding_size(self):
        return "计算词向量维度"

    def word_similarity_search(self):
        return "相似词搜索"

    def sentence_embedding(self):
        return "句子编码"    

class GensimSentenceEmbedding(object):
    def __init__(self):
        #self.word_embedding_path = word_embedding_path
        self.wv_model = gensim.models.KeyedVectors.load(config.extract_word_embedding, mmap='r')
        self.jieba_cut = JiebaTokenizer()

    def word_vector(self,word, wv_from_text, min_n=1, max_n=3):
        # 确认词向量维度
        word_size = wv_from_text.wv.syn0[0].shape[0]
        # 计算word的ngrams词组
        import pdb
        ngrams = self.compute_ngrams(word, min_n=min_n, max_n=max_n)
        print(ngrams)
        # 如果在词典之中，直接返回词向量
        found = 0
        if word in wv_from_text.index2word:
            found += 1
            return wv_from_text[word],found
        else:
            # 不在词典的情况下，计算与词相近的词向量
            word_vec = np.zeros(word_size, dtype=np.float32)
            ngrams_found = 0
            ngrams_single = [ng for ng in ngrams if len(ng) == 1] 
            ngrams_more = [ng for ng in ngrams if len(ng) > 1]  
            # 先只接受2个单词长度以上的词向量
            for ngram in ngrams_more:
                if ngram in wv_from_text.index2word:
                    word_vec += wv_from_text[ngram]
                    ngrams_found += 1
            # print(ngram)
            # 如果，没有匹配到，那么最后是考虑单个词向量
            if ngrams_found == 0:
                for ngram in ngrams_single:
                    if ngram in wv_from_text.index2word:
                        word_vec += wv_from_text[ngram]
                        ngrams_found += 1
            if word_vec.any():  # 只要有一个不为0
                return word_vec / max(1, ngrams_found),found
            else:
                print('all ngrams for word %s absent from model' % word)
                return 0,found
    def compute_ngrams(self,word, min_n, max_n):
        extended_word = word
        ngrams = []
        for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
            for i in range(0, len(extended_word) - ngram_length + 1):
                ngrams.append(extended_word[i:i + ngram_length])
        return list(set(ngrams)) 

    def sentence2vec(self,sentence):
        sentence_cut = self.jieba_cut.jieba_tokenize(sentence,'search')
        word_vector = []
        word_size = self.wv_model.wv.syn0[0].shape[0]
        compute_avarage =np.zeros(word_size, dtype=np.float32)
        for token in sentence_cut:
            vec,found = self.word_vector(token, self.wv_model, min_n=1, max_n=3)  # 词向量获取
            if vec is 0:
                continue
            ##只查询最
            #similar_word = self.wv_model.most_similar(positive=[vec], topn=1)
            compute_avarage += vec
        return compute_avarage

    def sentence_similarity(self,sentence1,sentence2):
        sentence1_vec = tf.constant(self.sentence2vec(sentence1))
        sentence2_vec = tf.constant(self.sentence2vec(sentence2))

        import pdb
        pdb.set_trace()
        norm_q = tf.sqrt(tf.reduce_sum(tf.multiply(sentence1_vec, sentence1_vec)))
        norm_a = tf.sqrt(tf.reduce_sum(tf.multiply(sentence2_vec, sentence2_vec)))
        mul_q_a = tf.reduce_sum(tf.multiply(sentence1_vec, sentence2_vec), 1)
        cos_sim_q_a = tf.div(mul_q_a, tf.multiply(norm_q, norm_a))
        print(cos_sim_q_a)


if __name__ == '__main__':
    q = '联影医疗是一种非常厉害的公司'
    a = '你好厉害哟'
    model = GensimSentenceEmbedding()
    model.sentence_similarity(q,a)

