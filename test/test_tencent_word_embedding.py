# -*- coding: utf-8 -*-
# @Time    : 2020/6/18 
# @Author  : jing.gao01
# @Email   : gaojing850063636@sina.com
# @File    : test_tencent_word_embedding.py
# @description :  测试腾讯词向量质量，采用skip_gram和负采样算法训练 800万个词向量
# @Dev tool: Vs code/Pycharm

import gensim
import numpy as np
import time
import datetime
import config 
class TestTencetWordEmbedding(object):
    def __init__(self):
        ##提取多少个词向量
        self.word_number = 5000000

    def extract_word_embedding(self):
        wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(config.tencent_word_embedding,limit = self.word_number,
                                                               binary = False)
        wv_from_text.init_sims(replace = True)
        wv_from_text.save_word2vec_format(config.extract_word_embedding)  ##保存二级制模型
        print("已经完成模型的抽取")

    def compute_ngrams(self,word, min_n, max_n):
        extended_word = word
        ngrams = []
        for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
            for i in range(0, len(extended_word) - ngram_length + 1):
                ngrams.append(extended_word[i:i + ngram_length])
        return list(set(ngrams))
    
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
    
    def test_similarity_word_search(self,input_text):
        print("开始载入文件...")
        print("Now：", datetime.datetime.now())
        t1 = time.time()
        wv_from_text = gensim.models.KeyedVectors.load(config.extract_word_embedding, mmap='r')
        print("文件载入完毕")
        print("文件载入耗费时间：", (time.time() - t1) / 60, "minutes")
        print('词表长度',len(wv_from_text.vocab))
        print("获取关键词列表")
        result_list = input_text.split("，")
        words_length = len(result_list)
        print(result_list)
        import pdb
        pdb.set_trace()
        for keyword in result_list:
            vec,found = self.word_vector(keyword, wv_from_text, min_n=1, max_n=3)  # 词向量获取
            if vec is 0:
                continue
            similar_word = wv_from_text.most_similar(positive=[vec], topn=15)  # 相似词查找
            result_word = [x[0] for x in similar_word]
            print(result_word)
            print("词库覆盖比例：", found, "/", words_length)
            print("词库覆盖百分比：", 100 * found / words_length, "%")
            print("整个推荐过程耗费时间：", (time.time() - t1) / 60, "minutes")    


if __name__ == '__main__':
    input_text = "感冒，冠脉造影，肾虚，金银花颗粒，板蓝根，交片打印，肺癌，肺炎，冠状病毒，如何进行交片打印操作，调窗，窗宽窗位"
    word_embedding = TestTencetWordEmbedding()
    word_embedding.test_similarity_word_search(input_text)
    #word_embedding.extract_word_embedding()