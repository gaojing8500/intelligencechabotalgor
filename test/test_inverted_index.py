# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 
# @Author  : jing.gao01
# @Email   : gaojing850063636@sina.com
# @File    : test_inverted_index.py
# @description :  测试倒排索引算法
# @Dev tool: Vs code/Pycharm
import sys
sys.path.append("../")
from nlu.tokenizer import Tokenizer
from preprocessors.utils.inverted_index import SimpleInvertedTableIndex
import jieba

class  TestInvertedIndex(object):
    def __init__(self,test_data_path,stop_words_path):
        self.test_path = test_data_path
        self.stop_words_path = stop_words_path
        self.document_list = []
        self.q_text_list = []
        self.a_text_list = []
        self.read_file()
        self.read_stop_words()
        self.q_text_list_tokenizer()
        #self.stop_words = set()
    def read_stop_words(self):
        stop_words = set()
        with open(self.stop_words_path,'r',encoding='utf-8') as f:
            for line in f:
                words = line.strip('\r\n')
                if not words:
                    continue
                stop_words.add(words)
        self.stop_words = stop_words        
    
    def read_file(self):
        with open(self.test_path,'r',encoding = 'utf-8') as f:
            for line in f:
                q_text,a_text = line.strip().split("\t")
                self.q_text_list.append(q_text)
                self.a_text_list.append(a_text)
    def q_text_list_tokenizer(self,stopwords=True):
        tokenizer = Tokenizer()
        for sent in self.q_text_list:
            sentence_token_list = []
            ##jieba分词
            sentence_token = tokenizer.tokenize(sent)
            for token in sentence_token:
                if stopwords and token[0] in self.stop_words:
                    continue
                sentence_token_list.append(token[0])
            self.document_list.append(sentence_token_list)

    def test_inverted_index_algor(self,query):
        inverted_model = SimpleInvertedTableIndex(self.document_list,self.a_text_list,self.q_text_list)
        query_list = list(jieba.cut(query))
        sim_q,sim_a = inverted_model.query_search_by_inverted_list(query_list)
        print(sim_q)
        #print(sim_a)



if __name__ == '__main__':
    test_data_path = "../../data/intelligence_chatbot/sentence_qa_service_total.txt"
    stop_words_path = "../../data/stopwords/stopword.txt"
    test_inveted_index = TestInvertedIndex(test_data_path,stop_words_path)
    while(1):
        user_query = input('请您输入: \n')
        if user_query == 'quit':
            break
        else:
            test_inveted_index.test_inverted_index_algor(user_query)
