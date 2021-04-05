# -*- coding: utf-8 -*-
# @Time    : 2020/7/10 
# @Author  : jing.gao01
# @Email   : gaojing850063636@sina.com
# @File    : main.py  
# @description :  main服务函数测试
# @Dev tool: Vs code/Pycharm
import sys
sys.path.append('../')
from preprocessors.utils.qa_data_preprocess import QADataPreprocess
from preprocessors.utils.inverted_index import SimpleInvertedTableIndex
from textmatch.tfidf import TF_IDF_Model
from service.config import Config
from textmatch.bert_similarity import BertSim
import tensorflow as tf
import jieba
from random import choice

from sanic import Sanic
from sanic.response import text,json,html

## restful api 
restful_app = Sanic()
class MainService(object):
    def __init__(self):
        self.name = "先进行tfidf初排在进行文本匹配算法精排.对于文本匹配算法后面会复现基本算法比如DSSM等"
        self.config = Config()
        self.data_process = QADataPreprocess(self.config.qa_data_path,self.config.stopword_path)
        self.init_tfidf()

        ##初始化Text_Match_BERT模型
        self.bert_text_macth_model = BertSim()
        self.bert_text_macth_model.set_mode(tf.estimator.ModeKeys.PREDICT)
        ##阈值设置
        self.max_threshold = 0.95
        self.min_threshold = 0.25


    def init_tfidf(self):
        ##倒排表暂时不加
        #sim_q,sim_a = self.inverted_index(query_list)
        candi_cut_sent = self.data_process.q_text_list_tokenizer(False)
        tfidf_model = TF_IDF_Model(candi_cut_sent)
        tfidf_model.gensim_tfidf_model()
        self.tfidf_model = tfidf_model

    def init_rank(self,query):
        ##粗排采用TDIDF和bm25算法，主要是对句子进行向量编码(词袋模型)也可以词向量累加句子向量在计算相似度fasttext+faiss
        
        index,score = self.tfidf_model.gensim_get_similarity_score(query,20)
        q_similarity = []
        score_list = []
        for index_,score_ in zip(index,score) :
            q_similarity.append(self.data_process.q_text_list[index_])
            score_list.append(score_)
        #print("gensim_tfidf_相似问题：{}".format(q_similarity))
        #print("gensim_tfidf_相似分数：{}".format(score_list))
        return q_similarity,score

    def inverted_index(self,query):
        cut_stopword_sent = self.data_process.q_text_list_tokenizer()
        q_text = self.data_process.q_text_list
        a_text = self.data_process.a_text_list
        inverted_model = SimpleInvertedTableIndex(cut_stopword_sent,a_text,q_text)
        sim_q,sim_a = inverted_model.query_search_by_inverted_list(query)
        return sim_q,sim_a

    def refuse_answer(self):
        refuse_corpus = ["对不起，您找的相关内容不在了","小吾吾还在学习中","小吾吾还在努力学习","您的内容好像不在,可以再次询问我"]
        return choice(refuse_corpus)

    def text_match(self,query):
        query_list = list(jieba.cut(query))
        ##tfidf粗排，BERT文本匹配精确排
        q_similarity,score = self.init_rank(query_list)
        if score[0] < self.max_threshold:
            ##精排
            tem_candatates = []
            for candatates_q in q_similarity:
                predict_score = self.bert_text_macth_model.predict(candatates_q,query)
                ##设置最低阈值
                if predict_score[0][1] > self.min_threshold:
                    tem_candatates.append(candatates_q)
            result_soreted = sorted(tem_candatates,reverse= False)
            if result_soreted:
                return result_soreted[0]
            return self.refuse_answer()
        else:
            return q_similarity[0]
##装饰器
@restful_app.route("/webhook/xiaowuwu",methods = ['POST','GET'])
async def chatbot_service(request):
    user_send = []
    user_send = request.json
    user_question = user_send['user_question']
    out_result = mainservice.text_match(user_question)
    return json({"result":str(out_result)})


if __name__ =="__main__":
    ##加载智能服务
    mainservice = MainService()
    restful_app.run(host="0.0.0.0",port=9760)
