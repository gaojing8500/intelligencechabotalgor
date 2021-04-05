# -*- coding: utf-8 -*-
# @Time    : 2020/5/20 
# @Author  : jing.gao01
# @Email   : gaojing850063636@sina.com
# @File    : text_match_algor.py  
# @description :  基于BERT的文本匹配算法baseline
# @Dev tool: Vs code/Pycharm
import sys
sys.path.append("../")
from sanic import Sanic
from sanic.response import text,json,html
from textmatch.bert_similarity import BertSim
import joblib
from nlu.classifiers.svm_classifiers import IntentClassifier
import tensorflow as tf

## restful api 
restful_app = Sanic()
class IntelligenceChabotService(object):
    def __init__(self):
        ##意图识别
        self.intent_class = IntentClassifier()
        ##初始化Text_Match_BERT模型
        self.bert_text_macth_model = BertSim()
        self.bert_text_macth_model.set_mode(tf.estimator.ModeKeys.PREDICT)
        ##阈值设置
        self.max_threshold = 0.25

    def intent_classifier(self,query,q_canditates):
        ##不同的意图采用不同机制
        pre_class = self.intent_class.intent_classifier(query)
        if str(pre_class[0]) == '1' and q_canditates != ' ':
            return self.task_intent(query,q_canditates)
        elif str(pre_class[0]) == '0' and q_canditates != ' ':
            return self.chat_intent(query,q_canditates)
        else:
            return ['小影还在努力学习中.....']
            
    def chat_intent(self,query,q_canditates):
        ##是不是可以采用其他策略来做
        q_canditates_ = []
        for single_q in q_canditates.strip().split(','):
            q_canditates_.append(str(single_q))
        index = -1
        chat_q_result = []
        for q_sentence in q_canditates_:
            predict_score = self.bert_text_macth_model.predict(q_sentence,query)
            index += 1
            chat_q_result_dict = {}
            chat_q_result_dict['index'] = index
            chat_q_result_dict['score'] = predict_score[0][1]
            chat_q_result.append(chat_q_result_dict)
        ##重排序
        chat_q_result_sorted = sorted(chat_q_result,key = lambda e:e.__getitem__('score'),reverse = True)
        temp_chat_q_result_sorted = []
        temp_chat_q_result_sorted.append(chat_q_result_sorted[0])
        return temp_chat_q_result_sorted


    def task_intent(self,query,q_canditates):
        ##任务型意图
        q_canditates_ = []
        for single_q in q_canditates.strip().split(','):
            q_canditates_.append(str(single_q))
        index = -1
        ##直接回答唯一问题
        only_q_result = []
        ##列表回答的相似问题
        similarity_q_result = []
        for q_sentence in q_canditates_:
            predict_score = self.bert_text_macth_model.predict(q_sentence,query)
            only_q_result_dict = {}
            index += 1
            similarity_q_result_dict = {}
            ##直接回答
            if predict_score[0][1] >= self.max_threshold:
                only_q_result_dict['index'] = index
                only_q_result_dict['score'] = predict_score[0][1]
                only_q_result.append(only_q_result_dict)
            ##列表回答
            else:
                similarity_q_result_dict['index'] = index
                similarity_q_result_dict['score'] = predict_score[0][1]
                similarity_q_result.append(similarity_q_result_dict)
            ##处理多个大于阈值的候选问题，取分数最高的候选问题
        if len(only_q_result) > 1:
            only_q_return = []
            list_sorted = sorted(only_q_result,key = lambda e:e.__getitem__('score'),reverse = True)
            only_q_return.append(list_sorted[0])
            return only_q_return
        if len(only_q_result) == 0:
            min_list_sorted = sorted(similarity_q_result,key = lambda e:e.__getitem__('score'),reverse = True)
            ##其中5个最高score
            return min_list_sorted[0:5]
        else:
            return only_q_result

##装饰器
@restful_app.route("/webhook/qa",methods = ['POST','GET'])
async def chatbot_service(request):
    user_send = []
    user_send = request.json
    q_canditates = user_send['q_canditates']
    user_question = user_send['user_question']
    out_result = intelligence_service.intent_classifier(user_question,q_canditates)
    return json({"max_similarity":str(out_result)})

if __name__ == '__main__':
    ##加载智能服务
    intelligence_service = IntelligenceChabotService()
    restful_app.run(host="0.0.0.0",port=9899)