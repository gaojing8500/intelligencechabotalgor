
# -*- coding: utf-8 -*-
# @Time    : 2020/6/8 
# @Author  : jing.gao01
# @Email   : gaojing850063636@sina.com
# @File    : test_textmatch_models.py  
# @description :  对文本匹配模型进行评测
# @Dev tool: Vs code/Pycharm

import sys
sys.path.append("../")
import preprocessors.utils.metrics as metrics
import textmatch.bert_similarity as bert_similairty
import tensorflow as tf


class TestTextmatchModels(object):
    def __init__(self,test_data_path):
        self.data_path = test_data_path

    def test_textmatch_models_f1score(self):
        sim = bert_similairty.BertSim()
        sim.set_mode(tf.estimator.ModeKeys.PREDICT)
        label_0 = '0'
        label_1 = '1'
        true_label_list = []
        predict_label_list = []
        import pdb
        pdb.set_trace()
        with open(self.data_path,'r',encoding='utf-8') as f:
            for line in f:
                q_text,a_text,label = line.strip().split('\t')
                true_label_list.append(int(label))
                predict_score = sim.predict(q_text,a_text)
                if predict_score[0][0] > predict_score[0][1]:
                    predict_label_list.append(int(label_0))
                else:
                    predict_label_list.append(int(label_1))
        return true_label_list,predict_label_list

    def computer_model_f1score_acc_recall_precision(self):
        true_label_list,predict_label_list = self.test_textmatch_models_f1score()
        acc, recall, precision, f_beta = metrics.get_binary_metrics(predict_label_list,true_label_list)
        print('acc:{}'.format(acc))
        print('recall:{}'.format(recall))
        print('precision:{}'.format(precision))
        print('f_beta:{}'.format(f_beta))

if __name__ == '__main__':
    lucqmc_test_path = "../../data/lucqmc/test.txt"
    '''
    acc:0.85536
    recall:0.95328
    precision:0.7971635001337971
    f_beta:0.868259982512387
    '''
    intelligence_chatbot = "../../data/intelligence_chatbot/sentence_qq_service_total.txt"
    '''
    acc:0.8303425774877651
    recall:0.5192307692307693 召回率较低
    precision:0.9642857142857143
    f_beta:0.6749999999999999
    '''

    test_model = TestTextmatchModels(intelligence_chatbot)
    test_model.computer_model_f1score_acc_recall_precision()




