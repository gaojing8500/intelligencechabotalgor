# -*- coding: utf-8 -*-
# @Time    : 2020/6/16
# @Author  : jing.gao01
# @Email   : gaojing850063636@sina.com
# @File    : svm_classifiers.py
# @description :  支持向量机意图分类任务
# @Dev tool: Vs code/Pycharm
import sys
sys.path.append('../../')
import os
import pandas as pd
import numpy as np
from nlu.tokenizer import JiebaTokenizer 
##词袋模型bag-of-words 三种词向量方式
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
from sklearn import svm
from sklearn import metrics
import nlu.classifiers.classifier_config as config

##模型保存的两种方式 pickle和joblib
import joblib

class  SVMClassifier(object):
    def __init__(self,train_data,test_data):
        self.train_data = train_data
        self.test_data = test_data
    def read_file(self,data_path):
        df_data = pd.read_csv(data_path)
        data_list = np.array(df_data).tolist()
        label_list = []
        q_text_list = []
        for text in data_list:
            label_list.append(text[2])
            q_text_list.append(text[1])
        return q_text_list,label_list
    def tfidf2vector(self):
        ##文本表示方法 hashing2vector count2vector tfidf2vector word_embedding(word2vec glove fasttext) 
        #语言模型(BERT ALBERT XLNET Roberta ELMO等)
        train_data_q,train_data_label = self.read_file(self.train_data)
        test_data_q,test_data_label = self.read_file(self.test_data)
        token_train_data_q_list = []
        test_train_data_q_list = []
        tokenizer = JiebaTokenizer()
       
        for train_token in train_data_q:
            ##分词
            token_list = tokenizer.tokenize(train_token)
            temp_string = ""
            for token in token_list:
                temp_string = temp_string + " " + token[0]
            token_train_data_q_list.append(temp_string)
        
        for test_token in test_data_q:
            ##分词
            token_list = tokenizer.tokenize(test_token)
            temp_string = ""
            for token in token_list:
                temp_string = temp_string + " " + token[0]
            test_train_data_q_list.append(temp_string)
        v1 = TfidfVectorizer()
        ##测试集共享训练集的词汇表，保持维度一直还有一种做法是CountVector + TDIDFvectorizer达到训练集和测试集维度一致
        ##相关细节https://blog.csdn.net/abcjennifer/article/details/23615947,https://blog.csdn.net/liujiandu101/article/details/51736436
        train_vector = v1.fit_transform(token_train_data_q_list)
        ##保存词汇表
        if not os.path.exists(config.tfidf_vocabulary):
                joblib.dump(v1.vocabulary_,config.tfidf_vocabulary)
        share_vocabulary = joblib.load(config.tfidf_vocabulary)
        v2 = TfidfVectorizer(vocabulary = share_vocabulary)
        test_vector = v2.fit_transform(test_train_data_q_list) 
        ##计算训练集和测试稀疏性问题
        #average_sparsity = train_vector.nnz/(float(train_vector.shape[0]*train_vector.shape[1])*100)
        #print(average_sparsity)
        return train_vector,train_data_label,test_vector,test_data_label
    def train_svm_model(self):
        ##构建SVM模型 
        train_vector,train_data_label,test_vector,test_data_label = self.tfidf2vector()
        svm_model = svm.SVC(C=700.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3,
                  gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False) 
        svm_model.fit(train_vector,np.asanyarray(train_data_label))
        ##保存svm模型
        import pdb
        pdb.set_trace()
        if not os.path.exists(config.svm_model):
            joblib.dump(svm_model,config.svm_model)
        self.svm_model = svm_model
        self.test_data_label = test_data_label
        self.test_vector = test_vector

    def evalute(self,actual,pred):
        m_precision =  metrics.precision_score(actual,pred,average = 'macro')
        m_recal = metrics.recall_score(actual,pred,average = 'macro')
        '''
        precision:0.9971098265895953
        recall:0.9962962962962962
        '''
        print('precision:{}'.format(m_precision))
        print('recall:{}'.format(m_recal))

    def test_predict(self):
        pre_score = self.svm_model.predict(self.test_vector)
        self.evalute(np.asarray(self.test_data_label),pre_score)

    def test_sklearn_algor(self):
        document = ["我 喜欢 NLP 这个 课程",
                    "联影医疗 是 一家 非常 牛叉 的 公司"]
        target_label = [0,1]
        v1 = TfidfVectorizer()
        tfifd_vectore = v1.fit_transform(document)
        print(v1.vocabulary_)
        print(tfifd_vectore)
        svm_model = svm.SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3,
                  gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False) 
        svm_model.fit(tfifd_vectore,np.asanyarray(target_label))

class IntentClassifier(object):
    def __init__(self):
        self.classifier_model = config.svm_model
    def intent_classifier(self,query):
        import jieba
        token_join = ''
        token_join_list = []
        for token in list(jieba.cut(query)):
            token_join = token_join + ' ' + token
        token_join_list.append(token_join)
        ##加载词汇表
        tfidf_vocabulary = joblib.load(config.tfidf_vocabulary)
        v1 = TfidfVectorizer(vocabulary=tfidf_vocabulary)
        user_query_vectore =v1.fit_transform(token_join_list) 
        ##加载svm模型
        svm_model = joblib.load(self.classifier_model)
        pre_class = svm_model.predict(user_query_vectore)
        print(pre_class)
        return pre_class

if __name__ == '__main__':
    train_data_path = "../../../data/intelligence_chatbot/classifier_train.csv"
    test_data_path = "../../../data/intelligence_chatbot/classifier_test.csv"
    #outpath = "../../../models/classifier_model"
    #load_model_path = os.path.join(outpath,'classifier_svm_model_01.m')
    svm_model = SVMClassifier(train_data_path,test_data_path)
    #svm_model.train_svm_model()
    #svm_model.test_sklearn_algor()
    #svm_model.test_predict()
    intent_class = IntentClassifier()

    while(1):
        query = input("请您输入: \n")
        if query == 'quit':
            break
        else:
            intent_class.intent_classifier(query)




        

            
            