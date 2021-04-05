# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 
# @Author  : jing.gao01
# @Email   : gaojing850063636@sina.com
# @File    : preprocess_classifiers_data.py
# @description :  预处理意图识别的基础数据（分类数据）
# @Dev tool: Vs code/Pycharm
import  json
import random

class PreClassifierData(object):
    def __init__(self,qa_pre_data_path,greet_data_path):
        self.qa_data_path = qa_pre_data_path
        self.greet_data_path = greet_data_path
        self.q_text_label_list = []
        self.greet_text_label_list = []
    def read_file(self,baike_q):
        label_1 = '1'
        label_0 = '0'
        with open(self.qa_data_path,'r',encoding = 'utf-8') as f:
            for line in f:
                q_text,_ = line.strip().split('\t')
                q_text = q_text + '\t' + label_1
                self.q_text_label_list.append(q_text)
        with open(self.greet_data_path,'r',encoding = 'utf-8') as f:
            for line in f:
                greet_q_text,_ = line.strip().split()
                greet_q_text = greet_q_text + '\t' + label_0
                self.greet_text_label_list.append(greet_q_text)
        for baike_q_text in baike_q:
            temp_baike_q_text = baike_q_text + '\t' + label_0
            self.greet_text_label_list.append(temp_baike_q_text)
    def write_file(self,out_path):
        print(len(self.q_text_label_list))
        print(len(self.greet_text_label_list))
        total_label_classifier_list = self.q_text_label_list + self.greet_text_label_list
        ##随机打散
        random.shuffle(total_label_classifier_list)
        with open(out_path,'w',encoding = 'utf-8') as f:
            for line in total_label_classifier_list:
                f.writelines(line)
                f.write('\n')

class PreprocessBaikeData(object):
    def __init__(self,baike_data_path):
        self.baike_data_path = baike_data_path
        self.baike_q = []

    def read_json_file(self):
        with open(self.baike_data_path,'r') as f:
            for text in f.readlines():
                json_text = json.loads(text)
                if json_text["category"] == "生活-生活常识":
                    self.baike_q.append(json_text["title"])
##抽取训练集和测试集
def pandas_read_txt_file(classfier_data):
    import pandas as pd
    df_text = pd.read_table(classfier_data)
    extrac_train = df_text.sample(frac = 0.8,replace = False)
    extrac_test = df_text.sample(frac = 0.2,replace = False)
    extrac_train.to_csv("../../data/intelligence_chatbot/classifier_train.csv")
    extrac_test.to_csv("../../data/intelligence_chatbot/classifier_test.csv")

##统计正负样本比例
def statistics_sample_label():
    import pandas as pd
    import numpy as np
    train_data_path = "../../data/intelligence_chatbot/classifier_train.csv"
    test_data_path = "../../data/intelligence_chatbot/classifier_test.csv"
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)
    train_count_1 = 0
    train_count_0 = 0
    test_count_1 = 0
    test_count_0 = 0
    import pdb
    pdb.set_trace()
    train_list = np.array(df_train).tolist()
    test_list = np.array(df_test).tolist()
    for train_label in train_list:
        label = train_label[2]
        if str(label) == '1':
            train_count_1 += 1
        else:
            train_count_0 += 1
    for test_label in test_list:
         label = test_label[2]
         if str(label) == '1':
            test_count_1 += 1
         else:
            test_count_0 += 1

    print("训练集正样本为：",train_count_1)
    print("训练集负样本为：",train_count_0)
    print("测试集正样本为：",test_count_1)
    print("测试集负样本为：",test_count_0)


if __name__ == '__main__':
    '''
    baike_data = "../../data/baike_qa_2019/baike_qa_valid.json"
    test_data_path = "../../data/intelligence_chatbot/sentence_qa_service_total.txt"
    greet_data_path = "../../data/intelligence_chatbot/greet_chat.txt"
    out_classifier_data = "../../data/intelligence_chatbot/preproceess_classfier_data.txt"
    pre_data = PreprocessBaikeData(baike_data)
    pre_data.read_json_file()
    preclassfie = PreClassifierData(test_data_path,greet_data_path)
    preclassfie.read_file(pre_data.baike_q)
    preclassfie.write_file(out_classifier_data)
    '''
   # pandas_read_txt_file("../../data/intelligence_chatbot/preproceess_classfier_data.txt")
    statistics_sample_label()
            
