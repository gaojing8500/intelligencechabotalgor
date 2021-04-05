
# -*- coding: utf-8 -*-
# @Time    : 2020/6/12 
# @Author  : jing.gao01
# @Email   : gaojing850063636@sina.com
# @File    : preprocess_intelligence_chatbot_data.py
# @description :  预处理智能问答对数据
# @Dev tool: Vs code/Pycharm


import yaml
import os
#import openpyxl
import pandas as pd
import numpy as np
##排列组合库 非常强大
from itertools import combinations
from itertools import product
from random import choice
import jieba

##先要对文件进行解密才行

def read_file(path_ct,path_u,path_mi,path_mr,qa_out_path,qq_out_path):

    ##读取U课堂数据
    import pdb
    result_u = pd.read_excel(path_u,sheet_name="Sheet1")
    result_mi = pd.read_excel(path_mi,sheet_name="Sheet1")
    result_mr = pd.read_excel(path_mr,sheet_name="Sheet1")
    result_ct = pd.read_excel(path_ct,sheet_name="工作表1")
    get_label_u = result_u['规则分类']
    label_list_u = set(get_label_u.values.tolist())
    ##获取Question-Answer数据格式数据
    def get_list_qa(result_dataframe):
        import pdb
        #pdb.set_trace()
        save_list=[]
        get_label = result_dataframe['规则分类']
        label_list = set(get_label.values.tolist())
        print(label_list)
        for label in label_list:
            a = result_dataframe.loc[result_dataframe['规则分类']==label]
            quick_operation_data = np.array(a)
            quick_operation_data_list = quick_operation_data.tolist()
            for q in quick_operation_data_list:
                if quick_operation_data_list[0][0] == label:
                    label_,standard_q,similarity_q_1,similarity_q_2,similarity_q_3,similarity_q_4,similarity_q_5,answ = q
                    ##文件中有些数据没有答案
                    if str(answ) == 'nan':
                        continue
                    answ = answ.replace('\n','')
                    joint_str_qa_1 = standard_q + '\t' + answ
                    save_list.append(joint_str_qa_1)
                    if str(similarity_q_1) != 'nan':
                        joint_str_qa_2 = similarity_q_1 + '\t' + answ
                        save_list.append(joint_str_qa_2)
                    if str(similarity_q_2) != 'nan':
                        joint_str_qa_3 = similarity_q_2 + '\t' + answ
                        save_list.append(joint_str_qa_3)
                    if str(similarity_q_3) != 'nan':
                        joint_str_qa_4 = similarity_q_3 + '\t' + answ
                        save_list.append(joint_str_qa_4)
                    if str(similarity_q_4) != 'nan':
                        joint_str_qa_5 = similarity_q_4 + '\t' + answ
                        save_list.append(joint_str_qa_5)
                    if str(similarity_q_5) != 'nan':
                        joint_str_qa_6 = similarity_q_5 + '\t' + answ
                        save_list.append(joint_str_qa_6)
                elif quick_operation_data_list[0][1] == label:
                    _,label_,standard_q,similarity_q_1,similarity_q_2,similarity_q_3,similarity_q_4,similarity_q_5,answ = q
                    ##文件中有些数据没有答案
                    if str(answ) == 'nan':
                        continue
                    answ = answ.replace('\n','')
                    joint_str_qa_1 = standard_q + '\t' + answ
                    save_list.append(joint_str_qa_1)
                    if str(similarity_q_1) != 'nan':
                        joint_str_qa_2 = similarity_q_1 + '\t' + answ
                        save_list.append(joint_str_qa_2)
                    if str(similarity_q_2) != 'nan':
                        joint_str_qa_3 = similarity_q_2 + '\t' + answ
                        save_list.append(joint_str_qa_3)
                    if str(similarity_q_3) != 'nan':
                        joint_str_qa_4 = similarity_q_3 + '\t' + answ
                        save_list.append(joint_str_qa_4)
                    if str(similarity_q_4) != 'nan':
                        joint_str_qa_5 = similarity_q_4 + '\t' + answ
                        save_list.append(joint_str_qa_5)
                    if str(similarity_q_5) != 'nan':
                        joint_str_qa_6 = similarity_q_5 + '\t' + answ
                        save_list.append(joint_str_qa_6)
                    

        return save_list
    ##Question-Qustion label 数据格式
    def get_list_qq(result_dataframe):
        save_list=[]
        question_group = []
        q_q_combin_group = []
        get_label = result_dataframe['规则分类']
        label_list = set(get_label.values.tolist())
        for label in label_list:

            a = result_dataframe.loc[result_dataframe['规则分类']==label]
            quick_operation_data = np.array(a)
            quick_operation_data_list = quick_operation_data.tolist()
            for q in quick_operation_data_list:
                if quick_operation_data_list[0][0] == label:
                    #label_,standard_q,similarity_q_1,similarity_q_2,similarity_q_3,similarity_q_4,similarity_q_5,answ = q
                    question_text = q
                    question_text = question_text[1:len(question_text)-1]
                    ##python list推导式写法
                    question_text = [i for i in question_text if str(i) != 'nan']
                    ##相似问题组合
                    q_q_combin = list(combinations(question_text,2))
                    q_q_combin_group += q_q_combin
                    ##相似问题集合
                    question_group +=question_text
        label_0 = '0'
        label_1 = '1'
        q_q_label_list = []
        import pdb
        #pdb.set_trace()
        for q_q in q_q_combin_group:
            ##正样本
            q_q_label_1 = q_q[0] +'\t'+ q_q[1] + '\t'+label_1 
            q_q_label_list.append(q_q_label_1)
            random_question = choice(question_group)
            if random_question != q_q[0] and random_question != q_q[1]:
                ##负样本
                q_q_label_0 = q_q[0] + '\t' + random_question + '\t'+ label_0
                q_q_label_list.append(q_q_label_0)
                q_q_label_0 = q_q[1] + '\t' + random_question +'\t' + label_0
                q_q_label_list.append(q_q_label_0)
        return q_q_label_list


    pdb.set_trace()
    save_list_total = []
    q_q_label_list_total = []   
    #save_list_u = get_list_qa(result_u)
    #save_list_ct = get_list_qa(result_ct)
    #save_list_mi = get_list_qa(result_mi)
    #print(len(save_list_mi))
    #save_list_mr = get_list_qa(result_mr) 
    ##合并list，写入txt
    #save_list_total = save_list_u + save_list_ct + save_list_mi + save_list_mr
    ##获取QQ数据格式
    q_q_label_list_u = get_list_qq(result_u)
    q_q_label_list_ct = get_list_qq(result_ct)
    q_q_label_list_mi = get_list_qq(result_mi)
    q_q_label_list_mr = get_list_qq(result_mr)
    q_q_label_list_total = q_q_label_list_u + q_q_label_list_ct + q_q_label_list_mi + q_q_label_list_mr
    ##q-a数据写入
    #with open(qa_out_path,'w',encoding = 'utf-8') as f:
    #    for line in save_list_total:
    #        f.writelines(line)
    #        f.write('\n')
    ##q-q数据写入
    with open(qq_out_path,'w',encoding = 'utf-8') as f:
        for line in q_q_label_list_total:
            f.writelines(line)
            f.write('\n')
        

def read_csv(path):
    import pdb; pdb.set_trace()
    qList = []
    # 问题的关键词列表
    qList_kw = []
    aList = []
    data = pd.read_csv(path)
    data_ = np.array(data).tolist()
    for t in data_:
        qList.append(t[0])
        qList_kw.append(jieba.cut(t[0]))
        aList.append(t[1])
    return qList_kw, qList, aList

if __name__ == '__main__':

    file_path = "E:/gaojing/知识图谱/智能问答项目/example/Customer-Chatbot-master/xiaotian-chatbot1.0/data/corpus1/xiaoyindata/"
    data_path_ct =os.path.join(file_path,"智能客服知识库_0325.xlsx")
    data_path_u =os.path.join(file_path, "U课堂FAQ.xlsx")
    data_path_mi = os.path.join(file_path,'客户日常操作问题_MI.xlsx')
    data_path_mr = os.path.join(file_path,'客户日常操作问题汇总_MR.xlsx')

    qa_out_path = "sentence_qa_service_total.txt"
    qq_out_path = "sentence_qq_service_total.txt"
    read_file(data_path_ct,data_path_u,data_path_mi,data_path_mr,qa_out_path,qq_out_path)