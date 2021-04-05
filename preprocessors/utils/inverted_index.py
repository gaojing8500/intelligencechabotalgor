# -*- coding: utf-8 -*-
# @Time    : 2020/6/12
# @Author  : jing.gao01
# @Email   : gaojing850063636@sina.com
# @File    : inverted_list.py
# @description :  简单倒排索引
# @Dev tool: Vs code/Pycharm



class InvertedTableIndex(object):
    def __init__(self,q_text_list):
        ##分词后去除停用词的列表
        self.q_text_list = q_text_list

    def build_dictionary(self):
        ##由于词典会非常庞大，查询该关键需要遍历这个词典，非常耗时间，通过hash加链表结构和树型结构
        return "构建单词词典"

    def inverted_list(self):
        return "构建倒排列表"

    def inverted_file(self):

        return "倒排文件"

class SimpleInvertedTableIndex(object):
    def __init__(self,q_text_list,a_text_list,q_origin_text):
        ##分词后去除停用词的句子列表
        self.q_text_list = q_text_list
        self.a_text_list = a_text_list
        self.q_origin_text = q_origin_text
        ##先构建倒排列表
        self.inverted_list()

    def inverted_list(self):
        inverted_list = {}
        for index,sentences in enumerate(self.q_text_list):
            for term in sentences:
                ##判断term 出现的文档的ID、词频和位置，暂时只记录所在文档的ID
                if term in inverted_list.keys():
                    inverted_list[term].append(index)
                else:
                    inverted_list[term] = [index]

        self.inverted_list = inverted_list

    def query_search_by_inverted_list(self,query):
        q_text = []
        a_text = []
        index_list = []
        for q_token in query:
            if q_token in self.inverted_list.keys():
                ##检索到相关文档
                index_list.extend(self.inverted_list[q_token])
        index_set = set(index_list)
        for index in index_set:
            q_text.append(self.q_origin_text[index])
            a_text.append(self.a_text_list[index])

        return q_text,a_text


    
