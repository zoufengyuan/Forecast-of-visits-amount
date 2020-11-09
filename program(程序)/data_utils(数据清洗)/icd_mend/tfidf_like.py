# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:59:58 2020

@author: 86156
"""

import jieba
import re
import numpy as np
from gensim import corpora,models,similarities


class Text_similarity(object):
    def __init__(self): 
        self.all_doc = []
        with open('AID_标准名.txt','r') as f:
            for line in f.readlines():
                doc = line.split('\t')[1].strip()
                self.all_doc.append(doc)
        self.all_doc_list = []
        for doc in self.all_doc:
            doc_list = [word for word in jieba.cut(doc)]
            self.all_doc_list.append(doc_list)
        self.model()
    def model(self):
        self.dictionary = corpora.Dictionary(self.all_doc_list)
        corpus = [self.dictionary.doc2bow(doc) for doc in self.all_doc_list]
        self.tfidf = models.TfidfModel(corpus)
        self.index = similarities.SparseMatrixSimilarity(self.tfidf[corpus], num_features=len(self.dictionary.keys()))

    def jieba_cut(self,doc_test):
        doc_test_list = [word for word in jieba.cut(doc_test)]
        doc_test_vec = self.dictionary.doc2bow(doc_test_list)
        sim = self.index[self.tfidf[doc_test_vec]]
        tmp_result = sorted(enumerate(sim), key=lambda item: -item[1])
        result = self.all_doc[tmp_result[0][0]]
        val = tmp_result[0][1]
        return result,val

class Text_similarity_2(object):
    def __init__(self): 
        self.all_doc = []
        with open('AID_标准名.txt','r') as f:
            for line in f.readlines():
                doc = line.split('\t')[0].strip()
                self.all_doc.append(doc)
        self.all_doc_list = []
        for doc in self.all_doc:
            tmp = doc.split('.')
            if len(tmp) == 1:
                doc_list = tmp
            else:
                doc_list = [tmp[0]]+[x for x in doc]
            self.all_doc_list.append(doc_list)
        self.model()
    def model(self):
        self.dictionary = corpora.Dictionary(self.all_doc_list)
        corpus = [self.dictionary.doc2bow(doc) for doc in self.all_doc_list]
        self.tfidf = models.TfidfModel(corpus)
        self.index = similarities.SparseMatrixSimilarity(self.tfidf[corpus], num_features=len(self.dictionary.keys()))

    def jieba_cut(self,doc_test):
        doc_test = doc_test.upper()
        if '.' in doc_test:
            tmp = doc_test.split('.')
        elif ' 'in doc_test:
            tmp = doc_test.split(' ')
        else:
            tmp = [doc_test[:3]]+[doc_test[3:]]
        if len(tmp) == 1:
            doc_test_list = [x for x in tmp]
        else:
            doc_test_list = [tmp[0]]+[x for x in tmp[1:]]
        doc_test_vec = self.dictionary.doc2bow(doc_test_list)
        sim = self.index[self.tfidf[doc_test_vec]]
        tmp_result = sorted(enumerate(sim), key=lambda item: -item[1])
        result = [self.all_doc[tmp_result[i][0]] for i in range(10)]
        val = [tmp_result[i][1] for i in range(10)]
        
        choiced_list_1 = list(zip(result,val))
        choiced_list_2 = list(filter(lambda x:x[1]>0.3,choiced_list_1))
        if len(choiced_list_2) == 0:
            return doc_test
        else:
            choiced_icd = [x[0] for x in choiced_list_2[1:]]
            max_deta = 999
            last_result = np.nan
            for icd in choiced_icd:
                last_num = int(re.findall(r'[A-Za-z\. ]|\d+[-][A-Za-z\. ]|\d+', icd)[-1])
                doc_test_num_list = re.findall(r'[A-Za-z\. ]|\d+[-][A-Za-z\. ]|\d+', doc_test)
                tmp = []
                for val in doc_test_num_list:
                    if val.isdigit() == True:
                        tmp.append(int(val))
                if len(tmp) ==0:
                    doc_test_num = 0
                else:
                    doc_test_num = tmp[-1]
                if abs(last_num-doc_test_num)< max_deta:
                    max_deta = abs(last_num-doc_test_num)
                    last_result = icd
            return last_result


if __name__ == '__main__':
    text_test = Text_similarity()
    text_test_2 = Text_similarity_2()
    result = text_test.jieba_cut('传染病')