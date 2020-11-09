# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:03:48 2020

@author: 86156
"""

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Yang
from flask import jsonify, current_app,request
import json
import os
import re
import ahocorasick
import Levenshtein

'''
调用diagnose_judge，输入诊断字符串，匹配成功返回[icd_name,icd_code]，不成功返回[]
'''
from flask import Blueprint
from os.path import dirname, abspath
path = dirname(abspath(__file__))
index9_blu = Blueprint("index9", __name__)
word_list,aid_name,words = {},{},{}
with open(path+'//AID_标准名.txt')as f:
	for i in f:
		if '@#$%^&*'not in i:
			aid,name = i.strip().split('\t')
			word_list[name] = aid
			aid_name[aid] = name
if path+'//words.txt'not in os.listdir():
	with open(path+'//词典.txt',encoding='utf-8')as f:
		with open(path+'//words.txt','w',encoding='utf-8')as f_w:
			for i in f:
				f_w.write('#'+i.strip()+'\n')
				for w in word_list:
					if i.strip() in w:
						f_w.write('\t'+w+'\n')
with open(path+'//words.txt',encoding='utf-8')as f:
	for i in f:
		if i[0] == '#':
			temp = i[1:-1]
			words[i[1:-1]] = set()
		else:
			words[temp].add(i.strip())



'''
调用diagnose_judge，输入诊断字符串，匹配成功返回[icd_name,icd_code]，不成功返回[]
'''

	# path = ''

actree = ahocorasick.Automaton()
for index, word in enumerate(words):
	actree.add_word(word, (index, word))
actree.make_automaton()

def get_match(s, actree=actree):
	res, final = [], []
	for i in actree.iter(s):
		res.append([i[0] - len(i[1][1]) + 1, i[0], i[1][1]])
        
        
	start,end = 0,0
    #start,end = 0,0
	if res != []:
		final.append(res[0])
	for i in res:
		rt = i[0]
		if start <= end:
			final_temp = final.copy()
			flag = None
			for index, j in enumerate(final):
				if j[0] >= i[0] and j[1] <= i[1]:
					final_temp[index] = [-1, -1]
					flag = True
			if flag:
				while [-1, -1] in final_temp:
					final_temp.remove([-1, -1])
				final = final_temp.copy()
				final.append(i)
		else:
			final.append(i)
		end = i[1]
	if len(final) > 1:
		if final[0] == final[1]:
			final.pop(0)
	res = []
	for i in final:
		res.append(i[-1])
	return res

def score(s, match_list, word_list=word_list, aid_name=aid_name):
	r, res = 0, []
	for i in match_list:
		sco = Levenshtein.ratio(i, s)
		if r < sco:
			r = sco
			res = []
			res.append([aid_name[word_list[i]], word_list[i]])
		elif r == sco:
			res.append([aid_name[word_list[i]], word_list[i]])
	return (r, res)

def find_lcseque(s1, s2):
		# 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
	m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
		# d用来记录转移方向
	d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
	for p1 in range(len(s1)):
		for p2 in range(len(s2)):
			if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
				m[p1 + 1][p2 + 1] = m[p1][p2] + 1
				d[p1 + 1][p2 + 1] = 'ok'
			elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
				m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
				d[p1 + 1][p2 + 1] = 'left'
			else:  # 上值大于左值，则该位置的值为上值，并标记方向up
				m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
				d[p1 + 1][p2 + 1] = 'up'
	(p1, p2) = (len(s1), len(s2))
	s = []
	while m[p1][p2]:  # 不为None时
		c = d[p1][p2]
		if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
			s.append(s1[p1 - 1])
			p1 -= 1
			p2 -= 1
		if c == 'left':  # 根据标记，向左找下一个
			p2 -= 1
		if c == 'up':  # 根据标记，向上找下一个
			p1 -= 1
	s.reverse()
	return ''.join(s)
def diagnose_judge(seq, score=score, get_match=get_match, words=words, word_list=word_list):
	word = get_match(seq)
	if word == []:
		sco, res = score(seq, word_list)
	else:
		candidate = set()
		for i in word:
			candidate.update(words[i])
		sco, res = score(seq, candidate)

	if sco >= 0.6 and len(res) == 1:
		return {'result': [{'name': res[0][0], 'origin_value': seq, 'rate': sco,'match_seq':find_lcseque(res[0][0],seq),'label':''}]}
	else:
		return {'result': [{'name': '','origin_value': seq,  'rate': 0.,'label':'','match_seq':''}]}

	# seq = '1型糖尿病伴多并发症'
result = diagnose_judge('1型糖尿病伴多并发症')


