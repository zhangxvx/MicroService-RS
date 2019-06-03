#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
    FileName: user_cf.py
    Author: zhangxv
    Date: 2019/5/16
    Description:
"""
import numpy as np


class UserCF:
    
    def __init__(self, way, num, sim=[], neighbor=20, top=150, threshold=0.5):
        self.sim = sim
        self.user_way = way
        self.user_num = num
        self.k = neighbor  # 最近邻数目
        self.n = top  # top-N数目
        self.th = threshold  # 预测值阈值
        self.ways = np.arange(3)  # 分类数目
        self.predict = np.zeros((self.user_num, 3))
    
    def train(self):
        """模型训练——根据相似度计算预测值"""
        # print('训练...')
        for i in range(self.user_num):
            nb_top = np.argsort(self.sim[i])[-self.k:]
            sim_top = self.sim[i][nb_top]
            way_top = self.user_way[nb_top]
            
            a = np.dot(sim_top, way_top)
            b = sim_top.sum()
            self.predict[i] = a / b if b != 0 else 0  # 防止相似度和为0，导致除数为0\
        self.predict = np.around(self.predict, decimals=3)
    
    def measure(self):
        """模型评估——根据预测值评价误差"""
        # print('评估...')
        result = {'accuracy': 0, 'precise': 0, 'recall': 0, 'tpr': 0, 'fpr': 0, 'f1score': 0}
        tp = 0  # 预测为正，实际为正
        tn = 0  # 预测为负，实际为负
        fp = 0  # 预测为正，实际为负
        fn = 0  # 预测为负，实际为正
        for i in self.ways:
            tp = ((self.predict[:, i] > self.th) & (self.user_way[:, i] == 1)).sum()
            tn = ((self.predict[:, i] <= self.th) & (self.user_way[:, i] == 0)).sum()
            fp = ((self.predict[:, i] > self.th) & (self.user_way[:, i] == 0)).sum()
            fn = ((self.predict[:, i] <= self.th) & (self.user_way[:, i] == 1)).sum()
            # print("tp tn fp fn:", tp, tn, fp, fn)
            result['accuracy'] += (tp + tn) / self.user_num  # 准确率
            result['precise'] += tp / (tp + fp) if tp + fp > 0 else 0  # 查准率，精准率
            result['recall'] += tp / (tp + fn)  # 查全率，召回率
            result['tpr'] += tp / (tp + fn)  # 真正例率
            result['fpr'] += fp / (fp + tn)  # 假正例率
            result['f1score'] += 2 * tp / (2 * tp + fp + fn)  # f1-score
        for key in result:
            result[key] = result[key] / len(self.ways)
            # result[key] = np.around(result[key] / len(self.ways), decimals=3)
        return result


if __name__ == '__main__':
    file_path = {
        # 相似度矩阵路径
        'attr_sim_path': '../data/sim/attr_sim.csv',
        'act_sim_path': '../data/sim/act_sim.csv',
        'user_sim_path': '../data/sim/user_sim.csv',
        # 用户访问方式标识数据集路径
        'user_way_path': '../data/input/user_way.csv',
        # 预测访问方式路径
        'predict_path': '../data/predict/user_sim_predict.csv',
        'recommend_path': '../data/predict/user_sim_recommend.csv',
    }
    user_way = np.loadtxt(file_path['user_way_path'], np.int)
    user_num = len(user_way)
    
    # attr_sim = np.zeros((user_num, user_num))
    # act_sim = np.zeros((user_num, user_num))
    # user_sim = np.zeros((user_num, user_num))
    
    user_cf = UserCF(user_way, user_num)
    user_cf.sim = np.loadtxt(file_path['user_sim_path'])
    # print(user_cf.predict)
    user_cf.train()
    # np.savetxt(file_path['predict_path'], user_cf.predict, fmt='%0.3f')
    res = user_cf.measure()
    print(res)
