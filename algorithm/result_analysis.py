#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
    FileName: result_analysis.py
    Author: zhangxv
    Date: 2019/5/17
    Description:
"""

from sn_recommend.algorithm.result_deal import *
from sn_recommend.algorithm.show_fig import *
from sn_recommend.algorithm.user_cf import UserCF


def sim_acc():
    """用户综合相似度结果"""
    user_way = np.loadtxt(path['user_way'], np.int)
    user_num = len(user_way)
    user_cf = UserCF(user_way, user_num, threshold=0.5)
    user_cf.sim = np.loadtxt(path['user_sim'])
    x = np.arange(5, 100, 5).tolist()
    result = {'K': x}
    for eva in evaluations:
        result[eva] = np.zeros(len(x)).tolist()
    
    for i in range(len(x)):
        user_cf.k = x[i]
        user_cf.train()
        res = user_cf.measure()
        for eva in res:
            result[eva][i] = res[eva]
    print(result)
    save_json(result, '../data/result/user_sim_th(0.5).json')
    label1['title'] = 'Accuracy变化趋势'
    show_accuracy(x, result, label1)


def sim_th_acc(sim):
    user_way = np.loadtxt(path['user_way'], np.int)
    user_num = len(user_way)
    user_cf = UserCF(user_way, user_num)
    user_cf.sim = np.loadtxt(path[sim])
    x = np.arange(5, 100, 5).tolist()
    result = dict()
    for th in thresholds:
        result[th] = dict()
        user_cf.th = th
        for eva in evaluations:
            result[th][eva] = np.zeros(len(x)).tolist()
        for i in range(len(x)):
            user_cf.k = x[i]
            user_cf.train()
            res = user_cf.measure()
            for eva in res:
                result[th][eva][i] = res[eva]
    label1['title'] = '%s不同阈值预测结果Accuracy对比' % names[sim]
    # print(result)
    save_json(result, '../data/result/th(0.3-0.7)/%s_th_k.json' % names[sim])
    compare_accuracy(x, result, label1, th_legend, '../img/th(0.3-0.7)/' + label1['title'])


def sim_compare_acc():
    user_way = np.loadtxt(path['user_way'], np.int)
    user_num = len(user_way)
    user_cf = UserCF(user_way, user_num, threshold=0.5)
    
    x = np.arange(5, 100, 5).tolist()
    result = dict()
    for sim in names:
        print('方法:' + sim)
        result[sim] = dict()
        user_cf.sim = np.loadtxt(path[sim])
        for eva in evaluations:
            result[sim][eva] = np.zeros(len(x)).tolist()
        for i in range(len(x)):
            user_cf.k = x[i]
            user_cf.train()
            res = user_cf.measure()
            for eva in res:
                result[sim][eva][i] = res[eva]
    label1['title'] = '不同相似度方法Accuracy对比'
    result = get_json('../data/result/compare_acc_th(0.5).json')
    print(result)
    save_json(result, '../data/result/compare_acc_th(0.5).json')
    compare_accuracy(x, result, label1, names, '../img/' + label1['title'])
    a = [names[name] for name in names]
    b = [np.max(result[sim]['accuracy']) for sim in names]
    show_hist(a, b, label3, '../img/' + label3['title'])


def sim_compare_roc():
    user_way = np.loadtxt(path['user_way'], np.int)
    user_num = len(user_way)
    user_cf = UserCF(user_way, user_num, neighbor=25)
    result = dict()
    a = [x / 100 for x in range(-1, 101, 1)]
    for sim in names:
        print('方法:' + sim)
        result[sim] = dict()
        for eva in evaluations:
            result[sim][eva] = np.zeros(user_num + 1).tolist()
        user_cf.sim = np.loadtxt(path[sim])
        user_cf.train()
        user_cf.ways = [0]
        for i in range(len(a)):
            user_cf.th = a[i]
            res = user_cf.measure()
            for eva in res:
                result[sim][eva][i] = res[eva]
    print(result)
    save_json(result, '../data/result/0_compare_roc.json')
    result = get_json('../data/result/0_compare_roc.json')
    show_roc(result, label2, names, '../img/0_' + label2['title'])


if __name__ == '__main__':
    path = {
        # 数据集路径
        'attr': '../data/input/attr.csv',
        'act': '../data/input/act.csv',
        'user_way': '../data/input/user_way.csv',
        # 相似度矩阵路径
        'attr_sim': '../data/sim/attr_sim.csv',
        'act_sigmoid_sim': '../data/sim/act_sigmoid_sim.csv',
        'act_cos_sim': '../data/sim/act_cos_sim.csv',
        'act_acos_sim': '../data/sim/act_acos_sim.csv',
        'user_sim': '../data/sim/user_sim.csv',
        'cos_sim': '../data/sim/cos_sim.csv',
        'acos_sim': '../data/sim/acos_sim.csv',
    }
    names = {'attr_sim': 'Attr-Sim',
             'act_sigmoid_sim': 'Act-Sim',
             'user_sim': 'Mixed-Sim',
             'act_cos_sim': 'COS-Sim',
             'cos_sim': 'Mixed-COS-Sim',
             'act_acos_sim': 'ACOS-Sim',
             'acos_sim': 'Mixed-ACOS-Sim'}  # 相似度方法集合
    evaluations = ['accuracy', 'precise', 'recall', 'tpr', 'fpr', 'f1score']  # 评价指标
    label1 = {'x': '近邻数目K', 'y': 'Accuracy', 'title': ''}  # 结果图例label
    label2 = {'x': '假正例率FPR', 'y': '真正例率TPR', 'title': '不同相似度方法ROC曲线对比'}
    label3 = {'x': '相似度方法', 'y': 'Accuracy', 'title': '不同相似度方法最优情况下Accuracy对比'}
    thresholds = [x / 10 for x in range(3, 8, 1)]  # 预测值阈值
    th_legend = {-0.1: 'threshold=-0.1', 0.0: 'threshold=0.0', 0.1: 'threshold=0.1', 0.2: 'threshold=0.2',
                 0.3: 'threshold=0.3', 0.4: 'threshold=0.4', 0.5: 'threshold=0.5', 0.6: 'threshold=0.6',
                 0.7: 'threshold=0.7', 0.8: 'threshold=0.8', 0.9: 'threshold=0.9', 1.0: 'threshold=1.0'}
    # sim_acc()
    # for k in names:
    #     sim_th_acc(k)
    # sim_compare_acc()
    sim_compare_roc()
