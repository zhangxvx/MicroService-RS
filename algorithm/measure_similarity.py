#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
    FileName: measure_similarity.py
    Author: zhangxv
    Date: 2019/5/16
    Description:
"""

from sn_recommend.algorithm.similarity_method import *


def attr_similarity(matrix, num):
    """计算用户属性相似度"""
    sim = np.zeros((num, num))
    for i in range(num):
        for j in range(i + 1, num):
            sim[i][j] = sim[j][i] = attribute(matrix[i], matrix[j])
    return sim


def act_sigmoid_similarity(matrix, num):
    """计算用户行为相似度"""
    sim = np.zeros((num, num))
    for i in range(num):
        for j in range(i + 1, num):
            sim[i][j] = sim[j][i] = action(matrix[i], matrix[j])
    return sim


def act_cos_similarity(matrix, num):
    """计算用户属性相似度"""
    sim = np.zeros((num, num))
    for i in range(num):
        for j in range(i + 1, num):
            sim[i][j] = sim[j][i] = cosine(matrix[i], matrix[j])
    return sim


def act_acos_similarity(matrix, num):
    """计算用户属性相似度"""
    sim = np.zeros((num, num))
    for i in range(num):
        for j in range(i + 1, num):
            sim[i][j] = sim[j][i] = adjust_cosine(matrix[i], matrix[j])
    return sim


def attr_act_similarity(sim1, sim2, act_mat, num):
    """计算用户融合相似度"""
    sim = np.zeros((num, num))
    for i in range(num):
        for j in range(i + 1, num):
            sim[i][j] = sim[j][i] = attr_act(sim1[i][j], sim2[i][j], act_mat[i], act_mat[j])
    return sim


if __name__ == '__main__':
    path = {
        # 输入数据集路径
        'attr': '../data/input/attr.csv',
        'act': '../data/input/act.csv',
        # 输出相似度矩阵路径
        'attr_sim': '../data/sim/attr_sim.csv',
        'act_sigmoid_sim': '../data/sim/act_sigmoid_sim.csv',
        'act_cos_sim': '../data/sim/act_cos_sim.csv',
        'act_acos_sim': '../data/sim/act_acos_sim.csv',
        'user_sim': '../data/sim/user_sim.csv',
        'cos_sim': '../data/sim/cos_sim.csv',
        'acos_sim': '../data/sim/acos_sim.csv',
    }
    attr_matrix = np.loadtxt(path['attr'], np.int)
    act_matrix = np.loadtxt(path['act'], np.int)
    user_num = len(attr_matrix)
    
    print("计算用户属性相似度：")
    attr_sim = attr_similarity(attr_matrix, user_num)
    np.savetxt(path['attr_sim'], attr_sim, fmt='%0.3f')
    
    print("计算用户行为sigmoid相似度：")
    act_sigmoid_sim = act_sigmoid_similarity(act_matrix, user_num)
    np.savetxt(path['act_sigmoid_sim'], act_sigmoid_sim, fmt='%0.3f')
    
    print("计算用户行为cos相似度：")
    act_cos_sim = act_cos_similarity(act_matrix, user_num)
    np.savetxt(path['act_cos_sim'], act_cos_sim, fmt='%0.3f')
    
    print("计算用户行为acos相似度：")
    act_acos_sim = act_acos_similarity(act_matrix, user_num)
    np.savetxt(path['act_acos_sim'], act_acos_sim, fmt='%0.3f')
    
    print("计算属性+行为sigmoid用户相似度：")
    user_sim = attr_act_similarity(attr_sim, act_sigmoid_sim, act_matrix, user_num)
    np.savetxt(path['user_sim'], user_sim, fmt='%0.3f')
    
    print("计算属性+行为cos用户相似度：")
    cos_sim = attr_act_similarity(attr_sim, act_cos_sim, act_matrix, user_num)
    np.savetxt(path['cos_sim'], cos_sim, fmt='%0.3f')
    
    print("计算属性+行为acos用户相似度：")
    acos_sim = attr_act_similarity(attr_sim, act_acos_sim, act_matrix, user_num)
    np.savetxt(path['acos_sim'], acos_sim, fmt='%0.3f')
