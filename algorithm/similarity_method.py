#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
    FileName: similarity_method.py
    Author: zhangxv
    Date: 2019/5/16
    Description:
"""

import numpy as np


def cosine(a, b):
    """余弦相似度计算"""
    return (a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum())


def adjust_cosine(a, b):
    """调整余弦相似度计算"""
    aa = a - np.mean(a)
    bb = b - np.mean(b)
    return (aa * bb).sum() / np.sqrt((aa ** 2).sum() * (bb ** 2).sum())


def attribute(a, b):
    """属性相似度计算"""
    weight = [0.055, 0.161, 0.133, 0.055, 0.161, 0.303, 0.133]  # 属性相似度计算权重
    return ((a == b) * weight).sum()


def action(a, b):
    """行为相似度计算"""
    sim = 0
    for k in range(len(a)):
        sim += 2 - 2 / (1 + np.exp(-np.fabs(a[k] - b[k])))  # sigmoid函数
    return sim / len(a)


def attr_act(attr_sim, act_sim, a, b, m=0):
    """用户融合相似度计算
    Args:
        attr_sim:user[i] and user[j] attributes similarity
        act_sim:user[i] and user[j] action similarity
        a:user[i] action vector
        b:user[j] action vector
        m:select method of similarity measure
    Return:
        user[i] and user[j] comprehensive similarity
    """
    
    # 对称相似度，sim[i][j]==sim[j][i]
    c = 2 - 2 / (1 + np.exp(-(a.sum() + b.sum()) / len(a)))
    return attr_sim * c + act_sim * (1 - c)
    
    """
    # 非对称相似度，sim[i][j]!=sim[j][i]
    c1 = 2 - 2 / (1 + np.exp(-a.sum() / len(a)))
    c2 = 2 - 2 / (1 + np.exp(-b.sum() / len(a)))
    return s1 * c1 + s2 * (1 - c1), s1 * c2 + s2 * (1 - c2)
    """


if __name__ == '__main__':
    """a = np.array([1, 2, 3])
    b = np.array([1, 2, 6])
    cos = cosine(a, b)
    print(cos)
    acos = adjust_cosine(a, b)
    print(acos)"""
    
    """ c = np.array([1, 2, 3, 4, 5, 6, 7])
     d = np.array([1, 2, 5, 4, 3, 7, 6])
     attr = attribute(c, d)
     print(attr)
     act = action(c, d)
     print(act)"""
    
    e = np.array([1, 1, 1, 1, 1, 1, 1])
    f = np.array([10, 10, 10, 10, 10, 10, 10])
    
    simij = simji = attr_act(0.5, 0.8, e, f, 0)
    print(simij, simji)
    simij, simji = attr_act(0.5, 0.8, e, f, 1)
    print(simij, simji)
