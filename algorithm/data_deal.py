#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
    FileName: data_deal.py
    Author: zhangxv
    Date: 2019/5/16
    Description:
"""

import numpy as np


def data_deal_sn():
    """分割数据集，分为三部分：用户类型，用户属性，用户行为"""
    matrix_path = '../data/input/users_filter_vector.csv'
    mat = np.loadtxt(matrix_path, dtype=np.int, delimiter=',', skiprows=1)
    
    access_way_path = '../data/input/access_way.csv'
    user_way_path = '../data/input/user_way.csv'
    attr_path = '../data/input/attr.csv'
    act_path = '../data/input/act.csv'
    
    np.savetxt(access_way_path, mat[:, 1], fmt='%d')
    np.savetxt(attr_path, mat[:, 2:9], fmt='%d')
    np.savetxt(act_path, mat[:, 9:], fmt='%d')
    user_way = user_way_update(mat[:, 1])
    np.savetxt(user_way_path, user_way, fmt='%d')


def user_way_update(mat):
    """访问方式矩阵更新"""
    user_way = np.zeros((len(mat), 3))
    for i in range(len(mat)):
        user_way[i][mat[i]] = 1  # 根据访问方式编码号生成访问方式矩阵
    return user_way


def data_deal_steam():
    """处理steam-200k数据"""
    """
    ID:用户编号  总数 12393
    GameID:游戏编号  总数 5153
    BehaviorID:行为编号，1购买，0玩游戏
    Time:行为
    """
    matrix_path = '../data/steam/steam.csv'
    mat = np.loadtxt(matrix_path, dtype=np.str, delimiter=',', skiprows=1)
    print(mat)
    play_path = '../data/steam/steam_play.csv'
    purchase_path = '../data/steam/steam_purchase.csv'
    play = np.zeros((12393, 5153))
    purchase = np.zeros((12393, 5153))
    for arr in mat:
        if arr[2] == '1':
            # print("purchase")
            purchase[int(arr[0]) - 1][int(arr[1]) - 1] = arr[3]
        else:
            # print("play")
            play[int(arr[0]) - 1][int(arr[1]) - 1] = arr[3]
        # break
    np.savetxt(play_path, play, fmt='%0.1f')
    np.savetxt(purchase_path, purchase, fmt='%d')


if __name__ == '__main__':
    # data_deal_sn()
    # data_deal_steam()
    mat = np.loadtxt('../data/input/user_way.csv', dtype=np.int)
    for i in range(3):
        print(mat[:, i].sum())
