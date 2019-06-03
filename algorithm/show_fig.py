#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
    FileName: show_fig.py
    Author: zhangxv
    Date: 2019/5/17
    Description:
"""
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=3)  # 控制元素小数点后位数为3
# plt.style.use('seaborn-darkgrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
markers = ['+', 'o', '*', '.', 'x', 's', 'd', '^', 'v', '>', '<', 'p', 'h']


def show_accuracy(x, y, label, lim=(), ticks=()):
    plt.plot(x, y['accuracy'], marker='+')
    plt.xlabel(label['x'])
    plt.ylabel(label['y'])
    # plt.ylim(lim)
    # plt.yticks(ticks)
    plt.title(label['title'])
    plt.show()


def compare_accuracy(x, y, label, legend, path, lim=()):
    i = 0
    plt.figure('compare_accuracy')
    for key in y:
        plt.plot(x, y[key]['accuracy'], label=legend[key], marker=markers[i])
        i += 1
    plt.xlabel(label['x'])
    plt.ylabel(label['y'])
    # plt.ylim(lim)
    plt.title(label['title'])
    plt.legend(loc='best', bbox_to_anchor=(0.9, 0.5), ncol=2)
    plt.savefig(path)
    plt.show()


def show_roc(y, label, legend, path):
    i = 0
    for key in y:
        plt.plot(y[key]['fpr'], y[key]['tpr'], label=legend[key])
        i += 1
    plt.plot(np.arange(0, 2), np.arange(0, 2), linestyle=':')
    plt.xlabel(label['x'])
    plt.ylabel(label['y'])
    plt.title(label['title'])
    plt.legend(loc='lower right')
    plt.savefig(path)
    plt.show()


def show_hist(x, y, label, path):
    plt.figure('show_hist', figsize=(8, 5))
    y = np.around(y, 3)
    react = plt.bar(x=np.arange(len(x)), height=y)
    for rect in react:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha="center", va="bottom")
    
    plt.xlabel(label['x'])
    plt.ylabel(label['y'])
    plt.xticks(np.arange(len(x)), x)
    plt.yticks(np.linspace(0.80, 0.95, 7))
    plt.ylim((0.80, 0.95))
    plt.title(label['title'])
    plt.savefig(path)
    plt.show()


if __name__ == '__main__':
    """main函数"""
    # show_hist([2012, 2013, 2014, 2015], [100, 200, 300, 400], {'x': 'x', 'y': 'y', 'title': 'title'})
    # show_accuracy([1, 2, 3, 4], {'accuracy': [0.1, 0.2, 0.4, 0.8]}, {'x': 'x', 'y': 'y', 'title': 'title'})
    # show_roc({'attr': {'fpr': [0, 0.2, 1], 'tpr': [0, 0.8, 1]}}, {'x': 'x', 'y': 'y', 'title': 'title'})
