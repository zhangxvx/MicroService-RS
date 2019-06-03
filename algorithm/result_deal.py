#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
    FileName: result_deal.py
    Author: zhangxv
    Date: 2019/5/23
    Description:
"""
import json


def save_json(result, file):
    with open(file, 'w') as f:
        json.dump(result, f, indent=2)


def get_json(file):
    with open(file, 'r') as f:
        res = json.load(f)
    return res
