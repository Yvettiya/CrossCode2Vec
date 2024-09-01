# -*- coding: utf-8 -*-
import os
import sys
# from extensions.logger import logger
from const_extract import const_extract
from datastruct_extract import datastruct_extract
from func_extract import func_extract
from string_extract import string_extract

if __name__ == "__main__":
    print("任务启动")
    print("提取对象: " + sys.argv[1])
    print("样本哈希值:" + sys.argv[2])
    try:
        print("开始函数提取")
        func_extract(sys.argv[1],sys.argv[2])
    except Exception as e:
        print(e)

    print("提取结束")
