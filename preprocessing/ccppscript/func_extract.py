# -*- coding: utf-8 -*-
import json

# from cassandra.cluster import Cluster
# from extensions.logger import logger

from . import func_call_process
from . import func_define_process
# from . from save_db import save_db

struct_dict = {}
func_dict = {}
func_list = []


cpptype = {'bool', 'char', 'int', 'float', 'double', 'wchar_t', 'enum', 'long', 'short', 'size_t', 'string', 'int8_t',
           'int16_t', 'int32_t', 'int64_t', 'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t'}


def save_json(save_path, data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path, 'w') as file:
        json.dump(data, file, indent=4)


def func_extract(path,hash):
    lastname = str()

    with open(path + '/tags') as tags_file:

        for (tags_line_num, tags_line_value) in enumerate(tags_file):
            print(tags_line_num, tags_line_value)
            if tags_line_num <= 10:
                continue
            # 分割tags文件行得到标签list:tag
            tag = tags_line_value.split("\t")

            # 有时tag[0]会返回一个运算符的表达式
            # symbol_name = tag[0].split(' ')[0]
            symbol_name = tag[0]
            print("symbol_name:",symbol_name)
            # if " " in tag[0]:
            #     continue

            try:
                if tag.count('f') > 1:
                    mtag_index = tag.index('f',1,4)
                else:
                    mtag_index = tag.index("f")
            except:
                continue
            # -----------函数相关指标-------------
            if tag[mtag_index] == "f":
                if symbol_name != lastname:
                    # 输出函数名
                    func_name = symbol_name
                    print("func_name:",func_name)
                    define_list = list()
                    call_list = []
                    very_func_dict = dict()
                    very_func_dict['func_name'] = func_name
                    # -----------函数定义相关指标-------------
                    try:
                        define_list = func_define_process.func_define_process(
                        mtag_index, path, tag)
                    except Exception as e :
                        # print(e)
                        continue
                    very_func_dict['define_list'] = define_list
                    # -----------函数调用相关指标-------------
                    call_list = func_call_process.func_call_process(
                        func_name, path)
                    very_func_dict['call_list'] = call_list
                    func_list.append(very_func_dict)
                else:
                    try:
                        define_list = define_list + func_define_process.func_define_process(
                            mtag_index, path, tag)
                        func_list[-1]["define_list"] = define_list
                    except Exception as e:
                        # print(e)
                        continue
                lastname = symbol_name
                
    save_json(path + "/function.json", func_list)
    return func_list
#     save_db(hash,path, "func", func_list)
    
