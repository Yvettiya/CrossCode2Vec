# -*- coding: utf-8 -*-
import re
import json
from extensions.logger import logger
from save_db import save_db
import ssdeep

def save_json(save_path, data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path, 'w') as file:
        json.dump(data, file, indent=4)


str_pattern_one = re.compile(r'\"(.*?)\"')
str_pattern_two = re.compile(r'\'(.*?)\'')


def string_extract(path,hash):
    file_string_list = list()
    file_info_list = list()
    with open(path + '/cscope.files') as cscope_files:
        for line in cscope_files:
            project_line_num = 0
            comment_line_num = 0
            file_dict = dict()
            string_num = 0
            line = line.strip('\n,"')
            line = line.strip('"')
            # try:
            with open(line) as source_file:
                include_info = str()
                file_name = line.split('/')[-1]
                file_dict['file_name'] = file_name
                file_content = str(source_file.read())
                file_dict['file_content'] = file_content
                file_dict['hash'] = ssdeep.hash(str(file_content))
                norm_list = list()
                string_dict = {"file": line, "string_list": list()}
                # 逐行提取"abc"内和'abc'内两种字符串
                source_file = file_content.split('\n')
                for (line_num,line_value) in enumerate(source_file):
                    try:
                        if '//' in line_value:
                            comment_line_num = comment_line_num+1
                        else:
                            project_line_num = project_line_num+1
                    except:
                        continue
                    norm_line = line_value.strip()
                    if ' ' in line_value:
                        norm_line = line_value.split(' ')
                        while '' in norm_line:
                            norm_line.remove('')
                        norm_list.append(' '.join(norm_line))
                    else:
                        norm_list.append(norm_line)
                    if '#include' in line_value:
                        include_info = include_info + line_value
                    strs_one = str_pattern_one.findall(line_value)
                    strs_two = str_pattern_two.findall(line_value)
                    strs = strs_one + strs_two
                    string_num = string_num + len(strs)
                    if len(strs):
                        for (i, val) in enumerate(strs):
                            very_string_dict = dict()
                            very_string_dict["string_content"] = val
                            very_string_dict["line_num"] = line_num
                        string_dict["string_list"].append(very_string_dict)
                    
                file_dict['norm_file_content'] = '\n'.join(norm_list)
                file_dict['norm_hash'] = ssdeep.hash('\n'.join(norm_list))
                file_dict['relative_path'] = '/'.join(line.split('/')[2:])
                file_dict['include_info'] = include_info
                file_dict['code_line_num'] = project_line_num
                file_dict['comment_line_num'] = comment_line_num
                string_dict["string_num"] = string_num
                file_string_list.append(string_dict)
                file_func_list = list()
                norm_func_list = list()
                with open(path+'/function.json') as func_json:
                    func_list = json.load(func_json)
                    for func in func_list:
                        func_origin = func
                        func = func['define_list']
                        for define in func:
                            if file_dict['relative_path'] == define['file']:
                                file_func_list.append(func_origin['func_name'])
                                norm_func_list.append(define['norm_code_block'])
                                try:
                                    for very_string in string_dict["string_list"]:
                                        if very_string['line_num'] >= int(define['line']) and very_string['line_num'] <= define['endline']:
                                                very_string['reference'] = func_origin['func_name']
                                except Exception as e:
                                    logger.exception(e)
                                    continue
                file_struct_list = list()
                norm_struct_list = list()
                with open(path+'/datastruct.json') as struct_json:
                    struct_list = json.load(struct_json)
                    for struct in struct_list:
                        try:
                            if file_dict['relative_path'] == '/'.join(struct['struct_def_file_name'].split('/')[2:]):
                                file_struct_list.append(struct['name'])
                                norm_struct_list.append(struct['norm_struct_def_code_block'])
                        except Exception as e:
                            #logger.error(e)
                            continue
                file_dict['func_list'] = file_func_list
                file_dict['norm_func_list'] = norm_func_list
                file_dict['struct_list'] = file_struct_list
                file_dict['norm_struct_list'] = norm_struct_list
                file_info_list.append(file_dict)
            # except Exception as e:
            #     continue
    save_json(path + "/string.json", file_string_list)
    save_json(path + "/file.json", file_info_list)
    save_db(hash,path,"string",file_string_list)
    save_db(hash,path,"file",file_info_list)

