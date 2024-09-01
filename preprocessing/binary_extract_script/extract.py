# encoding: utf-8 



from idc import *
from idaapi import *
import idc
import idautils
import json

import idaapi


import ida_nalt
# import uuid
import json



def get_pseudocode(function_start):

    cfunc = idaapi.decompile(function_start)

    cfuninsert = []
    cfunstr = str(cfunc)
    cfunclist = cfunstr.split('\n')
    for a in cfunclist:
        if a == '\n':
            continue
        else:
            cfuninsert.append(a) 
    
    print("  ** finish pseudocode analysis**  ")
    return cfuninsert



def edgenum(cfg):
    edge_num_add = 0
    for block in cfg:
        succs = list(block.succs())
        edge_num = len(succs)
        edge_num_add = edge_num_add + edge_num
    return edge_num_add


def get_file_name():
    file_path = ida_nalt.get_root_filename()
    # print(file_path)
    if "_" in file_path:
        files=file_path.split("_")
        real_name=''.join(files[1:])
    else:
        real_name=file_path
    # print(real_name)
    return real_name

def get_eachfuncinfo(func_start,func_end):
    # global function_assembly
    flag={}
    name = idc.get_func_name(func_start)
    flag['func_name']=name
    # flag['func_start_ea_hex'] = hex(func_start)
    # flag['func_end_ea_hex'] = hex(func_end)
    flag['func_start_ea'] = hex(func_start)
    flag['func_end_ea'] = hex(func_end)

    flag['assembly']=[]
    for head in idautils.Heads(func_start, func_end):
        a = idc.generate_disasm_line(head, 0)
        flag['assembly'].append(a)
    # function_assembly.append(flag)
    print("   **finish function analysis**   ")
    return flag

def get_func_fromsegment():
    functions_results=[]
    
    index = 0
    for ea in idautils.Segments():



        if idc.SegName(ea) != '.text':


            continue


        functions = list(idautils.Functions(idc.get_segm_start(ea), idc.get_segm_end(ea)))
        functions.append(idc.get_segm_end(ea))
        for i in range(len(functions) - 1):

            function_info = dict()
            
            function_start = functions[i]
            function_end = functions[i + 1]

            print('function_start:',hex(function_start))

            function_info['Function']  = get_eachfuncinfo(function_start,function_end)
            
            function_info['CFG']  = get_cfg(function_start,function_end)

            function_info['PseudoCode']  = get_pseudocode(function_start)

            function_info['CallChain'] = get_callchain(function_start)
            

            functions_results.append(function_info)


    return functions_results

# process all functions
def get_func_xref(ea):
    funcs=set()
    funcs_xref = list()
    t=get_first_cref_to(ea)
    # print("t:",t)
    while t!=BADADDR:

        cc = dict()
        cc['caller'] = idaapi.get_func_name(t)
        cc['caller_addr'] = t
        cc['callee'] = idaapi.get_func_name(ea)
        cc['callee_addr'] = ea

        funcs_xref.append(cc)

        funcs.add( tuple((hex(t),hex(ea)) ))
        t=get_next_cref_to(ea, t)
        # print(funcs)
    
    return list(funcs),funcs_xref


def get_callchain(function_start):

    call_from,funcs_xref = get_func_xref(function_start)

    result = dict()
    result['call_pair'] = call_from
    result['call_info'] = funcs_xref

    print("  ** finish callchain analysis**  ")

    return result




def print_cfg_node(cfg):

    funcblocks = dict()
    index = 0
    
    for block in cfg:
        blocknode = {}

        blocknode["blockid"] = int(block.id)
        blocknode["block_ea"] = hex(block.startEA)
        blocknode["block_size"] = int(str(block.endEA - block.startEA) , 16)
        blocknode["block_assem"] = []
        for head in idautils.Heads(block.startEA, block.endEA): 
            blocknode["block_assem"].append(hex(head)+":"+GetDisasm(head)) 
            
        funcblocks[index] = blocknode
        index += 1

    return funcblocks


def print_cfg_edge(cfg):

    edges = dict()
    
    index = 0



    for block in cfg:

        succs = list(block.succs())
        edge_num = len(succs)

        if edge_num == 0:
            #logging.debug("   no edge")
            break
        
        for i in range(edge_num):

            info = {}
            info["src_id"] = block.id
            info["dst_id"] = succs[i].id
            info["src_ea"] = hex(block.startEA)
            info["dst_ea"] = hex(succs[i].startEA)
            edges[index] = info 
            index += 1

        
    return edges

 

def get_cfg(function_start, function_end):

    dictfunction = {}
    
    cfg = idaapi.FlowChart(idaapi.get_func(function_start))
    dictfunction['nodes_num'] = cfg.size
    dictfunction['edges_num'] = edgenum(cfg)

    dictfunction['nodes'] = print_cfg_node(cfg)

    dictfunction['edges'] = print_cfg_edge(cfg)

    print("  ** finish CFG analysis**  ")

    return dictfunction




if __name__ == '__main__':

    idc.Wait()

    functins = get_func_fromsegment()

    datapath = os.getcwd()
    filename = ida_nalt.get_root_filename()

    resultdir = datapath+'//extract_output'
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    
    resultname = resultdir+'//'+filename+'.json'
    with open(resultname,'w') as f:
        json.dump(functins,f)


    idc.Exit(0)