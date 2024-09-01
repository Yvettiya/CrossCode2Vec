import yaml
import os
import copy
import time

#findutils-4.6-O2
template_path = "/root/astminer/configs/templete.yaml"

root_path = "/home/user/astminer/match_pairs"
# input_path = os.path.join(root_path,'src')
# output_path = os.path.join(root_path,'code2vecresults','bin')
thisyamlfilepath = "/root/astminer/my_configs"
funcindex = 1

def get_template(template_path):
    with open(template_path,'r') as f:
        tmplete_data = yaml.safe_load(f.read())
        
    return tmplete_data


def mkdirofpath(path):
    if not os.path.exists(path):
       
        os.makedirs(path)


def write_yaml(name,data):
    
    with open (os.path.join(thisyamlfilepath,name),'w') as f:
        yaml.safe_dump(data, f,sort_keys=False)
        

def do_code2vec_project(input_path,output_path,flag,projname):
    global funcindex
    templete_data = get_template(template_path)

    # for projname in os.listdir(input_path):
        
    projpath = os.path.join(input_path,projname)
    
    for funcname in os.listdir(projpath):
        funcpath = os.path.join(projpath,funcname)
        
        thisyaml = copy.deepcopy(templete_data)
        
        thisyaml['inputDir'] = funcpath
        
        thisyaml['outputDir'] = os.path.join(output_path,projname,funcname)
        
        
        thisyamlfilename = flag+'+'+projname+'+'+funcname+'.yaml'
        
        write_yaml(thisyamlfilename,thisyaml)
        
        
        mkdirofpath(funcpath)
        
        mkdirofpath(os.path.join(output_path,projname,funcname))

        os.system("/bin/bash /root/astminer/cli.sh /root/astminer/my_configs/"+thisyamlfilename)

        print("*********",funcindex,flag,projname,funcname,"*********")

        funcindex +=1

        # time.sleep(10)

            

        
if __name__ == "__main__":   

    projname  = "findutils-4.6-O2"  

    src_input_path = os.path.join(root_path,'src')  
    src_output_path = os.path.join(root_path,'code2vecresults','src')


    do_code2vec_project(src_input_path,src_output_path,'src',projname)

    # bin_input_path = os.path.join(root_path,'bin')
    # bin_output_path = os.path.join(root_path,'code2vecresults','bin')

    # do_code2vec_project(bin_input_path,bin_output_path,'bin',projname)