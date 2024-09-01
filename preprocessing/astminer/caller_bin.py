import yaml
import os
import copy
import time


template_path = "/root/astminer/configs/templete.yaml"

root_path = "/home/user/astminer/match_pairs"
# input_path = os.path.join(root_path,'src')
output_path = os.path.join(root_path,'code2vecresults','bin')
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
        

def do_code2vec(input_path,flag):
    global funcindex
    templete_data = get_template(template_path)

    for projname in os.listdir(input_path):
        
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

            print("*********",funcindex,projname,funcname,"*********")

            funcindex +=1

            # time.sleep(10)

            

        
if __name__ == "__main__":     
    # input_path = os.path.join(root_path,'src')    
    # do_code2vec(input_path,'src')
    input_path2 = os.path.join(root_path,'bin')

    do_code2vec(input_path2,'bin')