import os

datasetrootpath = "C://ida_script//dataset//binary"

index = 0

for arch in os.listdir(datasetrootpath):

    for proj in os.listdir(datasetrootpath+"//"+arch):

        for proj_version in os.listdir(datasetrootpath+"//"+arch+'//'+proj):

            for filename in os.listdir(datasetrootpath+"//"+arch+'//'+proj+'//'+proj_version):
                binary_full_path = datasetrootpath+"//"+arch+'//'+proj+'//'+proj_version+'//'+filename

                # command = "ida64 -c -LC:/ida_script/ida.log -A -S"+'"'+"C:/ida_script/extract.py "+binary_full_path+'" "'+binary_full_path+'"'
                command = "ida -c -LC:/ida_script/ida.log -A -SC:/ida_script/extract.py "+'"'+binary_full_path+'"'

                print(command)
                os.system(command=command)

                index +=1
                if index > 8:
                    exit()

                
