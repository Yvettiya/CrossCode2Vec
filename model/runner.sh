#!/bin/bash

#####
epochnum=100
margin=1.1 
alpha=0.2 
beta=0.3
datasetnum=15000
models=('HAN' 'code2vec' 'cross2code' )
losses=('CSD2D' 'tripletloss2D')
earlystopping=true
savepath='Results'
learningrate=0.0015
tasktype="src"

for model in "${models[@]}"; do
  for loss in "${losses[@]}"; do
    #
    python modelruner.py --epochnum $epochnum --learningrate $learningrate --margin $margin --alpha $alpha --beta $beta --datasetnum $datasetnum --model $model --loss $loss --earlystopping $earlystopping --savepath $savepath --tasktype $tasktype
  done
done

model3="code2seq"
loss3s=("csd" "default_loss")

for loss2 in "${loss3s[@]}"; do
# # 调用 Python 脚本并传递 learningrate 参数
  python modelruner.py --epochnum $epochnum --learningrate $learningrate --margin $margin --alpha $alpha --beta $beta --datasetnum $datasetnum --model $model3 --loss $loss2 --earlystopping $earlystopping --savepath $savepath --tasktype $tasktype
done



