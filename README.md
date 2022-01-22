# rmdet
code base of object detection.

## 环境安装：
    1. 安装conda python环境   
    - `conda create -n xxx python=3.7/3.8`  
    - `conda activate xxx`
    2. 运行脚本，自动安装pytorch1.8.0环境
    - `bash install.sh'

## 训练：
    1. 修改`configs`里的模型配置文件，包括数据集路径，类别等；
    2. 运行`train.py`训练，设置config，epochs，gpu等：
    - `python train.py -c configs/xxx.yaml --epochs * --n_gpu *`  

## 测试：
    `python val.py -c configs/xxx.yaml --weights (训练完成的模型权重)`

## 推理：
    `python detect.py -c configs/xxx.yaml --weights (权重) -s (voc/cam)`


## Reference:

 - YOLOv5: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
 - MMDetection:[https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
 - DBB: [https://github.com/DingXiaoH/DiverseBranchBlock](https://github.com/DingXiaoH/DiverseBranchBlock)
 - RepVGG:[https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
 - RetinaNet:[https://github.com/zhenghao977/RetinaNet-Pytorch-36.4AP](https://github.com/zhenghao977/RetinaNet-Pytorch-36.4AP)
