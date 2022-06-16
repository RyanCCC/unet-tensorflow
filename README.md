# Unet-tensorflow

Unet网络结构比较简单，适用与一些数据量比较少的任务，如医学图像、工业残次品检测等。具体的Unet介绍以及应用实例可参考我的一篇博客：[地物分类：基于Unet的建筑物轮廓识别](https://blog.csdn.net/u012655441/article/details/120373759)。

## 文件结构

```
│--GenerateDataset.py 划分训练集，测试集，验证集的脚本
│--inference.py  推理脚本
│--miou.py
│--processImage.py  处理图像脚本
│--README.md  
│--requirements.txt  
│--train.py  训练
│--unet_config.cfg  配置文件
├─logs  训练日志
│  └─train
├─model  保存的模型
├─nets  网络
│  │  unet.py  基础网络结构
│  │  unet_inference.py  推理的网络
│  │  unet_training.py   训练的网络
│  └─ vgg16.py  backbone
├─result
│      your result
└─utils
    └─metrics.py 评价指标
```

## 数据集

如何根据自己的任务制作训练数据，请查看我的博客：[图像分割数据集制作](https://blog.csdn.net/u012655441/article/details/120370578) ，对应代码可查看：[mantic-Segmentation-Datasets](https://github.com/RyanCCC/Semantic-Segmentation-Datasets)

## 训练
通过```pip install -r requiremetstxt```安装好相关的依赖以及修改好配置文件后，执行下面命令进行训练：
```sh
python train.py
```

当然也可以在命令行使用参数形式运行，本人比较习惯在windows环境下进行调试测试通过后再搬到服务器上进行训练。

## 推理

```sh
python inference.py
```

