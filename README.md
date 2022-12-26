# Fault detection based on deep learning（CWRU Dataset）

> 该代码主要参考论文和代码放在最后，主要用来研究和学习一些有关故障检测的深度学习算法和CWRU轴承数据集
>
> 也对该代码进行了通读和理解，并对其进行了简单的改造，也添加了一些可视化的代码

## Overall introduction

```
AE_Datasets:自编码器的三种数据预处理方式的相关代码
CNN_Datasets:卷积神经网络的三种数据预处理方式的相关代码
checkpoint:存放的是不同网络训练过程的日志（这里存放的是我当时跑训练模型时的日志信息）
logs：存放的是训练集和验证集训练过程的各种指标（准确率、精确率、召回率、误报率、漏检率、F1值、Loss值）的数据（使用tensorboard就可以可视化）
models：放置各种不同的网络模型的代码
utils：包含训练过程的一些函数
draw_models.py:对各个模型的训练集和验证集的ACC和LOSS进行绘图可视化的代码
draw_transform.py:对CWRU数据集的数据进行各种变换分析（CWT和STFT（汉宁窗）），并绘图进行可视化的代码
train.py:训练除自编码器的网络模型的代码
train_ae.py:训练自编码器的网络模型的代码
```

## Add

- 在train_utils.py和train_utils_ae.py的train函数中：增添了tensorboard可视化，增加了训练中的指标精确率、召回率、误报率、漏报率
- 增添了draw_models.py和draw_transform.py

## feelings

在故障检测的这个项目中，通过该代码的研读也对多种网络模型有了较为深刻的理解，也对pytorch的这套深度学习整体框架有了很多的了解，自己也对CWRU数据集进行了很多次的训练，也得出了对于适用于CWRU数据集故障检测的很棒的模型和数据输入。

## Citation

Codes:

```
@misc{Zhao2020,
author = {Zhibin Zhao and Tianfu Li and Jingyao Wu and Chuang Sun and Shibin Wang and Ruqiang Yan and Xuefeng Chen},
title = {Deep Learning Algorithms for Rotating Machinery Intelligent Diagnosis},
year = {2020},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark}},
}
```

Paper:

```
@article{zhao2020deep,
  title={Deep Learning Algorithms for Rotating Machinery Intelligent Diagnosis: An Open Source Benchmark Study},
  author={Zhibin Zhao and Tianfu Li and Jingyao Wu and Chuang Sun and Shibin Wang and Ruqiang Yan and Xuefeng Chen},
  journal={ISA Transactions},
  year={2020}
}
```