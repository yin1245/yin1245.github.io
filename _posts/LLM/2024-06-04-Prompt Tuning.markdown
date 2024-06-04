---
title: "Prompt Tuning"
layout: post
date: 2024-06-04 23:43
image: /assets/images/markdown.jpg
headerImage: ture
tag:
- markdown
- components
- extra
hidden: false
category: LLM
author: Zhenshuai Yin
description: 记录自己对于Prompt Tuning的理解
---

参考:[Prompt-Tuning——深度解读一种新的微调范式_prompt tuning-CSDN博客](https://blog.csdn.net/qq_36426650/article/details/120607050)

LLM的发展历程:

![在这里插入图片描述](https://tuchuang-yzs.oss-cn-beijing.aliyuncs.com/1614c2f147bf4ed4a4cee9af5c751a46.png)

面向LLM的Prompt-Tuning的发展历程:

![在这里插入图片描述](https://tuchuang-yzs.oss-cn-beijing.aliyuncs.com/9022caf1c4a74c75942539454f1b689f.png)

自从GPT等LLM相继提出, 以Pre-training+Fine-Tuning的模式在诸多NLP任务中被广泛应用: 首先在pretraining阶段通过在大规模无监督语料上的训练得到一个预训练语言模型(Pre-trained Language Model, PLM, 同样也是GPT中的P,即pretrained), 然后再Fine-tuning阶段基于训练好的语言模型在具体的下游任务中进行微调, 以获得适应下游任务的模型.

存在的问题: 在大多数的微调中, 下游任务的目标与预训练目标差距过大导致微调效果不明显;

​					微调过程中依赖大量的监督语料(即问答对数据集)

因此: 以GPT-3, PET为首提出了一种基于预训练语言模型的新的微调范式: Prompt-Tuning, 其旨在通过添加模版的方式来避免引入额外的参数, 从而让语言模型可以在小样本(Few-shot)或零样本(Zero-shot)场景下达到理想效果

Prompt-Tuning又称Prompt, Prompting, Prompt-based Fine-tuning等

Prompt Tuning的动机: 

​	降低语义差异(预训练和微调阶段)

​	避免过拟合: 由于在Fine-tuning阶段需要引入额外参数, 并且样本数量有限, 因此容易发生过拟合, 降低模型的泛化能力.

# 预训练语言模型

以GPT,ELMO, BERT为首的预训练语言模型在近两年大放异彩, 主要分为两种类型:

​	单向: 以GPT为首, 强调从左向右的编码顺序, 适用于Encoder-Decoder模式的自回归, 不会让后面的词影响到前面的词, 在Transformer的Attention部分会将后面词对前面词的影响使用掩码遮盖住

​	双向: 以ELMO为首, 强调双向编码, 但ELMO的主题是LSTM, 以串行进行编码, 无法并行, 因此最近以BERT(以Transformer为主体结构)作为双向语言模型的基准

## Transformer

Transformer模型是由谷歌的机器翻译团队在2017年末提出的(没错, 就是翻译团队), 是一种完全利用Attention机制构件的端到端模型.

Transformer的优势在于, 其推理完全由矩阵乘法实现, 实现了并行运算, 可以加速推理过程, 并且能够很好的体现出上下文对词义的影响(这个可能别的模型也可以)

> 在NLP领域中, Attention机制的目标是对具有强相关的token提高模型的关注度. 例如在文本分类中, 部分词对分类产生的贡献更大, 则会分配较大的权重(Transformer通过Q,K,V三个矩阵实现了Attention,并且可以兼顾较长的上下文).
>
> 对句子的编码目的是为了让模型记住token的语义(即将一个单词映射到高维向量, 并且考虑了位置等信息). 传统的LSTM只能通过长短期记忆的方法来捕捉token之间的关系, 容易导致梯度消失或记忆模糊问题, 而Transformer中, 任意两两token之间都有显式的连接, 避免了长距离依赖性问题.

## 经典的Pre-training方法

### Masked Language Modeling(MLM)

以word2vec, GloVe为代表的词向量模型, 主要以词袋(N-Gram)为基础. 如在word2vec的CBOW方法中, 随机选取一个固定长度的词袋区间, 然后挖掉中间部分的词, 让模型(简单深度神经网络)预测该位置的词

![img](https://tuchuang-yzs.oss-cn-beijing.aliyuncs.com/d2f37671b6cc4c6fb4f88021e41d1a75.png)

MLM采用了N-Gram的方法, 但N-Gram喂入的是被截断后的文本, 而MLM是完整的文本, 因此MLM更能保留原始的语义

![img](https://tuchuang-yzs.oss-cn-beijing.aliyuncs.com/5d2a6b9a27ef4176a658a1b7a0e0d8a0.png)

MLM是一种自监督的训练方法, 其先从大规模的无监督语料上通过固定的替换策略获得自监督语料, 设计预训练目标来训练模型, 流程如下:

​	替换策略: 在所有语料中, 随机抽取15%的文本. 被选中的文本中, 有80%随机挑选一个token并替换为[mask], 10%随机挑选一个token替换为其他token, 10%保持不变

​	训练目标: 当模型遇到[mask] token时, 根据学习得到的上下文语义去预测该位置可能的词.

​					因此, 训练的目标是对整个词表上的分类任务, 可以使用交叉信息熵作为目标函数

现有诸多针对MLM的改进版本, 作者挑选了两个经典的改进进行介绍:

​	Whole World Masking(WWM): BERT的MLM基于word piece进行随机替换, 而WWM表示被mask的必须是一个完整的单词

​	Entity Mention Replacement(EMR): 通常是在知识增强的预训练场景中, 对文本中的整个实体进行mask,而不是单一的token或字符

### Next Sentence Prediction(NSP)

在BERT原文中, 添加了NSP任务, 主要目标为给定两个句子, 判断他们之间的关系, 属于一种自然语言推理(NLI)任务, 在NSP中存在三种关系:

​	entailment: 蕴含关系, 认为位置相邻的两个句子属于entailment

​	contradiction: 矛盾关系, 认为两个句子不存在任何前后关系

​	Neutral: 中性关系, 认为当前两个句子可能来自同一篇文章, 但不属于isNext(entailment)关系

![image-20240605010243206](https://tuchuang-yzs.oss-cn-beijing.aliyuncs.com/image-20240605010243206.png)

通过这种方式, 来预测这两个句子属于上述三种关系的哪种关系(感觉确实没多大用)

在之后的模型中, 由于发现NSP对实验效果并没有太多正向效果, 因此均删除了NSP的任务























