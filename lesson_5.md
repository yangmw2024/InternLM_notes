# 1.LMDeploy环境部署

## 1.1 创建开发机
`Cuda12.2-conda`；选择`10% A100*1`GPU
## 1.2 创建conda环境

### InternStudio开发机创建conda环境

```sh
studio-conda -t lmdeploy -o pytorch-2.1.2
```


### 本地环境创建conda环境


```sh
conda create -n lmdeploy -y python=3.10
```
## 1.3 安装LMDeploy

接下来，激活刚刚创建的虚拟环境。

```sh
conda activate lmdeploy
```

安装0.3.0版本的lmdeploy。

```sh
pip install lmdeploy[all]==0.3.0
```

# 2.LMDeploy模型对话(chat)
## 2.2 下载模型

本次实战营已经在开发机的共享目录中准备好了常用的预训练模型，可以运行如下命令查看： 

```sh
ls /root/share/new_models/Shanghai_AI_Laboratory/
```

### InternStudio开发机上下载模型（推荐）

如果你是在InternStudio开发机上，可以按照如下步骤快速下载模型。

首先进入一个你想要存放模型的目录，本教程统一放置在Home目录。执行如下指令：

```sh
cd ~
```

然后执行如下指令由开发机的共享目录**软链接**或**拷贝**模型： 

```sh
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b /root/
# cp -r /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b /root/
```

## 2.3 使用Transformer库运行模型

Transformer库是Huggingface社区推出的用于运行HF模型的官方库。

在2.2中，我们已经下载好了InternLM2-Chat-1.8B的HF模型。下面我们先用Transformer来直接运行InternLM2-Chat-1.8B模型，后面对比一下LMDeploy的使用感受。

在终端中输入如下指令，新建`pipeline_transformer.py`。

```sh
touch /root/pipeline_transformer.py
```
