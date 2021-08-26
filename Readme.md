# Readme

**用CVAE实现Image Caption的代码框架**

#### 文件结构

![image-20210801230404521](/Users/cckevin/typora/图片/image-20210801230404521.png)

#### 训练&测试

1. 将项目拷贝到服务器上，由于项目已经预处理完成，因此可直接进行训练
2. 修改config.py中的log_dir参数为自己存储的位置`'/home/username/checkpoints/CVAE_Caption/log/{}`
3. 在服务器上开始训练，例如`CUDA_VISIBLE_DEVICES=0 python train_cvae.py --id cvae_u0.5_k0.05 --unk_rate 0.5 --kl_rate 0.05`，其中`id`参数为模型的名称必须指定，其余参数可选，参见config.py
4. 训练过程中，可通过端口映射登录服务器，例如`ssh -L 16038:127.0.0.1:6038 chengkz@210.28.132.173`，将服务器上的6038映射到本地的16038，之后在服务器激活虚拟环境后启动tensorboard`tensorboard logdir=/home/chengkz/checkpoints/CVAE_Caption/log --port=6038  `，之后就可以在本地`http://localhost:16038`查看
5. 训练完成后，使用`CUDA_VISIBLE_DEVICES=0 python test_cvae.py --id cvae_u0.5_k0.05 --step 120000`进行测试，通过`id`和`step`参数指定要测试的模型名称和训练步数

#### 说明

由于时间原因，本次代码的结构和之前褚有刚学长给出的并没有完全统一，但大致结构是类似的；因此同学们目前可跳过其中的数据预处理等繁琐的部分，掌握核心部分（模型代码、训练和测试代码），学会使用该框架训练，并在其基础上进行修改增添；代码中重要部分基本带有注释。

下面是该框架中相比之前代码的一些可能变化：

* 数据预处理上，将原始karpathsplit给出的coco数据又按照自己想要的格式进行了处理；
* 为了加速训练，预先生成了coco中所有图片的resnet152特征，训练测试时直接加载该特征，也就意味着训练测试过程中不包含编码端
* 训练时使用tensorboard以便随时查看分析训练情况，tensorboard是一种重要的训练辅助工具，可参考https://pytorch.org/docs/master/tensorboard.html并查找资料学习使用
* 为了满足oracle evaluation等需要，手动改写了pycoco的部分源码，使得能够直接计算自己生成的一组句子和给出的一组ref情况下的指标，如果需要可在`/home/data_ti4_c/chengkz/anaconda3/envs/nlp_caption/lib/python3.7/site-packages/pycocoevalcap`中查看和修改
* 除了oracle evaluation外，还添加了一些衡量多样性的指标，详见`test_cvae.py`

