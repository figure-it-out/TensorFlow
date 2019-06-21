1. 代码架构参考自：

   郑泽宇, 顾思宇. TensorFlow: 实战 Google 深度学习框架[J]. 2017.

   https://github.com/perhapszzy/tensorflow-tutorial

2. 在data目录下包含了mnist的训练集和测试集，这样在调用input_data.read_data_sets导入数据的时候就不会从MNIST数据集的官网上下载数据了，直接从本地文件夹"data/"读取，避免下载时间过长

3. 使用tf.summary相关函数对scalar以及其他tensor做相关记录，并使用tensorboard进行显示。

   每运行一次train.py会根据当前时间戳生成一个文件夹，这些文件夹共同的父文件夹为"log/"，在启动tensorboard的时候，tensorboard —logdir=“log/”将文件夹指定为"log/"，那么tensorboard会对每次train.py的运行结果都进行保存，并用不同的颜色显示在同一个plot中，方便对比

 
<img src="https://raw.githubusercontent.com/figure-it-out/TensorFlow/master/mnist/Res/2019-06-21 10.44.38.png" width=500 />
   **备注：**如果不想让tensorboard显示之前运行train.py的数据，那么需要提前清空log/文件夹，并且重新运行tensorboard
