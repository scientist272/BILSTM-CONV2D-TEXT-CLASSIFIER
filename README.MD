# BILSTM-Conv2D 文本分类模型
用BILSTM和2D卷积神经网络构建的文本分类模型，使用前需要先下载glove.6b.100d.txt到/glove目录下
## How to train
参数配置在/config/config.json中。可调整batch_size,epochs,训练集测试集分割比例等。
如果要使用你自己的数据训练，修改data为你自己的数据路径，text_col为csv文件的文本列，label_col为文本对应的标签列。
## How to run
python classify.py 
需要先将要测试的文本写进/config/text.json中的文本数组里。
