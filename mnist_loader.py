"""
mnist_loader   MNIST - Mixed National Institute of Standards and Technology database
               混合国家标准与技术研究所数据库
该程序导入MNIST数据集作为训练数据和验证数据

原始的NIST数据集的训练集是美国人口普查局的雇员的笔迹，而验证集是采集自美国中学生的笔迹。
所以从机器学习的角度看，这是一种不太合理的样本采集。
因此Lecun在进行卷积神经网络的研究时候从原始的NIST两个数据集中选择并重新混合形成新的训练集和验证集，也就是现在的MNIST。
MNIST的图片一般都是28*28的图片，每个像素值进行了归一化，使得其在0-1范围内，不过也有部分数据集把它们又恢复到0-255，每个数字图片都有对应的标记。
MNIST的训练集一共有60000个样本，而验证集有10000个样本。
--------------------- 

数据集背景介绍来源：CSDN 
原文：https://blog.csdn.net/nxcxl88/article/details/52098522 

"""
import pickle
import gzip
import numpy as np

def load_data():
    """
    将mnist数据集作为包含训练数据、验证数据、测试数据的元组返回
    training_data 以包含2个对象的元组返回,第一个是训练用的手写数字图像(以numpy ndarray类存储的50000个对象,每一个对象有784个值 分辨率28*28=784像素)
    第二个同样是以numpy ndarray类存储的50000个对象,他们是第一个对象所对应的正确数字标识(0,1,2,3,4,5,6,7,8,9)

    validation_data  和 test_data 与 training_data 类似 不过各自只包含10000个对象
    """
    f = gzip.open('mnist.pkl.gz', 'rb') #解压
    training_data, validation_data, test_data = pickle.load(f,encoding='bytes') #反序列化
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """
    返回包含训练数据、验证数据、测试数据的元组,基于load_data 但格式不同,更利于神经网络进行训练
    training_data 是一个 包含50000个对象的二元组数据集 其中x元素是784维numpy.ndarray向量 y是10维向量 (输入图片像素,正确数字)
    validation_data  和 test_data 与 training_data 类似 不过各自只包含10000个对象其中x元素是784维numpy.ndarray向量 y是单个数字


    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """
    返回10维向量 其中正确的数字分量为1,其他分量为0
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
