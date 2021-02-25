import numpy as np
import random

class Network(object):
    """A MLP Network 一个多层感知器神经网络"""
    def __init__(self, sizes):                                                    #输入参数sizes是一个列表，其元素是各层神经元个数(第一个元素为输入层，最后一个元素为输出层，中间的元素均为隐藏层)
        self.num_layers = len(sizes)                                              #该神经网络层数为sizes列表元素个数
        self.sizes = sizes                                                        #np.random.randn(y, x) 可以生成y行x列的随机数矩阵 其是 均值为0,标准差为1的高斯分布(正态分布).
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]                   #定义偏置矩阵(或者说偏置列向量),该列向量维度(元素个数,或者说某层的偏置个数)应当与该层神经元个数相等.  [1:]切片,去除头元素 只遍历除输入层之外的神经元层,因为输入层直接给定(图像灰度矩阵)无权重
        self.weights = [np.random.randn(y,x)for x,y in zip(sizes[:-1],sizes[1:])] #定义权重矩阵,从第2层算起,(即不计数除输入层)第n层权重总数应当等于 n层神经元*(n-1)层神经元数 而该层权重矩阵M[行,列]=[n层神经元数, (n-1)层神经元数]  [1:]切片,去除头元素 只遍历除输入层之外的神经元层 定义"第n层";[:-1]切片,去除尾元素 只遍历除输出层之外的神经元层 定义"第n-1层"
                                                                                  #weights[0]=第0层权重矩阵(输入层和下一层之间的权重) biases[0]=第0层权重矩阵 因为 神经元层[0]= 输入神经元层(矩阵乘法)weights[0]+biases[0]  (神经元层[0]指的是输入层的下一层)
                                                                                  #可画图理解.                                                                     神经元层[n]= 神经元层[n-1](矩阵乘法)weights[n]+biases[n] (n从输入层的下一层开始计数)
    
                                                                                  
                                                                                  
                                                                                  

    
    def feedforward(self,a):
        """"神经网络前馈,a为输入层神经元向量,该方法给出其神经网络的输出结果(得到输出层) """
        for w,b in zip(self.weights,self.biases):
            a = Relu(np.dot(w,a)+b)      #每一层神经元的值都由该层权重矩阵乘上层神经元值加该层偏置得到
        return a


    def SGD(self,train_data,train_times,mini_batch_size,learn_speed,test_data=None):
        """Stochastic gradient descent 随机梯度下降,利用该方法使代价函数下降到极小值使神经网络学习
        train_data是一个(x,y)元组的列表,表示训练输入和正确的输出
        train_times 训练次数
        mini_batch_size 每个mini-batch的大小
        learn_speed学习速率,代价函数下降速度
        test_data是测试神经网络的测试数据,使用此参数可以使得每次训练之后对神经网络进行测试并在屏幕上显示出来,以便追踪进度(会大幅度拖慢训练速度)

        在每次训练中,先将训练数据打乱成多个小批量数据(mini_batch),然后对于每一个mini_batch进行一次速率为learn_speed的梯度下降(即调用train_mini_batch方法),这就是SDG的本质操作.

        """
        if test_data: #若有测试数据,则
            n_test = len(test_data)#存储测试数据长度
        n = len(train_data) #存储总训练集长度
        for j in range(train_times):  #进行train_times次的训练
            random.shuffle(train_data) #打乱训练数据集
            mini_batches = [train_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]  #将总训练数据集分为 n/mini_batch_size 个mini_batches
            for mini_batch in mini_batches: #对每个mini_batch进行训练
                self.train_mini_batch(mini_batch,learn_speed) #调用update_mini_batch方法进行训练
            if test_data: #若有测试数据,则在屏幕上打印训练进度
                print("Train times {0}: {1}/{2}(正确识别个数/训练总数)".format(j,self.evaluate(test_data),n_test))
            else:   #没有则最终显示训练完成
                print("Train times {0} 完成".format(j))


    def train_mini_batch(self,mini_batch,learn_speed):
        """对于每一个mini_batch调用反向传播进行一次速率为learn_speed的梯度下降,以此来更新神经网络的权重和偏置使之学习.
           mini_batch 是一个(x,y)元组的列表,表示被随机分组的训练输入和正确的输出
           learn_speed是学习速率
        """
        # nabla是梯度算符 以下两行组成了代价函数的梯度
        nabla_w = [np.zeros(w.shape) for w in self.weights] #根据每一层的权重矩阵形状创建 0矩阵(一个给定形状和类型的用0填充的数组) 用以存放代价函数梯度的权重分量
        nabla_b = [np.zeros(b.shape) for b in self.biases]  #根据每一层的偏置矩阵形状创建 0矩阵(一个给定形状和类型的用0填充的数组) 用以存放代价函数梯度的偏置分量
        for x,y in mini_batch: #遍历mini_batch内每个训练数据
            # delta_nabla_w,delta_nabla_b 表示mini_batch中 单个训练样本的代价函数梯度
            delta_nabla_w,delta_nabla_b = self.backprop(x,y) #将本次遍历mini_batch拿到的训练样本数据(mini_batch中的某个数字的输入数据和正确输出信息)输入backprop方法,得到返回的该数字的代价函数梯度的所有权重和偏置分量
            #nabla_w,nabla_b 表示整个mini_batch 所有训练样本的总代价函数梯度
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
        #更新整个神经网络的权重和偏置
        self.weights = [w - (learn_speed/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases =  [b - (learn_speed/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]

    def backprop(self,x,y):
        """反向传播算法具体实现 返回一个元组(nabla_w,nabla_b)代表代价函数梯度的权重和偏置分量 由输出层一层一层向前传播
           x为训练样本的输入层神经元激活值
           y为训练样本正确值输出值
        """
        nabla_w = [np.zeros(w.shape) for w in self.weights] #产生权重矩阵形状和偏置矩阵形状的零矩阵用于存放每层的梯度分量
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        #正向传播
        activation = x #存储输入激活值
        activations = [x] #存储每一层神经元激活值
        z_vector = [] #用于存储每一层神经元激活函数的输入 z = wa + b
        for w,b in zip(self.weights,self.biases):#遍历当前每一层的权重和偏置 即正向传播每层
            z = np.dot(w,activation) + b #计算每层神经元激活函数的输入
            z_vector.append(z) #将当前层的z记录下来以便之后计算偏差
            activation = Relu(z)#计算当前层激活值
            activations.append(activation) #存储当前层激活值
        #反向传播
        #偏差被定义为每一层的代价函数对该层神经元激活值的偏导数和神经元激活值对该层z_vector的偏导数的Hadamard 乘积(或者 Schur 乘积) (两形状相同矩阵对应元素乘积，即相同下标元素相乘)
        #定义偏差的目的是为了简化链式法则计算量，牺牲内存空间换取计算速度:因为 每一层代价函数对权重的偏导数和对偏置的偏导数实际上都含有相同的一部分 即 代价函数对神经元激活值的偏导数*神经元激活值对z_vector的偏导数  而每层 偏差值又和上层偏差值有联系 即 上层权重矩阵乘上层偏差值矩阵Hadamard乘积本层神经元激活值对z_vector的偏导数(矩阵乘法蕴涵了每个输出层神经元想怎样改变上层神经元激活值以降低代价函数,即上层神经元通过不同的途径(权重)来影响代价函数)
        #神经元激活值对z_vector的偏导数实际上就是神经元激活函数的导数
        delta = self.cost_derivative(activations[-1],y) * Relu_prime(z_vector[-1]) #得到输出层偏差  输出层偏差=输出层神经元激活值 Hadamard乘积 输出层神经元激活值对z_vector的偏导数
        nabla_w[-1] = np.dot(delta,activations[-2].transpose()) #利用 偏差 简化计算链式法则 得到 梯度的输出层权重分量 注意要对激活值矩阵进行转置(使用np.transpose()方法) numpy向量按列存储 不转置则无法进行矩阵乘法
        nabla_b[-1] = delta  #利用 偏差 简化计算链式法则 得到 梯度的输出层偏置分量
        for layer in range(2,self.num_layers):#从输出层开始向前进行反向传播,计算各层权重和偏置的偏导数
            z = z_vector[-layer]#该层神经元激活函数的输入
            Rp = Relu_prime(z)#该层神经元激活函数导数值
            delta = np.dot(self.weights[-layer+1].transpose(),delta) * Rp#得到该层偏差矩阵
            nabla_w[-layer] = np.dot(delta,activations[-layer-1].transpose())
            nabla_b[-layer] = delta
        return (nabla_w,nabla_b)#返回该次反向传播得到的梯度






    def evaluate(self,test_data):
        """用于评估测试神经网络是否能正确识别判断数字 返回正确判断的数字数量 """   #神经网络的输出结果是输出层激活值最大的一个神经元所对应的结果(即神经元内数字最大的一个神经元) 使用numpy的argmax方法来找到该输出层神经元的编号
        test_results = [(np.argmax(self.feedforward(x)),y) for (x,y) in test_data] #将test_data内所有数据作为测试数据输入神经网络进行测试判断,将得到的结果以 二元组(神经网络判断结果,正确结果) 的列表形式存储
        return sum(int(x == y) for (x,y) in test_results) #返回正确判断的数量 注意将bool型转换为整型再求和

    def cost_derivative(self,output_activations,y):
        """该方法返回代价函数对输出层神经元的偏导数
           output_activations 神经网络的输出层激活值
           y 输出层正确值
        """
        #使用 (output_activations - y) 而不用 0.5(output_activations - y) 减少计算量
        return (output_activations - y) 
      







                                                                                  
def Relu(z):     #神经元激活函数
    """Relu 线性整流函数"""
    #(z + np.abs(z)) / 2.0
    #return z
    return 1.0/(1.0+np.exp(-z))
    

def Relu_prime(z):
    """Relu 线性整流函数的导数 """
    
    #return np.where(z > 0, 1, 0)
    return Relu(z)*(1-Relu(z))


            


            

        



