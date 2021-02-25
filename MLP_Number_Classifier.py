import mnist_loader
import class_Network
import cv2
import numpy as np
import codecs
import json
import os
import time

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

root_path = os.getcwd() + "\\Saves\\"
if not os.path.exists(root_path):
    os.makedirs(root_path)

print("建议的参数:[784, 30, 10]")
net = class_Network.Network(eval(input("(格式:[输入层,隐藏层,输出层])配置您的神经网络:"))) #[输入层,隐藏层,输出层]

while True:
    choose = input("您想要?(请输入选项数字):                     1-训练您的神经网络 \n \
                                            2-测试您的神经网络 \n \
                                            3-保存当前神经网络 \n \
                                            4-更改神经网络设置 \n \
                                            0-读取保存的神经网络 :")
    if (choose == "1"):
        print("1-训练您的神经网络")
        
        train_times = eval(input("请输入训练次数:"))
        
        print("建议的参数:10")
        mini_batch_size = eval(input("每个mini-batch的大小:"))
        
        print("建议的参数:3.0")
        learn_speed = eval(input("请设置学习速率:"))
        

        print("\n")

        print("训练次数:{}".format(train_times))
        print("mini-batch的大小:{}".format(mini_batch_size))
        print("学习速率:{}".format(learn_speed))
        

        print("\n")
        print("Loading....")
        print("少女祈祷中....")
        net.SGD(training_data, train_times, mini_batch_size, learn_speed, test_data=test_data)#训练数据 训练次数 minibatch大小 学习速率 测试数据

    elif (choose == "2"):
        print("测试您的神经网络")
        print("Train times {0}: {1}/{2}(正确识别个数/训练总数)".format(0,net.evaluate(test_data),10000))
        if(input("是否想查看测试图片(注意有10000张!)y/n?") == "y"):
            for test in test_data:
                image = np.reshape(test[0], (28, 28))
                num = np.argmax(net.feedforward(test[0]))
                cv2.imshow(str(num),image)
                print("结果:{} 正确结果:{}".format(num,test[1]))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    elif (choose == "3"):
        print("保存当前神经网络")

        os.chdir(root_path)
        
        
        save_weights = [weight.tolist() for weight in net.weights]
        save_biases = [bias.tolist() for bias in net.biases]
        



        save_time = time.strftime('%Y.%m.%d-%H.%M.%S',time.localtime(time.time()))
        save_dir = root_path+save_time+"_save\\"
        os.makedirs(save_dir)

        save_weights_path = save_dir + save_time+"_weights"+".json"
        save_biases_path = save_dir + save_time+"_biases"+".json"

        print("保存中......")
        print("少女祈祷中....")

        #print(save_weights)
        #print(save_biases)

        json.dump(save_weights, codecs.open(save_weights_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        json.dump(save_biases, codecs.open( save_biases_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

        print("保存成功")
    elif (choose == "0"):
        print("读取保存的神经网络")
        os.chdir(root_path)
        print("已保存的神经网络:")
        print(os.listdir(root_path))
        slect = input("请您输入选择的存档:")
        read_weights_path =  slect+ "\\" + slect[:-5] + "_weights.json"
        read_biases_path = slect+ "\\" + slect[:-5] + "_biases.json"

        print("读取中...")
        print("少女祈祷中......")

        read_weights = json.loads(codecs.open(read_weights_path, 'r', encoding='utf-8').read())
        read_biases = json.loads(codecs.open(read_biases_path, 'r', encoding='utf-8').read())

        net.weights = [np.array(weight) for weight in read_weights]
        net.biases = [np.array(bias) for bias in read_biases]

        #print(net.weights)
        #print(net.biases)

        print("读取成功")

    elif (choose == "4"):
        net = class_Network.Network(eval(input("(格式:[输入层,隐藏层,输出层])配置您的神经网络:"))) #[输入层,隐藏层,输出层]
    else:
        print("无效选项")


#net = class_Network.Network([784, 20, 10]) #[输入层,隐藏层,输出层]          


#net.SGD(training_data, 50, 10, 0.5, test_data=test_data)#训练数据 训练次数 minibatch大小 学习速率 测试数据



    

