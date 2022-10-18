import pandas as pd
import numpy as np
import torch


def crTable(net, test_iter, saveName, device=None):
    """"
    net : 尊立案好的神经网络
    X_test   :测试集样本
    y_test    ：测试集真实结果
    """
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device

    with torch.no_grad():
        net.eval()  # 评估模式, 这会关闭dropout
        for X_test,y_test in test_iter:
            X_out = net(X_test.to(device)).argmax(dim=1)
            y = y_test
            y = y.to(device)
            #统计数据
            #创建5*5table
            table = torch.zeros(5, 5, dtype=torch.long)
            for i in range(y.shape[0]):
                table[X_out[i]][y[i]] += 1
            # print(table)
            table = table.numpy()
            df = pd.DataFrame(table)
            df.to_csv(saveName +  '+'+ 'table'  +   '.csv',index=False)
            # 计算三个率
            allone = np.array([[1],[1],[1],[1],[1]],dtype='float32')
            A = table
            A = np.array(A,dtype='float32')
            P1 = ( A*np.identity( 5 )).dot(allone)/(A.dot(allone))
            R1 = (A*np.identity( 5 ) ).dot(allone)/( (A.transpose()).dot(allone))
            F1 = 2*(P1*R1)/(R1+P1)

            e = np.hstack((P1,R1,F1))
            df = pd.DataFrame(e)
            df.to_csv(saveName +    '+'+  'P1R1F1'  +   '.csv',index=False)
