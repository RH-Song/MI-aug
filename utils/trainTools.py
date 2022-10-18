import time
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix,roc_curve,auc
from sklearn.preprocessing import label_binarize
import numpy as np
import warnings
from numpy import interp
from itertools import cycle
warnings.filterwarnings("ignore")


# 计算准确率函数
def evaluate_accuracy(data_iter, loss, net, flag=False, device=None): #增加一个标志判断位
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n, acc_train_sum = 0.0, 0, 0.0

    with torch.no_grad():
        test_l_sum, batch_count = 0.0, 0

        # 准确值与预测值
        Y_True = []
        Y_Predict = []

        #将准确值与预测值二值化
        True_Binary = []
        Pre_Binaty = []

        start_time = time.time()

        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                X = X.to(device)
                y_predict = net(X).argmax(dim=1)
                acc_sum += (y_predict == y.to(device)).float().sum().cpu().item()
                l2 = loss(net(X.to(device)), y.to(device))  # 算测试集loss
                batch_count += 1

                # 把正确值和预测值添加到列表
                Y_True += y.tolist()
                Y_Predict += y_predict.cpu().tolist()


                net.train()  # 改回训练模式
            else:  # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    y_predict = net(X, is_training=False).argmax(dim=1)
                    acc_sum += (y_predict == y).float().sum().item()
                else:
                    y_predict = net(X).argmax(dim=1)
                    acc_sum += (y_predict == y).float().sum().item()

            n += y.shape[0]
            test_l_sum += l2.cpu().item()
        test_l = test_l_sum / batch_count
        end_time = time.time()
        infer_time = end_time - start_time

        # 计算precision,recall,F1-score,根据计算原理不同分为macro,micro和weighted
        P_macro = precision_score(Y_True, Y_Predict, average="macro")
        R_macro = recall_score(Y_True, Y_Predict, average="macro")
        F1_macro = f1_score(Y_True, Y_Predict, average="macro")

        P_micro = precision_score(Y_True, Y_Predict, average="micro")
        R_micro = recall_score(Y_True, Y_Predict, average="micro")
        F1_micro = f1_score(Y_True, Y_Predict, average="micro")

        P_weighted = precision_score(Y_True, Y_Predict, average="weighted")
        R_weighted = recall_score(Y_True, Y_Predict, average="weighted")
        F1_weighted = f1_score(Y_True, Y_Predict, average="weighted")

        # 每个类的precision,recall,F1-score
        class_report = classification_report(Y_True, Y_Predict, target_names=['LAD', 'LCX', 'RCA', 'normal', 'other'],
                                             output_dict=True)

        # 混淆矩阵
        conf_matrix = confusion_matrix(Y_True, Y_Predict)

        #判度是否为最后一个epoch
        if flag:
            #标签二值化
            True_Binary = label_binarize(Y_True,classes=[i for i in range(5)]) # 注意i是类别数，根据需要修改
            Pre_Binaty = label_binarize(Y_Predict,classes=[i for i in range(5)])
            '''
            example:               [[1 0 0 0 0]
                                    [0 1 0 0 0]
                [0,1,2,3,4]   -->   [0 0 1 0 0]
                                    [0 0 0 1 0]
                                    [0 0 0 0 1]]
            '''

            return acc_sum / n, test_l, infer_time, [[P_micro, R_micro, F1_micro], [P_macro, R_macro, F1_macro],
                                                     [P_weighted, R_weighted, F1_weighted], class_report, conf_matrix,[True_Binary,Pre_Binaty]]
        else:
            return acc_sum / n, test_l, infer_time, [[P_micro, R_micro, F1_micro], [P_macro, R_macro, F1_macro],
                                                     [P_weighted, R_weighted, F1_weighted], class_report, conf_matrix]


def print_metrics(metric):
    # 打印各项指标,根据需要进行调整
    print('micro avg: precision %.4f, recall %.4f, f1-score %.4f\n' % (metric[0][0], metric[0][1], metric[0][2]))
    print('macro avg: precision %.4f, recall %.4f, f1-score %.4f\n' % (metric[1][0], metric[1][1], metric[1][2]))
    print('weighted avg: precision %.4f, recall %.4f, f1-score %.4f\n' % (metric[2][0], metric[2][1], metric[2][2]))

    print("LAD: precision %.4f, recall %.4f, f1-score %.4f, support %d\n" % (
        metric[3]["LAD"]["precision"], metric[3]["LAD"]["recall"], metric[3]["LAD"]["f1-score"],metric[3]["LAD"]["support"]))
    print("LCX: precision %.4f, recall %.4f, f1-score %.4f, support %d\n" % (
        metric[3]["LCX"]["precision"], metric[3]["LCX"]["recall"], metric[3]["LCX"]["f1-score"],metric[3]["LCX"]["support"]))
    print("RCA: precision %.4f, recall %.4f, f1-score %.4f, support %d\n" % (
        metric[3]["RCA"]["precision"], metric[3]["RCA"]["recall"], metric[3]["RCA"]["f1-score"],metric[3]["RCA"]["support"]))
    print("normal: precision %.4f, recall %.4f, f1-score %.4f, support %d\n" % (
        metric[3]["normal"]["precision"], metric[3]["normal"]["recall"], metric[3]["normal"]["f1-score"],metric[3]["normal"]["support"]))
    print("other: precision %.4f, recall %.4f, f1-score %.4f, support %d\n" % (
        metric[3]["other"]["precision"], metric[3]["other"]["recall"], metric[3]["other"]["f1-score"],metric[3]["other"]["support"]))

    # 这两个与上面的macro和weighted是一样的
    # print("macro avg: precision %.4f, recall %.4f, f1-score %.4f\n"%(metric[3]["macro avg"]["precision"],metric[3]["macro avg"]["recall"],metric[3]["macro avg"]["f1-score"]))
    # print("weighted avg: precision %.4f, recall %.4f, f1-score %.4f\n"%(metric[3]["weighted avg"]["precision"],metric[3]["weighted avg"]["recall"],metric[3]["weighted avg"]["f1-score"]))

    print("confusion_matrix:\n{}".format(metric[4]))


def train_ch5(net, train_iter, test_iter, batch_size, lr, wd, device, num_epochs, saveName):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    loss_list = []
    test_l_list = []
    train_acc_list = []
    test_acc_list = []
    iteration = 0
    lr_decade = [20, 30, 35]
    lr_decade_index = 0
    time_needed = 9999
    for epoch in range(num_epochs):
        if (epoch % lr_decade[lr_decade_index] == 0) and (epoch != 0):
            lr = lr / 10
            if lr_decade_index < len(lr_decade) - 1:
                lr_decade_index += 1
        print('lr %.6f' % lr)
        ###换了Adam算法
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        last_start = start
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l = l.cpu().item()
            train_l_sum += train_l
            train_acc = (y_hat.argmax(dim=1) == y).sum().cpu().item()
            train_acc_sum += train_acc
            n += y.shape[0]
            batch_count += 1
            iteration += 1
            print('epoch %d iteration %d train_loss %.4f' % (epoch + 1, iteration, train_l))
            if iteration % 50 == 0:
                test_acc, test_loss, infer_time, metric = evaluate_accuracy(test_iter, loss, net)
                print('epoch %d iteration %d train_loss %.4f test_loss %.4f train_acc %.4f test_acc %.4f time %.1f sec'
                      % (epoch + 1, iteration, train_l, test_loss, train_acc / y.shape[0], test_acc,
                         infer_time))
                print_metrics(metric)
        time_needed = time.time() - last_start
        last_start = time.time()
        total_need_time = time_needed * (num_epochs - epoch - 1)
        num_of_hour = total_need_time // 3600
        num_of_min = total_need_time % 3600 // 60
        num_of_sec = total_need_time % 3600 % 60 // 1
        test_acc, test_loss, train_loss, metric = evaluate_accuracy(test_iter, loss, net,flag=(epoch==num_epochs-1))
        print('epoch %d train_loss %.6f test_loss %.6f train_acc %.4f test_acc %.4f time %d hours %d mins %d secs'
              % (epoch + 1, train_l_sum / batch_count, test_loss, train_acc_sum / n, test_acc,
                 num_of_hour, num_of_min, num_of_sec))
        print_metrics(metric)

        loss_list.append(train_loss)
        test_l_list.append(test_loss)
        train_acc_list.append(train_acc_sum / n)
        test_acc_list.append(test_acc)

    x1 = range(0, num_epochs)
    plt.subplot(2, 1, 1)
    plt.plot(x1, loss_list, label='train')
    plt.plot(x1, test_l_list, label='test')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(["train_loss", "test_loss"])
    plt.title('loss vs. epochs')
    plt.subplot(2, 1, 2)
    plt.plot(x1, train_acc_list)
    plt.plot(x1, test_acc_list)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(["train_acc", "test_acc"])
    plt.title('accuracy vs. epochs')
    # plt.show()
    plt.savefig(saveName + "accuracy_loss.jpg")
    plt.close()

    # 绘制混淆矩阵
    # 热度图，后面是指定的颜色块，cmap可设置其他的不同颜色
    plt.imshow(metric[4], cmap=plt.cm.Greens)
    plt.colorbar()

    # 设置横纵坐标
    index = range(len(metric[4]))
    lables = ['LAD', 'LCX', 'RCA', 'normal', 'other']
    plt.xticks(index, lables)
    plt.yticks(index, lables)
    # plt.xlabel("预测值")
    # plt.ylabel("真实值")
    plt.xlabel("predict")
    plt.ylabel("true")

    # 解决汉字显示问题,linux缺少字体会报错，可以把xlabel和ylabel改为英文
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    # 显示数据
    for w in range(metric[4].shape[0]):
        for h in range(metric[4].shape[1]):
            plt.text(w, h, metric[4][w, h], verticalalignment='center', horizontalalignment='center')
    plt.savefig(f"{saveName}_heatmap_confusion_matrix.jpg")
    plt.close()

    #绘制ROC曲线
    fpr = dict() #假正率
    tpr = dict() #真正率
    roc_auc = dict() #AUC值

    #计算每个类别的ROC曲线
    for i in range(len(lables)):
        fpr[i],tpr[i],_ = roc_curve(metric[-1][0][:,i],metric[-1][1][:,i]) #取出二值化矩阵每一行的第i个数，形成列表
        roc_auc[i] = auc(fpr[i],tpr[i])

    #计算micro的ROC曲线
    fpr["micro"],tpr["micro"],_ = roc_curve(metric[-1][0].ravel(),metric[-1][1].ravel()) #把二值化矩阵展开，形成列表
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #计算macro的ROC曲线
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(lables))])) #拼接所有假正率，去除重复元素，并返从小到大返回
    mean_tpr = np.zeros_like(all_fpr) #复制与all_fpr长度相同值为零的列表
    for i in range(len(lables)):
        mean_tpr += interp(all_fpr,fpr[i],tpr[i]) # 线性插值函数：all_fpr代表要生成点的横坐标，fpr[i]代表原来区间的横坐标，tpr代表原来区间值得纵坐标
    mean_tpr /= len(lables) #取平均值
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
    label='micro-average ROC curve (area = {0:0.2f})'
    ''.format(roc_auc["micro"]),
    color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
    label='macro-average ROC curve (area = {0:0.2f})'
    ''.format(roc_auc["macro"]),
    color='navy', linestyle=':', linewidth=4)
    colors = cycle(['blue', 'palegreen', 'purple','darkviolet','orange'])
    for i, color in zip(range(len(lables)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
        label='ROC curve of class {0} (area = {1:0.2f})'
        ''.format(lables[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('multi-class')
    plt.legend(loc="lower right")
    plt.savefig(f"{saveName}_ROC_5分类.png")
    plt.close()



#参考链接：https://cloud.tencent.com/developer/article/1725618
