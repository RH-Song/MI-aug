import os

#目标路径
AIM_ROOT = '/home/lixiaohang/桌面/5_22_Data/origin'

#数据保存路径
TRAIN_ROOT = './origin_MI-set/origin_baseline_train'
TEST_ROOT = './origin_MI-set/origin_baseline_test'

#生成文件夹目录
if not os.path.exists(TEST_ROOT) and not os.path.exists(TRAIN_ROOT):
        os.makedirs(TEST_ROOT)
        os.makedirs(TRAIN_ROOT)

#读取文件并另存
for root,dirs,files in os.walk(AIM_ROOT):
    for f in files:

        #目录类别
        dir_flag = root.split("_")[-1]

        # 文件类别
        file_flag = f.split("_")[0]


        if dir_flag == "train":
            if file_flag == 'LAD' or file_flag == 'LCX' or file_flag == 'RCA':

                #读文件
                with open(root + "/" + f,"r") as r:
                    txt = r.read()

                #写文件
                with open(f'{TRAIN_ROOT}' + '/' + f,'w') as w:
                    w.write(txt)
            else:
                pass

        elif dir_flag == "test":
            if file_flag == 'LAD' or file_flag == 'LCX' or file_flag == 'RCA':

                with open(root + "/" + f,"r") as r:
                    txt = r.read()
                with open(f'{TEST_ROOT}' + '/' + f,'w') as w:
                    w.write(txt)
            else:
                pass

        else:
            print("ERROR!")
