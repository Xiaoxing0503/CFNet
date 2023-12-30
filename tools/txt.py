import os  #通过os模块调用系统命令

file_path = "D:\EPan\Xml\PycharmProjects\TarDAL-main\data\MSRS/vi"  #文件路径
path_list = os.listdir(file_path) #遍历整个文件夹下的文件name并返回一个列表

path_name = []#定义一个空列表

for i in path_list:
    path_name.append(i) #若带有后缀名，利用循环遍历path_list列表，split去掉后缀名

#path_name.sort() #排序

for file_name in path_name:
    # "a"表示以不覆盖的形式写入到文件中,当前文件夹如果没有"save.txt"会自动创建
    with open("D:\EPan\Xml\PycharmProjects\TarDAL-main\data\MSRS\meta/val.txt", "a") as file:
        file.write(file_name + "\n")
        print(file_name)
    file.close()



#####             4200个标签中挑选300个txt文件           ################
# file_path = "D:\EPan\Xml\PycharmProjects\TarDAL-main\data\M3FD_Fusion\labels"  #文件路径
# path_list = os.listdir(file_path)#遍历整个文件夹下的文件name并返回一个列表
#
# path_name = []#定义一个空列表
#
# for i in path_list:
#     path_name.append(i.split(".")[0])
#
# file=open('D:\EPan\Xml\PycharmProjects\TarDAL-main\data\M3FD_Fusion\meta/val.txt')
# dataMat=[]
# labelMat=[]
# for line in file.readlines():
#     curLine=line.strip().split(".")[0]
#     labelMat.append(curLine)
#
# for i in path_name:
#     if i in labelMat:
#         pass
#     else:
#         os.remove(file_path + '/' + str(i) + '.txt')