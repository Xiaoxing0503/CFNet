import os


class BatchRename():

    def __init__(self):
        self.path = 'D:\EPan\Xml\Experiment\MSRSname\PIAFusion'  # 图片的路径

    def rename(self):
        filelist = os.listdir(self.path)
        filelist.sort()
        total_num = len(filelist)  # 获取文件中有多少图片

        file = open('D:\EPan\Xml\PycharmProjects\PSFusion-SegFormer/tools\data\VOCdevkit\VOC2012\ImageSets\Segmentation/val.txt')

        for i,line in zip(range(total_num),file.readlines()):
            i+=1
            src = os.path.join(self.path, str(i) + '.png')
            dst = os.path.join(os.path.abspath(self.path), line.replace('\n','') + '.png')

            os.rename(src, dst)

        print('total %d to rename & converted %d jpgs' % (total_num, i))


if __name__ == '__main__':
    demo = BatchRename()  # 创建对象
    demo.rename()  # 调用对象的方法