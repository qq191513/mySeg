import os
def finddir(startdir, target):# 查找target关键字的文件是否在startdir目录之下，在则返回完整路径
    for new_dir in os.listdir(startdir):  # 列表出该目录下的所有文件(返回当前目录'.')
        # print(new_dir)
        if target in new_dir:
            print(" 找到啦！！！！！！！！！")
            dir_path = os.path.join(startdir,new_dir)
            return dir_path
    print(target + "不存在!")
    return None
