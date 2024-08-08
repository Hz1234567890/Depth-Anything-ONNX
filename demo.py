import os

def print_images(folder_path):
    # 获取文件夹中的所有文件名
    files = os.listdir(folder_path)
    jpg_files = [file for file in files if file.endswith('.jpg')]

    # 按顺序重命名文件
    for index, file in enumerate(jpg_files):
        print(index,file)

# 指定文件夹路径
folder_path = '/media/hz/新加卷/0mywork/mine/UseGeo_1'

# 调用重命名函数
print_images(folder_path)
