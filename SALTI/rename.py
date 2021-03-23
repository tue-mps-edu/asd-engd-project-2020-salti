
import os
dir = r'C:\Github\asd-pdeng-project-2020-developer\test_images\labels night'
suffix = '_night'
import os

for root, dirs, files in os.walk(dir, topdown=False):
    for name in files:
        file, ext = os.path.splitext(name)
        os.rename(os.path.join(root,file+ext),os.path.join(root, file+suffix+ext))

