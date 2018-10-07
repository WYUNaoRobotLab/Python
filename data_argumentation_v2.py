# encoding=utf-8
import cv2
import glob
import numpy as np
import os
class argumentation():
    def __init__(self):
        pass
    def for_all_jpg(self,path,reflect = True , rotate = True,reflection_path = None , rotation_path = None):
        
        all_jpg_path = glob.glob(path+"\\"+"*.jpg")
        # self.reflect_path = os.path.dirname(path)+"\\"+"reflection"
        # self.rotate_path = os.path.dirname(path) + "\\" + "rotation"
        self.reflect_path = os.path.dirname(path)
        self.rotate_path = os.path.dirname(path)
        for jpg_path in all_jpg_path:
            img = cv2.imread(jpg_path)
            self.argumentate(img,jpg_path,reflect,rotate,reflection_path, rotation_path)
    def argumentate(self,img,jpg_path,reflect , rotate ,reflection_path , rotation_path ):
        basename = os.path.basename(jpg_path)
        # reflection 镜像
        img_shape = img.shape
        if reflect:
            reflection_img = np.zeros(shape = img_shape,dtype=img.dtype)
            
            for row in range(img_shape[0]):
                reflection_img[row] = img[row,-1::-1]
            # 存放reflect 图片的路径
            if reflection_path is None:
                cv2.imwrite(self.reflect_path + "\\" + "reflection"+"\\"+ basename,reflection_img)
            else:
                cv2.imwrite(reflection_path +"\\"+ basename,reflection_img)
        
        # 顺时针旋转90度
        if rotate:
            rotation_img = np.zeros(shape = (img_shape[1],img_shape[0],3),dtype = img.dtype)
            height = img_shape[0]
            for row in range(height):
                rotation_img[:,height-1-row] = img[row]
            # 存放rotate 图片的路径
            if reflection_path is None:
                cv2.imwrite(self.reflect_path + "\\" + "rotation"+"\\"+ basename,rotation_img)
            else:
                cv2.imwrite(rotation_path +"\\"+ basename,rotation_img)
        




if __name__ == "__main__":
    one = argumentation()
    one.for_all_jpg(r"D:\studyINF\AI\YOLOv3\yolo_img3\additional_img",reflect = True, rotate = False, reflection_path = r"D:\studyINF\AI\YOLOv3\yolo_img3\additional_img_reflection")