import cv2
import glob
import re
class Change_imgsize():
    def __init__(self):
        pass
    def change(self,jpg_path):
        jpg = cv2.imread(jpg_path)
        jpg = cv2.resize(jpg,(416,416))
        return jpg
    def write(self,path):
        all_jpg_path = glob.glob(path+"\\"+"*.jpg")
        for jpg_path in all_jpg_path:
            jpg = self.change(jpg_path)
            cv2.imwrite(jpg_path,jpg)
    def draw_rectangle(self,jpg,left_upper,right_bottom):
        jpg = cv2.rectangle(jpg,left_upper,right_bottom,color=(255,0,0))
        return jpg
    def show(self,jpg):
        cv2.imshow("test",jpg)
        cv2.waitKey(0)


if __name__ == "__main__":
    # with open(r"C:\Users\lcq\Desktop\1533728661.32.xml","r") as f:
        # content = f.read()
    # xmin = re.findall("<xmin>(\d+)</xmin>",content)
    # xmax = re.findall("<xmax>(\d+)</xmax>",content)
    # ymin = re.findall("<ymin>(\d+)</ymin>",content)
    # ymax = re.findall("<ymax>(\d+)</ymax>",content)
    # jpg_path = re.findall("<path>(.+)</path>",content)[0]
    # for i in range(len(xmin)):
        # xmin[i] = int(xmin[i])
        # ymin[i] = int(ymin[i])
        # xmax[i] = int(xmax[i])
        # ymax[i] = int(ymax[i])

    # one = Change_imgsize()
    # jpg = one.change(jpg_path)
    # for i in range(len(xmin)):
        # jpg = one.draw_rectangle(jpg,(int(xmin[i]/640.0*256),int(ymin[i]/480.0*256)),(int(xmax[i]/640.0*256),int(ymax[i]/480.0*256)))

    # one.show(jpg)
    one = Change_imgsize()
    one.write(r"D:\studyINF\AI\YOLOv3\yolo_img3\background")
