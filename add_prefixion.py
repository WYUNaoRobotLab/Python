import os
class Filename():
    def __init__(self):
        pass
    def add_prefixion(self,parent_path):
        folder_list = os.listdir(parent_path)
        for element in folder_list:
            if not os.path.isfile(parent_path+"\\"+element):
                sub_dir = parent_path+"\\"+element
                jpg_list = os.listdir(sub_dir)
                for jpg_file in jpg_list:
                    os.rename(sub_dir+"\\"+jpg_file,sub_dir+"\\"+element+jpg_file)
            else:
                pass

if __name__ == "__main__":
    one = Filename()
    one.add_prefixion(r"D:\studyINF\AI\YOLOv3\yolo_img3")