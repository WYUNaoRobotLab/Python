import re
import glob
class changes_of_path():
    def  __init__(self):
        pass
    def change(self,all_xml_path,target,new_path):
        for xmlfile in all_xml_path:
            with open(xmlfile,"r") as f:
                content = f.read()
                new_content = content.replace(target,new_path)
            with open(xmlfile,"w") as f:
                f.write(new_content)

if __name__ =="__main__":
    one = changes_of_path()
    dirname = r"D:\studyINF\AI\YOLOv3\yolo_img3\additional_img_reflection_xml"
    all_xml_path = glob.glob(dirname+'\\'+'*.xml')
    one.change(all_xml_path,target=r"<path>D:\studyINF\AI\YOLOv3\yolo_img3\additional_img",new_path = r"<path>D:\studyINF\AI\YOLOv3\yolo_img3\additional_img_reflection")