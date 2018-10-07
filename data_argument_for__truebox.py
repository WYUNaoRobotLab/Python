import glob
import os.path
import re
class Xml_file():
    def __init__(self):
        pass
    def write_xml(self,source_path,destination_dir,img_width,img_height):
        
        all_xml_path = glob.glob(source_path+"\\"+"*.xml")
        for xml_path in all_xml_path:
            with open(xml_path,"r") as f:
                content = f.read()
            content = self.replace_coordinate(content,img_width,img_height)
            content = content.replace(">\t",">\n\t")
            content = content.replace("> ",">\n ")
            with open(destination_dir+"\\"+os.path.basename(xml_path),"w") as f:
                f.write(content)
    def replace_coordinate(self,content,img_width,img_height):
        content = content.replace("\n","")
        bndboxes = re.findall("<bndbox>(.*?)</bndbox>",content)
        for bndbox in bndboxes:
            temp_bndbox = bndbox
            xmax = re.findall("<xmax>(\d*)</xmax>",bndbox)[0]
            xmin = re.findall("<xmin>(\d*)</xmin>",bndbox)[0]
            # 替换xmin,

            new_xmin = str(img_width - int(xmax))
            bndbox = bndbox.replace("".join(["<xmin>",xmin,"</xmin>"]),"".join(["<xmin>",new_xmin,"</xmin>"]))

            # 替换xmax,

            new_xmax = str(img_width - int(xmin))
            bndbox = bndbox.replace("".join(["<xmax>",xmax,"</xmax>"]),"".join(["<xmax>",new_xmax,"</xmax>"]))
            content = content.replace(temp_bndbox,bndbox)
        return content

if __name__ == "__main__":
    one = Xml_file()
    one.write_xml(r"D:\studyINF\AI\YOLOv3\yolo_img3\additional_xml",r"D:\studyINF\AI\YOLOv3\yolo_img3\additional_img_reflection_xml",416,416)
