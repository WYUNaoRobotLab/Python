from aip import AipFace
import base64
import cv2 as cv
import re
import base64
import os
import sys
class Face_detection():
    def __init__(self):
        self.APP_ID = "14295274"
        self.API_KEY = "TbR4I13iWePpbBtavdyB2F1H"
        self.SERECT_KEY = "WG1pwzsZNdFlCSTRN7hVCQxTPIgsXaze"
        self.client = AipFace(self.APP_ID,self.API_KEY,self.SERECT_KEY)
    def upload(self,img_file):
        with open(img_file,"rb") as f:
            original_img = f.read()
            image = base64.b64encode(original_img)
            image = str(image,"utf8")
            imageType = "BASE64"
            """ 有可选参数 """
            options = {}
            options["face_field"] = "age,gender"
            options["face_type"] = "LIVE"

            """ 带参数调用人脸检测 """
            information = self.client.detect(image, imageType, options)
            if information["error_code"]==0:
                result = information["result"]
                face_num = result["face_num"]
                location_information = result["face_list"][0]["location"]
                age = result["face_list"][0]["age"]
                gender = result["face_list"][0]["gender"]
                left_upper = (int(location_information["left"]),int(location_information["top"]))
                right_bottom = (left_upper[0]+int(location_information["width"]),left_upper[1]+int(location_information["height"]))
                return face_num, age,gender,left_upper,right_bottom

    def draw_rect(self,left_upper,right_bottom,img_file):
        print("draw_rect is uesd")
        #print(left_upper,right_bottom)
        #print(img_file)
        img = cv.imread(img_file)
        cv.rectangle(img,left_upper,right_bottom,(0,0,255),5)
        #cv.imshow("test",img)
        # cv.waitKey(5000)
        cv.imwrite("temp.jpg",img)
        #with open("temp.jpg","rb") as f:
        #    img = f.read()

        #img = base64.b64encode(img)
        #print(img)
        #s =  "data:image/jpg;base64,".encode("utf8")
        #s += img
        #print(type(s), s)
        path = (os.path.dirname(os.path.abspath(sys.argv[0])) + "\\" + "temp.jpg").replace("\\","/")
        return path



def re_clean(path):
    substring = re.findall("\w:[\\\w | \.\w+]+", path)
    if substring is not None:
        try:
            return substring[0]
        except Exception as e:
            raise e
    else:
        raise "没有成功清除字符"


if __name__ == "__main__":
    one = Face_detection()
    img_path = r"D:\Users\Yeah_Kun\Desktop\video\kk.JPG"
    img_path1 = re_clean(img_path)
    print(img_path)
    face_num, age, gender, left_upper, right_bottom = one.upload(img_path1)
    one.draw_rect(left_upper,right_bottom,img_path)

