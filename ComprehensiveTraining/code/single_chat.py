"""
    create by Ian in 2018-9-26 17:57:19
    模仿QQ：
        单独聊天界面
"""
import sys
from PyQt5.QtWidgets import QApplication, QWidget
import PyQt5.uic
from PyQt5.QtCore import QObject, QUrl, pyqtProperty, QFileInfo, pyqtSignal
import re
from aip import AipImageClassify
import requests
from translate import Translation
from face_detection import Face_detection
from name2image import name_2_image
import random

ui_file = '../ui/single_chat.ui'
(class_ui, class_basic_class) = PyQt5.uic.loadUiType(ui_file)


def identify_objects(url):
    options = {}
    options["top_num"] = 5
    APP_ID = '14295936'
    API_KEY = '3Rmi7dHoVMufM0vRGHskXo2u'
    SECRET_KEY = 'SfmfRNCzj0z6Q7jlFneptmfOs7nGbFAf'
    classify = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)

    if "http" in url:
        reader = requests.get(url)
        result = classify.advancedGeneral(reader.content, options)
    else:
        def get_file_content(url):
            with open(url, 'rb') as fp:
                return fp.read()
        result = classify.advancedGeneral(get_file_content(url), options)

    if "error_msg" in result:
        return "对不起，无法识别"
    else:
        return result["result"][0]["keyword"]


class RobotChat(QWidget):
    """docstring for RobotChat"""
    Robot_Messages = pyqtSignal(str)  # 文本信息
    Robot_Image = pyqtSignal(str)  # 文本信息
    Robot_byte = pyqtSignal(bytes)  # byte信息

    def __init__(self):
        super(RobotChat, self).__init__()
        self._sensitive_value = 0 #敏感值
        self.face_detection = Face_detection()

    def get_messages(self, msg):
        """
            获取数据
        """
        print("获得的信息：", msg)
        self.process_message(msg)

    def process_message(self, msg):
        """
            解读信息
        """
        if msg == None:
            pass
        # 人脸检测
        elif self._sensitive_value == 1:
            msg = msg.replace("file:///", "")
            msg = msg.replace("<br>", "") 
            face_num, age, gender, left_upper, right_bottom = self.face_detection.upload(msg)
            face_image = self.face_detection.draw_rect(left_upper,right_bottom,msg)
            print("face_image:", face_image)
            self.send_image(face_image)
            self._sensitive_value = 0
        elif "检测" in msg:
            self._sensitive_value += 1
        # 翻译
        elif "翻译" in msg:
            msg = re.findall("(?<=翻译).*", msg)
            if msg[0] == "":
                self.send_message("遇到了一些BUG，需要修复")
            else:
                trans = Translation()
                trans_text = trans.translate(msg[0])[0]
                self.send_message(trans_text)
        # 看名找图
        elif "长什么" in msg:
            msg = re.findall(".*?(?=长什么)", msg)[0]
            print(msg)
            image = name_2_image(msg)
            temp = random.random()
            image += "?a="+ str(temp)
            self.send_image(image)

        # 识别物体
        elif "file:///" in msg:
            msg = msg.replace("file:///", "")
            object_name = identify_objects(msg)
            self.send_message(object_name)
        elif len(msg) >= 100:
            object_name = identify_objects(msg)
            self.send_message(object_name)
        else:
            self.send_message(msg)

    def send_message(self, msg):
        """
            机器人发送数据
        """
        self.Robot_Messages.emit(msg)

    def send_image(self, image_url):
        """
            机器人发送图片
        """
        self.Robot_Image.emit(image_url)


    def send_byte(self, bytes_mess):
        """
            识别物体
        """
        self.Robot_byte.emit(bytes_mess)


class Controller(object):
    """docstring for Controller
        核心控制器
    """

    def __init__(self):
        super(Controller, self).__init__()


class SingleChat(class_ui, class_basic_class):
    """docstring for SingleChat"""
    Signal_Messages = pyqtSignal(str)  # 文本信息
    Signal_Images = pyqtSignal(str)  # 图片信息

    def __init__(self):
        super(SingleChat, self).__init__()
        self.setupUi(self)  # 加载UI
        self.test_web_view()
        self.send_btn.clicked.connect(self.send_onclick)

    def test_web_view(self):
        url = QFileInfo("../web/index.html").absoluteFilePath()
        self.web_view.load(QUrl(url))

    def send_onclick(self):
        """
            发送数据
        """
        text = self.text_edit.toPlainText()
        text = re.sub("[\\n | \\r]", "<br>", text)
        print(text)
        if text == "":
            pass
        elif "http" in text:
            self.web_view.page().runJavaScript('sendImg("%s")' % text)
            self.Signal_Images.emit(text)
        elif "file://" in text:
            self.web_view.page().runJavaScript('sendImg("%s")' % text)
            self.Signal_Images.emit(text)

        # 文本信息
        elif "file:///" not in text:
            self.web_view.page().runJavaScript('sendMessage("%s")' % text)
            self.Signal_Messages.emit(text)

        # 图片信息
        else:
            print(text)
            raise "出现非法内容"

        self.text_edit.clear()  # 清空输入栏

    def robot_messages(self, msg):
        """
            机器人发送信息
        """
        self.web_view.page().runJavaScript('getMessage("%s")' % msg)

    def robot_image(self, image):
        """
            机器人发送图片
        """
        self.web_view.page().runJavaScript('getImg("%s")' % image)



def main():
    # 初始化
    app = QApplication(sys.argv)
    window = SingleChat()
    robot = RobotChat()

    # 连接准备
    window.Signal_Messages.connect(robot.get_messages)
    window.Signal_Images.connect(robot.get_messages)
    robot.Robot_Messages.connect(window.robot_messages)
    robot.Robot_Image.connect(window.robot_image)
    robot.Robot_byte.connect(window.robot_image)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
