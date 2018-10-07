from detection import Yolo
from tensorflow.python import pywrap_tensorflow
import numpy as np
class prediction():
    def __init__(self):
        self.one = Yolo(num_true_boxes=10, num_anchor_per_box=8,
                   img_width=416, img_height=416, training_able=None)
        self.reader = pywrap_tensorflow.NewCheckpointReader(r'D:\studyINF\AI\YOLOv3\test_samples_copy2\model.ckpt-80')
    def predict(self,sample,target):
        sample =sample.reshape(1,416,416,3)
        anchor_box = np.array([[19.07632094, 17.52739726],
                               [45.97297297, 15.77927928],
                               [27.28571429, 25.73086735],
                               [16.81603774, 45.10849057],
                               [53.87195122, 28.81097561],
                               [50.42105263, 83.23684211],
                               [92.86666667, 69.46666667],
                               [31.02752294, 50.16055046]], dtype=np.float32) / (
                         np.array([416.0, 416.0], dtype=np.float32).reshape(1, 2))
        fin_boxes, fin_scores, fin_classes, temp_value = self.one.prediction(self.reader, sample / 255.0, num_classes=3,
                                                                        anchor_box_tensor=anchor_box,
                                                                        score_threshold=0.4,iou_threshold=0.3)
        if len(fin_classes)>0:
            index_ = np.where(fin_classes == target)[0][0]
            return fin_boxes[index_].astype(np.int32)

if __name__=="__main__":
    import cv2
    img = cv2.imread(r"D:\studyINF\AI\YOLOv3\yolo_img3\temp2\1534479830.3.jpg")
    cv2.cvtColor(img,cv2.COLOR_BGR2RGB,img)
    one = prediction()
    box = one.predict(img,0)
    print(box)
