# -*- coding: utf-8 -*-  
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python import pywrap_tensorflow
from tensorflow.python import debug as tfdbg
import cv2
from needed_function import random_mini_batches,readh5


class Yolo():
    def __init__(self,num_true_boxes,num_anchor_per_box,img_width,img_height,training_able):
        self.save_list = []
        self.obj_cost_scale = 6 
        self.obj_location_scale = 5
        self.num_true_boxes = num_true_boxes # 每张图片的true_boxes 的数量的最大值
        self.num_anchor_per_box = num_anchor_per_box
        self.img_width = img_width
        self.img_height = img_height
        self.training_able = training_able
    def my_conv(self, name, shape, input_data, strides, padding, training_able=True, init_filter=None, init_gama = None,init_beta = None,skip_item =None):
        """

        :param name: name of filter
        :param shape: shape of filter
        :param input_data: data will be convlutioned
        :param strides: strides of convlution
        :param padding: "valid" or "SAME"
        :return leaky_conv:      result of convlutional layer
        """
        if init_filter is None:
            filter_ = tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=training_able)
        else:
            filter_ = tf.get_variable(name=name, dtype=tf.float32, initializer=init_filter, trainable=training_able)
        conv_ = tf.nn.conv2d(input_data, filter_, strides=strides, padding=padding)
        mean, variance =tf.nn.moments(conv_,axes=[0,1,2],shift=None,name=None,keep_dims=False)
        if init_gama is None:
            gama = tf.get_variable(name = "gama" + name.split("_")[-1],initializer=tf.ones(shape = variance.shape))
        else:
            gama = tf.get_variable(name = "gama" + name.split("_")[-1],initializer=init_gama,trainable=training_able)
        if init_beta is None:
            beta = tf.get_variable(name = "beta" + name.split("_")[-1],initializer=tf.zeros(shape=mean.shape))
        else:
            beta = tf.get_variable(name = "beta" + name.split("_")[-1],initializer=init_beta,trainable=training_able)
        if skip_item is not None:
            z_ =tf.nn.batch_normalization(variance_epsilon=1e-6,x =conv_+skip_item,mean=mean,variance=variance,offset=beta,scale=gama)
        else:
            z_ =tf.nn.batch_normalization(variance_epsilon=1e-6,x =conv_,mean=mean,variance=variance,offset=beta,scale=gama)
        leaky_conv = tf.nn.leaky_relu(z_, 0.3)
        self.save_list.append(filter_)
        self.save_list.append(beta)
        self.save_list.append(gama)
        return leaky_conv
    
    def my_conv_no_bn(self, name, shape, input_data, strides, padding, training_able=True, init_filter=None,skip_item =None):
        """

        :param name: name of filter
        :param shape: shape of filter
        :param input_data: data will be convlutioned
        :param strides: strides of convlution
        :param padding: "valid" or "SAME"
        :return leaky_conv:      result of convlutional layer
        """
        if init_filter is None:
            filter_ = tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=training_able)
        else:
            filter_ = tf.get_variable(name=name, dtype=tf.float32, initializer=init_filter, trainable=training_able)
        conv_ = tf.nn.conv2d(input_data, filter_, strides=strides, padding=padding)
       
        
        if skip_item is not None:
            z_ = skip_item + conv_
        else:
            z_ = conv_
        leaky_conv = tf.nn.leaky_relu(z_, 0.1)
        self.save_list.append(filter_)
        
        return leaky_conv

    def train(self, training_data, training_label,training_treu_boxes, training_prior_boxes,classes, learning_rate, minibatch_size, num_epochs,anchor_box,reader):
        """

        :param training_data: shape(None,416,416,3)
        :param training_label: shape(None,13,13,6,8)
        :training_true_boxes:  shape(None,num_true_box,5) 5:true_x,true_y,true_w, true_h, which class
        :prior_boxes:          shape(None,13,13,6,4)
        :param classes:        scale ,3
        :param learning_rate:   
        :param minibatch_size:
        :param num_epochs:
        :param anchor_box_tensor:   (num_anchor_boxes_per_cell,2), 以32像素为单位长度
                                     
        :return:
        """
        input_img = tf.placeholder(dtype=tf.float32, shape=[None, self.img_height, self.img_width, 3], name="input_img")
        input_label = tf.placeholder(dtype=tf.float32, shape=[None, self.img_height/32, self.img_width/32, self.num_anchor_per_box, 5 + classes], name="input_label")
        true_boxes = tf.placeholder(dtype=tf.float32, shape=[None, self.num_true_boxes, 5], name="true_boxes")
        prior_boxes = tf.placeholder(dtype = tf.float32,shape =[None,self.img_height/32, self.img_width/32,self.num_anchor_per_box,4],name="prior_boxes")
        anchor_box_tensor =tf.constant(value = anchor_box)
        conv_9,conv_8 = self.backbone(input_img,reader,self.training_able)
        conv_15 = self.yolo_head(conv_9,conv_8,classes)
        box_xy, box_wh, box_confidence, class_probability = self.detection(conv_15,num_calsses = classes,anchor_box_tensor = anchor_box_tensor)
        iou_mask, detection_mask =self.preprocess_true_box(true_boxes,box_wh,box_xy,input_label,threshold_iou = 0.5)
        total_cost = self.calculate_cost(iou_mask,detection_mask,box_confidence,box_xy,box_wh,input_label,class_probability,prior_boxes)
        global_step = tf.Variable(0,trainable = False)
        decay_learning_rate = tf.train.exponential_decay(learning_rate, global_step, 40, 0.90, staircase=True)
        optimizer = tf.train.AdamOptimizer(decay_learning_rate).minimize(total_cost,global_step)
        saver = tf.train.Saver(
            self.save_list, max_to_keep=51)
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        with tf.Session(config=config) as sess:

            costs = []
            initzer = tf.global_variables_initializer()
            sess.run(initzer)
            num_minibatches = int(
                training_data.shape[
                    0] / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = 0

            for epoch in range(num_epochs):
                epoch_cost = 0
                seed += 1
                mini_batches = random_mini_batches(training_data, training_label, training_treu_boxes,training_prior_boxes, minibatch_size,
                                                   seed=seed)
                for batch in mini_batches:
                    _, minibatch_cost = sess.run([optimizer, total_cost],
                                                 feed_dict={input_img: batch[0] / 255, input_label: batch[1],
                                                            true_boxes: batch[2],prior_boxes:batch[3]})
                    epoch_cost += minibatch_cost
                epoch_cost /= num_minibatches
                if epoch % 20 == 0:
                    print(epoch_cost)
                    costs.append(epoch_cost)
                    save_path = saver.save(sess, r"D:\YOLOv3\save_dir9\model.ckpt",
                                           global_step=epoch)

            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()


    def backbone(self,input_img,reader,training_able):

        conv_1 = self.my_conv(name = "filter_1",shape = [3,3,3,32],input_data = input_img, strides = [1,1,1,1], 
                              padding ="SAME",training_able=self.training_able[0] ,init_filter = reader.get_tensor("filter_1"),
                             init_beta = reader.get_tensor("beta1"),init_gama = reader.get_tensor("gama1"))
        print("conv_1 shpae is :",conv_1.shape) # (None,416,416,32)
        conv_2 = self.my_conv(name = "filter_2",shape = [3,3,32,64],input_data = conv_1, strides = [1,2,2,1],
                              padding ="SAME",training_able=self.training_able[1],init_filter = reader.get_tensor("filter_2"),
                             init_beta = reader.get_tensor("beta2"),init_gama = reader.get_tensor("gama2"))
        print("conv_2 shape is:",conv_2.shape) # (None,208,208,64)
        conv_3 = self.my_conv(name="filter_3", shape=[1, 1, 64, 32], input_data=conv_2, strides=[1, 1, 1, 1],
                              padding="VALID",training_able=self.training_able[2],init_filter = reader.get_tensor("filter_3"),
                             init_beta = reader.get_tensor("beta3"),init_gama = reader.get_tensor("gama3"))
        print("conv_3 shape is:", conv_3.shape) # (None,208,208,32)
        conv_4 = self.my_conv(name="filter_4", shape=[3, 3, 32, 64], input_data=conv_3, strides=[1, 1, 1, 1],
                              padding="SAME",training_able=self.training_able[3],init_filter = reader.get_tensor("filter_4"),
                             init_beta = reader.get_tensor("beta4"),init_gama = reader.get_tensor("gama4"))
        print("conv_4 shape is:", conv_4.shape) # (None,208,208,64)
        resi_1 = conv_2 + conv_4 # 残差项
        conv_5 = self.my_conv(name="filter_5", shape=[3, 3, 64, 128], input_data=resi_1, strides=[1, 2, 2, 1],
                              padding="SAME",training_able=self.training_able[4],init_filter = reader.get_tensor("filter_5"),
                             init_beta = reader.get_tensor("beta5"),init_gama = reader.get_tensor("gama5"))
        print("conv_5 shape is:", conv_5.shape) # (None,104,104,128)
        conv_6 = self.my_conv(name = "filter_6", shape =[1,1,128,64], input_data=conv_5, strides=[1,1,1,1],
                              padding="SAME",training_able=self.training_able[5],init_filter = reader.get_tensor("filter_6"),
                             init_beta = reader.get_tensor("beta6"),init_gama = reader.get_tensor("gama6"))
        print("conv_6 shape is:",conv_6.shape) # (None,104,104,64)
        conv_7 = self.my_conv(name="filter_7", shape=[3, 3, 64, 128], input_data=conv_6, strides=[1, 1, 1, 1],
                              padding="SAME",training_able=self.training_able[6],init_filter = reader.get_tensor("filter_7"),
                             init_beta = reader.get_tensor("beta7"),init_gama = reader.get_tensor("gama7")) # (None,64,64,128)
        print("conv_7 shape is:", conv_7.shape) # (None,104,104,128)
        resi_2 = conv_7 +conv_5

        conv_8 =self.my_conv(name ="filter_8",shape=[3,3,128,64],input_data=resi_2,strides=[1,2,2,1],
                             padding="SAME",training_able=self.training_able[7],init_filter = reader.get_tensor("filter_8"),
                            init_beta = reader.get_tensor("beta8"),init_gama = reader.get_tensor("gama8"))
        print("conv_8 shape is:",conv_8.shape) # (None,52,52,64)
        conv_9 = self.my_conv(name="filter_9", shape=[3, 3, 64, 64], input_data=conv_8, strides=[1, 2, 2, 1],
                              padding="SAME",training_able=self.training_able[8],init_filter = reader.get_tensor("filter_9"),
                             init_beta = reader.get_tensor("beta9"),init_gama = reader.get_tensor("gama9"))
        print("conv_9 shape is:", conv_9.shape)  # (None,26,26,64)
        """
        conv_10 = self.my_conv(name="filter_10", shape=[3, 3, 16, 14], input_data=resi_2, strides=[1, 2, 2, 1],
                              padding="SAME")
        print("conv_10 shape is:", conv_10.shape)  # (None,8,8,14)
        """
        

        return conv_9 ,conv_8
    # darknet 完成，下面开始写 yolo_head
    def yolo_head(self,extract_result,skip_connection,classes):
        conv_10 = self.my_conv_no_bn(name="filter_10", shape=[1, 1, 64, 64], input_data=extract_result, strides=[1, 1, 1, 1],
                              padding="SAME")    # (None,26,26,64)
        print("conv_10 shape is:", conv_10.shape)
        conv_11 = self.my_conv_no_bn(name="filter_11",shape=[3,3,64,64],input_data=conv_10,strides=[1,1,1,1],
                               padding="SAME")   # (None,26,26,64)
        print("conv_11 shape is:", conv_11.shape)
        #conv_skip_con = self.my_conv(name = "skipfilter_01",shape = [1,1,64,64],input_data = skip_connection,strides=[1,1,1,1],
                                     #padding="SAME")
        #to_depth_skip_con = tf.space_to_depth(conv_skip_con,2) # (None,26,26,256)
        #reorg = tf.concat(axis = -1,values = [conv_11,to_depth_skip_con])    # (None,26,26,320)
        #print("reorg.shape is:",reorg.shape)
        conv_12 = self.my_conv_no_bn(name="filter_12",shape=[1,1,64,128],input_data=conv_11,strides=[1,1,1,1],
                               padding="SAME")  # (None,26,26,128)
        print("conv_12 shape is:", conv_12.shape)
        conv_13 = self.my_conv_no_bn(name="filter_13", shape=[3, 3, 128, 128], input_data=conv_12, strides=[1, 2, 2, 1],
                               padding="SAME")  # (None,13,13,128)
        print("conv_13 shape is:", conv_13.shape)
        conv_skip_con = self.my_conv_no_bn(name = "skipfilter_01",shape = [1,1,64,64],input_data = skip_connection,strides=[1,1,1,1],
                                     padding="SAME")
        to_depth_skip_con = tf.space_to_depth(conv_skip_con,4) # (None,13,13,1024)
        reorg = tf.concat(axis = -1,values = [conv_13,to_depth_skip_con])    # (None,13,13,1024+128)
        print("reorg.shape is:",reorg.shape)
        #conv_14 = self.my_conv(name = "filter_14",shape = [1,1,64,64],input_data = skip_connection,strides = [1,1,1,1],padding = "SAME")
        
        filter_15 = tf.get_variable(name="filter_15",shape = [1,1,1152,self.num_anchor_per_box*(classes + 5)],dtype = tf.float32,initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG',
                                                                                                 uniform=True,
                                                                                                 seed=None))
        self.save_list.append(filter_15)
        conv_15 = tf.nn.conv2d(reorg, filter_15, strides=[1, 1, 1, 1], padding="SAME")
        # (None,13,13,?)
        print("conv_15 shape is:", conv_15.shape)
        return conv_15

        # yolo_head 完成,开始detection ,得到输出
    def detection(self,feat,num_calsses,anchor_box_tensor):
        grid_dims = tf.stack([feat.shape[2],feat.shape[1]]) # width, height
        grid_width_index = tf.cast(tf.range(grid_dims[0]),tf.float32)
        grid_height_index = tf.cast(tf.range(grid_dims[1]),tf.float32)
        grid_width_index ,grid_height_index = tf.meshgrid(grid_width_index,grid_height_index)
        grid_width_index = tf.reshape(grid_width_index,[-1,1])
        grid_height_index = tf.reshape(grid_height_index,[-1,1])
        grid_wh_index = tf.concat(values =[grid_width_index,grid_height_index],axis=1) # shape (8*8,2)
        # 0,0
        # 1,0
        # ...
        # 7,0
        # 0,1
        # 1,1
        # ...
        # 7,7
        print(grid_wh_index.shape)
        grid_wh_index = tf.reshape(grid_wh_index,[1,grid_dims[1],grid_dims[0],1,2])
        
        
        feat = tf.reshape(feat,[-1,feat.shape[1],feat.shape[2],self.num_anchor_per_box,5+num_calsses]) # 8 是 anchor boxes 的数量
        box_xy = tf.sigmoid(feat[...,1:3])
        box_xy = (box_xy + grid_wh_index) /tf.cast(grid_dims,tf.float32)                              # bx = (sigmoid(tx) + cx)/grid_width
                                                                                         # by 同理
        anchor_box_tensor = tf.reshape(anchor_box_tensor,shape=[1,1,1,anchor_box_tensor.shape[0],2]) # anchor_box_tensor shape (1,1,1,num_anchor_per_grid,2)

        box_wh = tf.exp(feat[...,3:5])* anchor_box_tensor                                         # box_width = exp(tw)*anchor_width 这里之所以不除以grid_width，
                                                                                                     # 是因为anchor_box_tensor已经归一化到[0,1]区间
        box_confidence = tf.sigmoid(feat[...,0:1]) # 得到 confidence
        #box_confidence = tf.reshape(box_confidence,shape=[box_confidence.shape+(1,))  # shape(batch, grid, grid, num_anchor_per_box,1)
        print("box_confidence shape is",box_confidence.shape)
        class_probability = tf.sigmoid(feat[...,5:])
        return box_xy, box_wh, box_confidence, class_probability
        # detection 完成，得到 xy,wh,confidence, probability
    def preprocess_true_box(self,true_boxes, box_wh, box_xy, input_label, threshold_iou =0.5):
        # 参考过GitHub上的开源代码和吴恩达课程作业的代码，会发现有matching_true_box，matching_box, matching_class
        # 这些变量，这是为了计算cost做准备 (class cost 和 xy 坐标, wh大小的cost),但我并没有计算上述的这三个变量，
        # 因为这些变量要表达的信息是图像里的true_box 落在哪一个grid_cell，x,y,w,h 和类别，而这些信息
        # 在将anchor_label写入h5前，就包含在anchor_label里 了
        # preprocess true box 开始，主要是为得到掩码用于区分有object和没有object(noobject)

            # 计算 预测得到的box与 true_box(ground truth box)的iou
                # true_boxes shape ( batch, num_true_box,5);5 意味着(x,y,w,h,class) num_true_box的数值为各图片中true_boxes的数量的最大值
        true_boxes_shape = tf.shape(true_boxes)
        true_boxes = tf.reshape(true_boxes,[true_boxes_shape[0],1,1,1,true_boxes_shape[1],true_boxes_shape[2]])
        # shape is (batch,1,1,1,num_true_box,5)
        true_box_wh_half = true_boxes[...,2:4]/2
        true_left_upper = true_boxes[...,0:2] - true_box_wh_half

        true_right_bottom = true_boxes[..., 0:2] + true_box_wh_half

        box_wh = tf.expand_dims(box_wh,4) # shpae (batch, grid,grid,num_anchor_per_cell,1,2)
        box_xy = tf.expand_dims(box_xy, 4)
        pred_box_wh_half = box_wh/2
        pred_left_upper = box_xy - pred_box_wh_half

        pred_right_bottom = box_xy + pred_box_wh_half


        # 下面的这个比较，应该是先进行broadcasting,各自被广播成(batch,grid,grid,num_anchor_per_cell,num_true_box,2)
        # 广播的概念与numpy的相同
        inter_min = tf.maximum(true_left_upper,pred_left_upper)
        inter_max = tf.minimum(true_right_bottom,pred_right_bottom)
        inter_wh = tf.maximum(inter_max - inter_min,0)
        intersection_area = inter_wh[...,0] * inter_wh[...,1]
        # 求交集的代码也可写成下面那样
        """
        xmin = tf.maximum(true_left_upper[...,0],pred_left_upper[...,0])
        ymin = tf.maximum(true_left_upper[...,1],pred_left_upper[...,1])
        xmax = tf.minimum(true_right_bottom[...,0],pred_right_bottom[...,0])
        ymax = tf.minimum(true_right_bottom[...,1],pred_right_bottom[...,1])
        width_difference = tf.maximum(xmax - xmin,0)
        height_difference = tf.maximum(ymax - ymin,0)
        intersection_area = width_difference * height_difference
        """
        true_area = true_boxes[...,2] * true_boxes[...,3]
        pred_area = box_wh[...,0] * box_wh[...,1]
        union_area = true_area + pred_area - intersection_area
        iou = intersection_area / union_area        # shape (batch,grid,grid,num_anchor_per_cell,num_true_box)
        best_iou = tf.reduce_max(iou,axis = -1,keepdims = True) # reduce 意味着沿着axis轴减少一个维度,即减少axis轴那个维度（除非 keepdims is True）
                                                                # axis选为最后一个维度,意味着对每一个pred_box而言，从num_true_box中，选出一个与pred_box最接近的）
        iou_mask = tf.cast(best_iou>threshold_iou,dtype = tf.float32) # 用以判断anchor box 是否检测到object
                                                                      # shape is (batch,grid,grid,num_anchor_per_cell,1)

        # 下面要得到一个数据:detection_mask, 通过它可知每一个gridcell,对应哪一个anchor
        detection_mask = input_label[...,0:1] # 通过 confidence 获知，哪个anchor负责该gridcell，但这里并没有保证每一个gridcell最多只有一个anchor为1，
                                            # 要保证也可以，但还是应该通过获取图片，标注label时,检查每个gridcell的中心数是否大于1
                                            # shape is (batch,grid,grid,num_anchor_per_cell)
        #detection_mask = tf.reshape(detection_mask,shape= detection_mask.shape+(1,))  # shape is (batch,grid,grid,num_anchor_per_cell,1)
        print("detection_mask is:",detection_mask.shape)
        return iou_mask, detection_mask


    def calculate_cost(self, iou_mask, detection_mask, box_confidence,box_xy,box_wh,anchor_label, class_probability,prior_boxes,lambd = 0.1):
        no_obj_conf_cost = (1-iou_mask)*(1-detection_mask)*tf.square(box_confidence)
        obj_conf_cost = detection_mask*tf.square(1-box_confidence)*self.obj_cost_scale
        conf_cost = tf.reduce_sum(no_obj_conf_cost + obj_conf_cost) # confidence cost

        # probability cost
        prob_cost = tf.reduce_sum(detection_mask * tf.square(anchor_label[...,5:] - class_probability))

        # center cost
        
        xy_cost = self.obj_location_scale*tf.reduce_sum((4-1.5*anchor_label[...,3:4]*anchor_label[...,4:5])*detection_mask*tf.square(anchor_label[...,1:3] - box_xy))
        
        # height width cost 
        wh_cost = self.obj_location_scale*tf.reduce_sum((4-1.5*anchor_label[...,3:4]*anchor_label[...,4:5])*detection_mask*tf.square(anchor_label[...,3:5]-box_wh))
        
        # height width cost between prior and pred box
        #prior_pred_xy_cost = tf.reduce_sum(detection_mask*tf.square(prior_boxes[...,0:2] - box_xy))
        
        # center cost between prior and pred box
        #prior_pred_wh_cost = tf.reduce_sum(detection_mask*tf.square(prior_boxes[...,2:4] - box_wh))
        # total cost
        total_cost = conf_cost + prob_cost + xy_cost + wh_cost 
        return total_cost



    def retrain(self, training_data, training_anchor_label, training_treu_boxes,training_prior_boxes,classes, learning_rate, minibatch_size, num_epochs,anchor_box,reader,lambd=0.1):
        input_img = tf.placeholder(dtype=tf.float32, shape=[None, self.img_height, self.img_width, 3], name="input_img")
        input_label = tf.placeholder(dtype=tf.float32, shape=[None, self.img_height/32, self.img_width/32,self.num_anchor_per_box , 5 + classes], name="input_label")
        true_boxes = tf.placeholder(dtype=tf.float32, shape=[None, self.num_true_boxes, 5], name="true_boxes")
        prior_boxes = tf.placeholder(dtype = tf.float32,shape =[None,self.img_height/32, self.img_width/32,self.num_anchor_per_box,4],name="prior_boxes")
        anchor_box_tensor =tf.constant(value = anchor_box)
        conv_9,conv_8 = self.backbone_for_retrain(input_img,reader,self.training_able)
        conv_15 = self.yolo_head_for_retrain(conv_9,conv_8,reader)
        box_xy, box_wh, box_confidence, class_probability = self.detection(conv_15, num_calsses=classes,
                                                                           anchor_box_tensor=anchor_box_tensor)
        iou_mask, detection_mask = self.preprocess_true_box(true_boxes, box_wh, box_xy, input_label, threshold_iou=0.5)
        total_cost = self.calculate_cost(iou_mask, detection_mask, box_confidence, box_xy, box_wh, input_label,
                                         class_probability,prior_boxes,lambd)
        
        global_step = tf.Variable(0,trainable = False)
        decay_learning_rate = tf.train.exponential_decay(learning_rate, global_step, 40, 0.9, staircase=False)
        optimizer = tf.train.AdamOptimizer(decay_learning_rate).minimize(total_cost,global_step)
        saver = tf.train.Saver(
            self.save_list, max_to_keep=30)
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        
        
        with tf.Session(config=config) as sess:
           
            initzer = tf.global_variables_initializer()
            sess.run(initzer)
            costs = []
            
            num_minibatches = int(
                training_data.shape[
                    0] / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = 0

            for epoch in range(num_epochs):
                epoch_cost = 0
                seed += 1
                mini_batches = random_mini_batches(training_data, training_anchor_label, training_treu_boxes,training_prior_boxes, minibatch_size,
                                                   seed=seed)
                for batch in mini_batches:
                    _, minibatch_cost = sess.run([optimizer, total_cost],
                                                 feed_dict={input_img: batch[0] / 255, input_label: batch[1],
                                                            true_boxes: batch[2],prior_boxes:batch[3]})
                    epoch_cost += minibatch_cost
                epoch_cost /= num_minibatches
                if epoch % 10 == 0:
                    print(epoch_cost)
                    costs.append(epoch_cost)
                    save_path = saver.save(sess, r"D:\YOLOv3_2\test_samples_copy2\model.ckpt",
                                           global_step=epoch)
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
    def backbone_for_retrain(self,input_img,reader,training_able):
        
        conv_1 = self.my_conv(name = "filter_1",shape = [3,3,3,32],input_data = input_img, strides = [1,1,1,1], 
                              padding ="SAME",training_able=self.training_able[0] ,init_filter = reader.get_tensor("filter_1"),
                             init_beta = reader.get_tensor("beta1"),init_gama = reader.get_tensor("gama1"))
        print("conv_1 shpae is :",conv_1.shape) # (None,416,416,32)
        conv_2 = self.my_conv(name = "filter_2",shape = [3,3,32,64],input_data = conv_1, strides = [1,2,2,1],
                              padding ="SAME",training_able=self.training_able[1],init_filter = reader.get_tensor("filter_2"),
                             init_beta = reader.get_tensor("beta2"),init_gama = reader.get_tensor("gama2"))
        print("conv_2 shape is:",conv_2.shape) # (None,208,208,64)
        conv_3 = self.my_conv(name="filter_3", shape=[1, 1, 64, 32], input_data=conv_2, strides=[1, 1, 1, 1],
                              padding="VALID",training_able=self.training_able[2],init_filter = reader.get_tensor("filter_3"),
                             init_beta = reader.get_tensor("beta3"),init_gama = reader.get_tensor("gama3"))
        print("conv_3 shape is:", conv_3.shape) # (None,208,208,32)
        conv_4 = self.my_conv(name="filter_4", shape=[3, 3, 32, 64], input_data=conv_3, strides=[1, 1, 1, 1],
                              padding="SAME",training_able=self.training_able[3],init_filter = reader.get_tensor("filter_4"),
                             init_beta = reader.get_tensor("beta4"),init_gama = reader.get_tensor("gama4"))
        print("conv_4 shape is:", conv_4.shape) # (None,208,208,64)
        resi_1 = conv_2 + conv_4 # 残差项
        conv_5 = self.my_conv(name="filter_5", shape=[3, 3, 64, 128], input_data=resi_1, strides=[1, 2, 2, 1],
                              padding="SAME",training_able=self.training_able[4],init_filter = reader.get_tensor("filter_5"),
                             init_beta = reader.get_tensor("beta5"),init_gama = reader.get_tensor("gama5"))
        print("conv_5 shape is:", conv_5.shape) # (None,104,104,128)
        conv_6 = self.my_conv(name = "filter_6", shape =[1,1,128,64], input_data=conv_5, strides=[1,1,1,1],
                              padding="SAME",training_able=self.training_able[5],init_filter = reader.get_tensor("filter_6"),
                             init_beta = reader.get_tensor("beta6"),init_gama = reader.get_tensor("gama6"))
        print("conv_6 shape is:",conv_6.shape) # (None,104,104,64)
        conv_7 = self.my_conv(name="filter_7", shape=[3, 3, 64, 128], input_data=conv_6, strides=[1, 1, 1, 1],
                              padding="SAME",training_able=self.training_able[6],init_filter = reader.get_tensor("filter_7"),
                             init_beta = reader.get_tensor("beta7"),init_gama = reader.get_tensor("gama7")) # (None,64,64,128)
        print("conv_7 shape is:", conv_7.shape) # (None,104,104,128)
        resi_2 = conv_7 +conv_5

        conv_8 =self.my_conv(name ="filter_8",shape=[3,3,128,64],input_data=resi_2,strides=[1,2,2,1],
                             padding="SAME",training_able=self.training_able[7],init_filter = reader.get_tensor("filter_8"),
                            init_beta = reader.get_tensor("beta8"),init_gama = reader.get_tensor("gama8"))
        print("conv_8 shape is:",conv_8.shape) # (None,52,52,64)
        conv_9 = self.my_conv(name="filter_9", shape=[3, 3, 64, 64], input_data=conv_8, strides=[1, 2, 2, 1],
                              padding="SAME",training_able=self.training_able[8],init_filter = reader.get_tensor("filter_9"),
                             init_beta = reader.get_tensor("beta9"),init_gama = reader.get_tensor("gama9"))
        print("conv_9 shape is:", conv_9.shape)  # (None,26,26,64)
        """
        conv_10 = self.my_conv(name="filter_10", shape=[3, 3, 16, 14], input_data=resi_2, strides=[1, 2, 2, 1],
                              padding="SAME")
        print("conv_10 shape is:", conv_10.shape)  # (None,8,8,14)
        """
        

        return conv_9 ,conv_8

    def yolo_head_for_retrain(self,extract_result,skip_connection,reader):
        conv_10 = self.my_conv_no_bn(name="filter_10", shape=None, input_data=extract_result, strides=[1, 1, 1, 1],
                              padding="SAME",init_filter = reader.get_tensor("filter_10"),training_able = self.training_able[9])    # (None,26,26,64)
        print("conv_10 shape is:", conv_10.shape)
        conv_11 = self.my_conv_no_bn(name="filter_11",shape=None,input_data=conv_10,strides=[1,1,1,1],
                               padding="SAME",init_filter = reader.get_tensor("filter_11"),training_able = self.training_able[10])   # (None,26,26,64)
        print("conv_11 shape is:", conv_11.shape)
        #conv_skip_con = self.my_conv(name = "skipfilter_01",shape = [1,1,64,64],input_data = skip_connection,strides=[1,1,1,1],
                                     #padding="SAME")
        #to_depth_skip_con = tf.space_to_depth(conv_skip_con,2) # (None,26,26,256)
        #reorg = tf.concat(axis = -1,values = [conv_11,to_depth_skip_con])    # (None,26,26,320)
        #print("reorg.shape is:",reorg.shape)
        conv_12 = self.my_conv_no_bn(name="filter_12",shape=None,input_data=conv_11,strides=[1,1,1,1],
                               padding="SAME",init_filter = reader.get_tensor("filter_12"),training_able = self.training_able[11])  # (None,26,26,128)
        print("conv_12 shape is:", conv_12.shape)
        conv_13 = self.my_conv_no_bn(name="filter_13", shape=None, input_data=conv_12, strides=[1, 2, 2, 1],
                               padding="SAME",init_filter = reader.get_tensor("filter_13"),training_able = self.training_able[12])  # (None,13,13,128)
        print("conv_13 shape is:", conv_13.shape)
        conv_skip_con = self.my_conv_no_bn(name = "skipfilter_01",shape = None,input_data = skip_connection,strides=[1,1,1,1],
                                     padding="SAME",init_filter = reader.get_tensor("skipfilter_01"),training_able = self.training_able[13])
        to_depth_skip_con = tf.space_to_depth(conv_skip_con,4) # (None,13,13,1024)
        reorg = tf.concat(axis = -1,values = [conv_13,to_depth_skip_con])    # (None,13,13,1024+128)
        print("reorg.shape is:",reorg.shape)
        #conv_14 = self.my_conv(name = "filter_14",shape = [1,1,64,64],input_data = skip_connection,strides = [1,1,1,1],padding = "SAME")
        
        filter_15 = tf.get_variable(name="filter_15",initializer=reader.get_tensor("filter_15"),
                                   trainable = self.training_able[14])
        self.save_list.append(filter_15)
        conv_15 = tf.nn.conv2d(reorg, filter_15, strides=[1, 1, 1, 1], padding="SAME")
        # (None,13,13,?)
        # (None,13,13,?)
        print("conv_15 shape is:", conv_15.shape)
        return conv_15
    def prediction(self, reader, sample,num_classes, anchor_box_tensor, score_threshold =0.5,iou_threshold=0.6):
        sample_placeholder = tf.placeholder(name="sample", shape=[ 1,self.img_height, self.img_width, 3], dtype=tf.float32)
        conv_9,conv_8 = self.backbone_for_pred(reader,sample_placeholder)
        
        conv_14 = self.yolo_head_for_pred(conv_9,conv_8,reader)
        
        print("conv_14.shape",conv_14.shape)
        box_xy, box_wh, box_confidence, class_probability = self.detection(conv_14, num_calsses=num_classes,
                                                                           anchor_box_tensor=anchor_box_tensor)
        print("box_xy.shape is :",box_xy.shape)
        pred_loc = tf.concat([box_xy[...,1:2] - box_wh[...,1:2]*0.5,
                              box_xy[..., 0:1] - box_wh[..., 0:1] * 0.5,
                              box_xy[...,1:2] + box_wh[...,1:2]*0.5,
                              box_xy[..., 0:1] + box_wh[..., 0:1] * 0.5,
                               ], axis=-1)                                   # 传进non_max_suppression时，location是先行，后列，故先y,后x
                                                                             # shape (batch, grid, grid,num_anchor_per_cell,4)
        print(pred_loc.shape)
        pred_loc = tf.minimum(tf.maximum(pred_loc,0),1)  # 让其落在0,1之间
        temp = pred_loc
        # 将坐标转换到像素级别
        height_width_height_width = tf.cast(tf.expand_dims(tf.stack([self.img_height, self.img_width]*2,axis = 0),axis=0),tf.float32)
        print("height_width_height_width shape is:",height_width_height_width.shape)
        print("pred_loc shape is:",pred_loc.shape)
        pred_loc = tf.reshape(pred_loc,[-1,4])*height_width_height_width    # shape (batch*grid*grid*num_anchor_per_cell,4)
        #pred_loc = tf.cast(pred_loc,dtype = tf.int32)
        pred_conf = tf.reshape(box_confidence, shape=[-1])                  # shape (batch*grid*grid*num_anchor_per_cell,)
        pred_prob = tf.reshape(class_probability, [-1, num_classes])            # shape (batch*grid*grid*num_anchor_per_cell,classes)
        box_scores = tf.expand_dims(pred_conf, axis=1) * pred_prob          # confidence * probability_class  shape(batch*grid*grid*num_anchor_per_cell,classes)
        box_label = tf.argmax(box_scores, axis=-1)                          # shape (batch*grid*grid*num_anchor_per_cell,)
        box_scores_max = tf.reduce_max(box_scores, axis=-1)                 # shape (batch*grid*grid*num_anchor_per_cell,)
        pred_mask = box_scores_max > score_threshold                        # shape (batch*grid*grid*num_anchor_per_cell,)
        boxes = tf.boolean_mask(pred_loc, pred_mask)                        # shape (unknown,4)， 因为pred_mask中mask对应的pred_loc的数据会被抛弃
        scores = tf.boolean_mask(box_scores_max, pred_mask)                 # shape  (unknown,)
        pred_classes = tf.boolean_mask(box_label, pred_mask)                # shape  (unknown,)
        idx_nms = tf.image.non_max_suppression(boxes, scores,
                                               max_output_size=8,
                                               iou_threshold=iou_threshold)
        boxes = tf.gather(boxes, idx_nms)
        scores = tf.gather(scores, idx_nms)
        classes = tf.gather(pred_classes, idx_nms)
        print("boxes.shape is:",boxes.shape)
        print("scores shape is:",scores.shape)
        print("classes shape is :",classes.shape)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            fin_boxes , fin_scores, fin_classes ,temp_value = sess.run([boxes,scores,classes,temp],feed_dict={sample_placeholder:sample/255})
        return fin_boxes, fin_scores, fin_classes ,temp_value
    def draw_rectangle(self,boxes, scores, classes,img,class_dict):
        """

        :param boxes:  (?, 4) 均为正整数
        :param scores: (?,)
        :param classes:  (?,) 为目标类的序号
        :param img:      (416,416,3) 正整数
        :param class_dict:  以 目标类序号为键，目标类名为值的字典
        :return:          nothing
        """
        
        color_list = [(0,0,255),(0,255,0),(255,0,0)]
        boxes = boxes.astype(np.int32)
        
        for i in range(boxes.shape[0]):
            now_color = color_list[classes[i]]
            cv2.rectangle(img,(boxes[i,1],boxes[i,0]),(boxes[i,3],boxes[i,2]),now_color)
            #draw_obj.text([boxes[i,1]+2,boxes[i,0]+2,],class_dict[classes[i]]+str(scores[i]), now_color)
        
        return img
    def backbone_for_pred(self,reader,sample_placeholder):
        conv_1 = self.my_conv(name = "filter_1",shape = [3,3,3,32],input_data = sample_placeholder, strides = [1,1,1,1], 
                              padding ="SAME",training_able=False ,init_filter = reader.get_tensor("filter_1"),
                             init_beta = reader.get_tensor("beta1"),init_gama = reader.get_tensor("gama1"))
        print("conv_1 shpae is :",conv_1.shape) # (None,416,416,32)
        conv_2 = self.my_conv(name = "filter_2",shape = [3,3,32,64],input_data = conv_1, strides = [1,2,2,1],
                              padding ="SAME",training_able=False,init_filter = reader.get_tensor("filter_2"),
                             init_beta = reader.get_tensor("beta2"),init_gama = reader.get_tensor("gama2"))
        print("conv_2 shape is:",conv_2.shape) # (None,208,208,64)
        conv_3 = self.my_conv(name="filter_3", shape=[1, 1, 64, 32], input_data=conv_2, strides=[1, 1, 1, 1],
                              padding="VALID",training_able=False,init_filter = reader.get_tensor("filter_3"),
                             init_beta = reader.get_tensor("beta3"),init_gama = reader.get_tensor("gama3"))
        print("conv_3 shape is:", conv_3.shape) # (None,208,208,32)
        conv_4 = self.my_conv(name="filter_4", shape=[3, 3, 32, 64], input_data=conv_3, strides=[1, 1, 1, 1],
                              padding="SAME",training_able=False,init_filter = reader.get_tensor("filter_4"),
                             init_beta = reader.get_tensor("beta4"),init_gama = reader.get_tensor("gama4"))
        print("conv_4 shape is:", conv_4.shape) # (None,208,208,64)
        resi_1 = conv_2 + conv_4 # 残差项
        conv_5 = self.my_conv(name="filter_5", shape=[3, 3, 64, 128], input_data=resi_1, strides=[1, 2, 2, 1],
                              padding="SAME",training_able=False,init_filter = reader.get_tensor("filter_5"),
                             init_beta = reader.get_tensor("beta5"),init_gama = reader.get_tensor("gama5"))
        print("conv_5 shape is:", conv_5.shape) # (None,104,104,128)
        conv_6 = self.my_conv(name = "filter_6", shape =[1,1,128,64], input_data=conv_5, strides=[1,1,1,1],
                              padding="SAME",training_able=False,init_filter = reader.get_tensor("filter_6"),
                             init_beta = reader.get_tensor("beta6"),init_gama = reader.get_tensor("gama6"))
        print("conv_6 shape is:",conv_6.shape) # (None,104,104,64)
        conv_7 = self.my_conv(name="filter_7", shape=[3, 3, 64, 128], input_data=conv_6, strides=[1, 1, 1, 1],
                              padding="SAME",training_able=False,init_filter = reader.get_tensor("filter_7"),
                             init_beta = reader.get_tensor("beta7"),init_gama = reader.get_tensor("gama7")) # (None,64,64,128)
        print("conv_7 shape is:", conv_7.shape) # (None,104,104,128)
        resi_2 = conv_7 +conv_5

        conv_8 =self.my_conv(name ="filter_8",shape=[3,3,128,64],input_data=resi_2,strides=[1,2,2,1],
                             padding="SAME",training_able=False,init_filter = reader.get_tensor("filter_8"),
                            init_beta = reader.get_tensor("beta8"),init_gama = reader.get_tensor("gama8"))
        print("conv_8 shape is:",conv_8.shape) # (None,52,52,64)
        conv_9 = self.my_conv(name="filter_9", shape=[3, 3, 64, 64], input_data=conv_8, strides=[1, 2, 2, 1],
                              padding="SAME",training_able=False,init_filter = reader.get_tensor("filter_9"),
                             init_beta = reader.get_tensor("beta9"),init_gama = reader.get_tensor("gama9"))
        print("conv_9 shape is:", conv_9.shape)  # (None,26,26,64)
        """
        conv_10 = self.my_conv(name="filter_10", shape=[3, 3, 16, 14], input_data=resi_2, strides=[1, 2, 2, 1],
                              padding="SAME")
        print("conv_10 shape is:", conv_10.shape)  # (None,8,8,14)
        """
        

        return conv_9 ,conv_8

    def yolo_head_for_pred(self, extract_result,skip_connection, reader):
        conv_10 = self.my_conv_no_bn(name="filter_10", shape=None, input_data=extract_result, strides=[1, 1, 1, 1],
                              padding="SAME",init_filter = reader.get_tensor("filter_10"),training_able = False)    # (None,26,26,64)
        print("conv_10 shape is:", conv_10.shape)
        conv_11 = self.my_conv_no_bn(name="filter_11",shape=None,input_data=conv_10,strides=[1,1,1,1],
                               padding="SAME",init_filter = reader.get_tensor("filter_11"),training_able = False)   # (None,26,26,64)
        print("conv_11 shape is:", conv_11.shape)
        #conv_skip_con = self.my_conv(name = "skipfilter_01",shape = [1,1,64,64],input_data = skip_connection,strides=[1,1,1,1],
                                     #padding="SAME")
        #to_depth_skip_con = tf.space_to_depth(conv_skip_con,2) # (None,26,26,256)
        #reorg = tf.concat(axis = -1,values = [conv_11,to_depth_skip_con])    # (None,26,26,320)
        #print("reorg.shape is:",reorg.shape)
        conv_12 = self.my_conv_no_bn(name="filter_12",shape=None,input_data=conv_11,strides=[1,1,1,1],
                               padding="SAME",init_filter = reader.get_tensor("filter_12"),training_able = False)  # (None,26,26,128)
        print("conv_12 shape is:", conv_12.shape)
        conv_13 = self.my_conv_no_bn(name="filter_13", shape=None, input_data=conv_12, strides=[1, 2, 2, 1],
                               padding="SAME",init_filter = reader.get_tensor("filter_13"),training_able = False)  # (None,13,13,128)
        print("conv_13 shape is:", conv_13.shape)
        conv_skip_con = self.my_conv_no_bn(name = "skipfilter_01",shape = None,input_data = skip_connection,strides=[1,1,1,1],
                                     padding="SAME",init_filter = reader.get_tensor("skipfilter_01"),training_able = False)
        to_depth_skip_con = tf.space_to_depth(conv_skip_con,4) # (None,13,13,1024)
        reorg = tf.concat(axis = -1,values = [conv_13,to_depth_skip_con])    # (None,13,13,1024+128)
        print("reorg.shape is:",reorg.shape)
        #conv_14 = self.my_conv(name = "filter_14",shape = [1,1,64,64],input_data = skip_connection,strides = [1,1,1,1],padding = "SAME")
        
        filter_15 = tf.get_variable(name="filter_15",initializer=reader.get_tensor("filter_15"),trainable = False)
        self.save_list.append(filter_15)
        conv_15 = tf.nn.conv2d(reorg, filter_15, strides=[1, 1, 1, 1], padding="SAME")
      
        # (None,13,13,?)
        print("conv_15 shape is:", conv_15.shape)
        return conv_15








if __name__ =="__main__":
    reader = pywrap_tensorflow.NewCheckpointReader(r'D:\studyINF\AI\YOLOv3\test_samples_copy2\model.ckpt-80')
    dict_class = {}
    dict_class[0] = "ball"
    dict_class[1] = "chess"
    dict_class[2] = "pen"

    anchor_box = np.array([[55.13103448, 29.12413793],
                           [24.93540052, 30.69250646],
                           [21.9784264, 20.24365482],
                           [15.75308642, 47.22222222],
                           [30.5497076, 52.80116959],
                           [49.05109489, 13.37956204],
                           [34.78171091, 23.25073746],
                           [16.73464912, 15.02631579]], dtype=np.float32) / (
                     np.array([416.0, 416.0], dtype=np.float32).reshape(1, 2))

    train_images, anchor_labels, true_box_labels, prior_boxes = readh5(r"D:\studyINF\AI\YOLOv3\train_data2.h5")

    for i in range(0,20):

        sample = train_images[i].reshape((-1,) + train_images[0].shape)
        tf.reset_default_graph()
        one = Yolo(num_true_boxes=true_box_labels.shape[1], num_anchor_per_box=anchor_labels.shape[3],
                   img_width=train_images.shape[2], img_height=train_images.shape[1], training_able=None)
        fin_boxes, fin_scores, fin_classes, temp_value = one.prediction(reader, sample / 255, num_classes=3,
                                                                    anchor_box_tensor=anchor_box, score_threshold=0.4,
                                                                    iou_threshold=0.3)


        img = one.draw_rectangle(fin_boxes, fin_scores, fin_classes, np.squeeze(sample, axis=0), dict_class)
        cv2.imshow("img",cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)




















