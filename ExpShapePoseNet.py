import sys
import numpy as np
import tensorflow as tf
import cv2
import scipy.io as sio
import pose_utils as pu
import os
import os.path
import glob
import time
import scipy
import scipy.io as sio
import ST_model_nonTrainable_AlexNetOnFaces as Pose_model
import utils_3DMM
import csv
import argparse

sys.path.append('./ResNet')
from ThreeDMM_shape import ResNet_101 as resnet101_shape
from ThreeDMM_expr import ResNet_101 as resnet101_expr


class m4_3DMM:
    def __init__(self, cfg):
        self.cfg = cfg

        # Get training image/labels mean/std for pose CNN
        try:
            file = np.load(self.cfg.train_imgs_mean_file_path, )
            self.train_mean_vec = file["train_mean_vec"]  # [0,1]
            print('Load ' + self.cfg.train_imgs_mean_file_path + ' successful....')
        except:
            raise Exception('Load ' + self.cfg.train_imgs_mean_file_path + ' failed....')
        del file

        try:
            file = np.load(self.cfg.train_labels_mean_std_file_path)
            self.mean_labels = file["mean_labels"]
            self.std_labels = file["std_labels"]
            print('Load ' + self.cfg.train_labels_mean_std_file_path + ' successful....')
        except:
            raise Exception('Load ' + self.cfg.train_labels_mean_std_file_path + ' failed....')
        del file

        try:
            # Get training image mean for Shape CNN
            mean_image_shape = np.load(self.cfg.ThreeDMM_shape_mean_file_path)  # 3 x 224 x 224
            self.mean_image_shape = np.transpose(mean_image_shape, [1, 2, 0])  # 224 x 224 x 3, [0,255]
            print('Load ' + self.cfg.ThreeDMM_shape_mean_file_path +' successful....')
        except:
            raise Exception('Load ' + self.cfg.ThreeDMM_shape_mean_file_path +' failed....')
        del mean_image_shape

    def extract_PSE_feats(self, x):
        '''
        :param x: x format is RGB and is value range is [-1,1].
        :return: fc1ls: shape, fc1le: expression, pose_model.preds_unNormalized: pose
        '''
        x = tf.image.resize_images(x, [227, 227])

        # x is RGB and is value range is [-1,1].
        # first we need to change RGB to BGR;
        batch_, height_, width_, nc = x.get_shape().as_list()
        R = tf.reshape(x[:, :, :, 0], [batch_, height_, width_, 1])
        G = tf.reshape(x[:, :, :, 1], [batch_, height_, width_, 1])
        B = tf.reshape(x[:, :, :, 2], [batch_, height_, width_, 1])
        x = tf.concat([B, G, R], axis=3)
        # second change range [-1,1] to [0,255]
        x = (x + 1.0) * 127.5

        ###################
        # Face Pose-Net
        ###################
        try:
            net_data = np.load(self.cfg.PAM_frontal_ALexNet_file_path, encoding="latin1").item()
            pose_labels = np.zeros([self.cfg.batch_size* self.cfg.num_gpus, 6])
            print('Load ' + self.cfg.PAM_frontal_ALexNet_file_path+ ' successful....')
        except:
            raise Exception('Load ' + self.cfg.PAM_frontal_ALexNet_file_path+ ' failed....')
        x1 = tf.image.resize_bilinear(x, tf.constant([227, 227], dtype=tf.int32))

        # Image normalization
        x1 = x1 / 255.  # from [0,255] to [0,1]
        # subtract training mean
        mean = tf.reshape(self.train_mean_vec, [1, 1, 1, 3])
        mean = tf.cast(mean, 'float32')
        x1 = x1 - mean

        pose_model = Pose_model.Pose_Estimation(x1, pose_labels, 'valid', 0, 1, 1, 0.01, net_data, self.cfg.batch_size* self.cfg.num_gpus,
                                                self.mean_labels, self.std_labels)
        pose_model._build_graph()
        self.pose = pose_model.preds_unNormalized
        del net_data

        ###################
        # Shape CNN
        ###################
        x2 = tf.image.resize_bilinear(x, tf.constant([224, 224], dtype=tf.int32))
        x2 = tf.cast(x2, 'float32')
        x2 = tf.reshape(x2, [self.cfg.batch_size * self.cfg.num_gpus, 224, 224, 3])

        # Image normalization
        mean = tf.reshape(self.mean_image_shape, [1, 224, 224, 3])
        mean = tf.cast(mean, 'float32')
        x2 = x2 - mean

        with tf.variable_scope('shapeCNN'):
            net_shape = resnet101_shape({'input': x2}, trainable=True)
            pool5 = net_shape.layers['pool5']
            pool5 = tf.squeeze(pool5)
            pool5 = tf.reshape(pool5, [self.cfg.batch_size* self.cfg.num_gpus, -1])
            try:
                npzfile = np.load(self.cfg.ShapeNet_fc_weights_file_path)
                print('Load ' + self.cfg.ShapeNet_fc_weights_file_path + ' successful....')
            except:
                raise Exception('Load ' + self.cfg.ShapeNet_fc_weights_file_path + ' failed....')

            ini_weights_shape = npzfile['ini_weights_shape']
            ini_biases_shape = npzfile['ini_biases_shape']
            with tf.variable_scope('shapeCNN_fc1'):
                fc1ws = tf.Variable(tf.reshape(ini_weights_shape, [2048, -1]), trainable=True, name='weights')
                fc1bs = tf.Variable(tf.reshape(ini_biases_shape, [-1]), trainable=True, name='biases')
                self.fc1ls = tf.nn.bias_add(tf.matmul(pool5, fc1ws), fc1bs)

        ###################
        # Expression CNN
        ###################
        with tf.variable_scope('exprCNN'):
            net_expr = resnet101_expr({'input': x2}, trainable=True)
            pool5 = net_expr.layers['pool5']
            pool5 = tf.squeeze(pool5)
            pool5 = tf.reshape(pool5, [self.cfg.batch_size* self.cfg.num_gpus, -1])

            try:
                npzfile = np.load(self.cfg.ExpNet_fc_weights_file_path)
                ini_weights_expr = npzfile['ini_weights_expr']
                ini_biases_expr = npzfile['ini_biases_expr']
                print('Load ' + self.cfg.ExpNet_fc_weights_file_path + '  successful....')
            except:
                raise Exception('Load ' + self.cfg.ExpNet_fc_weights_file_path + '  failed....')

            with tf.variable_scope('exprCNN_fc1'):
                fc1we = tf.Variable(tf.reshape(ini_weights_expr, [2048, 29]), trainable=True, name='weights')
                fc1be = tf.Variable(tf.reshape(ini_biases_expr, [29]), trainable=True, name='biases')
                self.fc1le = tf.nn.bias_add(tf.matmul(pool5, fc1we), fc1be)



        # return fc1ls, fc1le, pose_model.preds_unNormalized


# parser = argparse.ArgumentParser()
#
# parser.add_argument("--image_size", default=227, type=int, help="image_size")
# parser.add_argument("--num_gpus", default=1, type=int, help="num_gpus")
# parser.add_argument("--batch_size", default=2, type=int, help="batch_size")
# parser.add_argument("--mesh_folder", default='./output_ply', type=str, help="mesh_folder")
#
# parser.add_argument("--train_imgs_mean_file_path",
#                     default='/home/yang/My_Job/fpn_new_model/perturb_Oxford_train_imgs_mean.npz', type=str,
#                     help="Load perturb_Oxford_train_imgs_mean.npz")
# parser.add_argument("--train_labels_mean_std_file_path",
#                     default='/home/yang/My_Job/fpn_new_model/perturb_Oxford_train_labels_mean_std.npz', type=str,
#                     help="Load perturb_Oxford_train_labels_mean_std.npz")
# parser.add_argument("--ThreeDMM_shape_mean_file_path",
#                     default='/home/yang/My_Job/Shape_Model/3DMM_shape_mean.npy', type=str,
#                     help="Load 3DMM_shape_mean.npy")
# parser.add_argument("--PAM_frontal_ALexNet_file_path",
#                     default='/home/yang/My_Job/fpn_new_model/PAM_frontal_ALexNet.npy', type=str,
#                     help="Load PAM_frontal_ALexNet.npy")
# parser.add_argument("--ShapeNet_fc_weights_file_path",
#                     default='/home/yang/My_Job/study/Expression-Net/ResNet/ShapeNet_fc_weights.npz', type=str,
#                     help="Load ShapeNet_fc_weights.npz")
# parser.add_argument("--ExpNet_fc_weights_file_path",
#                     default='/home/yang/My_Job/study/Expression-Net/ResNet/ExpNet_fc_weights.npz', type=str,
#                     help="Load ResNet/ExpNet_fc_weights.npz")
# parser.add_argument("--fpn_new_model_ckpt_file_path",
#                     default='/home/yang/My_Job/fpn_new_model/model_0_1.0_1.0_1e-07_1_16000.ckpt', type=str,
#                     help="Load model_0_1.0_1.0_1e-07_1_16000.ckpt")
# parser.add_argument("--Shape_Model_file_path",
#                     default='/home/yang/My_Job/Shape_Model/ini_ShapeTextureNet_model.ckpt', type=str,
#                     help="Load ini_ShapeTextureNet_model.ckpt")
# parser.add_argument("--Expression_Model_file_path",
#                     default='/home/yang/My_Job/Expression_Model/ini_exprNet_model.ckpt', type=str,
#                     help="Load ini_exprNet_model.ckpt")
# parser.add_argument("--BaselFaceModel_mod_file_path",
#                     default='/home/yang/My_Job/Shape_Model/BaselFaceModel_mod.mat', type=str,
#                     help="Load BaselFaceModel_mod.mat")
#
# cfg = parser.parse_args()
#
# if __name__ == '__main__':
#     if not os.path.exists(cfg.mesh_folder):
#         os.makedirs(cfg.mesh_folder)
#
#     # placeholders for the batches
#     x = tf.placeholder(tf.float32, [cfg.batch_size, 227, 227, 3])
#     with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#         init_op = tf.global_variables_initializer()
#         sess.run(init_op)
#
#         expr_shape_pose = m4_3DMM(sess, cfg)
#         fc1ls, fc1le, pose_model = expr_shape_pose.extract_PSE_feats(x)
#
#         print('> Start to estimate Expression, Shape, and Pose!')
#
#         image = cv2.imread('/home/yang/My_Job/study/Expression-Net/subject1_a.jpg', 1)  # BGR
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image_size_h, image_size_w, nc = image.shape
#         image = image / 127.5 - 1.0
#
#         image1 = cv2.imread('/home/yang/My_Job/study/Expression-Net/subject15_a.jpg', 1)  # BGR
#         image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
#         image_size_h, image_size_w, nc = image1.shape
#         image1 = image1 / 127.5 - 1.0
#
#         image_list = []
#         image_list.append(image)
#         image_list.append(image1)
#
#         image_np = np.asarray(image_list)
#         image_np = np.reshape(image_np, [2, image_size_h, image_size_w, 3])
#
#         (Shape_Texture, Expr, Pose) = sess.run([fc1ls, fc1le, pose_model], feed_dict={x: image_np})
#
#         # -------------------------------make .ply file---------------------------------
#         ## Modifed Basel Face Model
#         BFM_path = cfg.BaselFaceModel_mod_file_path
#         model = scipy.io.loadmat(BFM_path, squeeze_me=True, struct_as_record=False)
#         model = model["BFM"]
#         faces = model.faces - 1
#         print('> Loaded the Basel Face Model to write the 3D output!')
#
#         for i in range(cfg.batch_size):
#             outFile = cfg.mesh_folder + '/' + 'haha' + '_' + str(i)
#
#             Pose[i] = np.reshape(Pose[i], [-1])
#             Shape_Texture[i] = np.reshape(Shape_Texture[i], [-1])
#             Shape = Shape_Texture[i][0:99]
#             Shape = np.reshape(Shape, [-1])
#             Expr[i] = np.reshape(Expr[i], [-1])
#
#             #########################################
#             ### Save 3D shape information (.ply file)
#
#             # Shape + Expression + Pose
#             SEP, TEP = utils_3DMM.projectBackBFM_withEP(model, Shape_Texture[i], Expr[i], Pose[i])
#             utils_3DMM.write_ply_textureless(outFile + '_Shape_Expr_Pose.ply', SEP, faces)
