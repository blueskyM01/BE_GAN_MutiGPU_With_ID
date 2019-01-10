import tensorflow as tf

'''
#-----------------------------m4_gan_network-----------------------------
dataset_dir = '/media/yang/F/DataSet/Face'
dataset_name = 'lfw-deepfunneled'
datalabel_dir = '/media/yang/F/DataSet/Face/Label'
datalabel_name = 'pair_FGLFW.txt'
log_dir = './logs'
sampel_save_dir = './samples'
num_gpus = 2
epoch = 40
learning_rate = 0.001
beta1 = 0.5
beta2 = 0.5
batch_size = 16
z_dim = 128
g_feats = 64
saveimage_period = 10
savemodel_period = 40
#-----------------------------m4_gan_network-----------------------------
'''

#-----------------------------m4_BE_GAN_network-----------------------------
dataset_dir = '/media/yang/F/DataSet/Face/LFW_FF-GAN'
dataset_name = 'LFW_FF-GAN'
datalabel_dir = '/media/yang/F/DataSet/Face/Label'
datalabel_name = 'LFW_FF-GAN.txt'
log_dir = '/media/yang/F/ubuntu/My_Code/My_GAN/1/logs'
sampel_save_dir = '/media/yang/F/ubuntu/My_Code/My_GAN/1/samples'
checkpoint_dir = '/media/yang/F/ubuntu/My_Code/My_GAN/1/checkpoint'
num_gpus = 1
epoch = 100
batch_size = 64
z_dim = 64

conv_hidden_num = 128

data_format = 'NHWC'

g_lr = 0.00008
d_lr = 0.00008

lr_lower_boundary = 0.00002

gamma = 0.5
lambda_k = 0.5

saveimage_period = 50
savemodel_period = 200
#-----------------------------m4_BE_GAN_network-----------------------------


