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

# -----------------------------m4_BE_GAN_network-----------------------------
save_dir = '/WebFace_generate_lr_0.00008/'

dataset_dir = '/media/yang/F/DataSet/Face'
dataset_name = 'CASIA-WebFace'
datalabel_dir = '/media/yang/F/DataSet/Face/Label'
datalabel_name = 'CASIA-WebFace.txt'
log_dir = '/media/yang/F/ubuntu/My_Code/My_GAN' + save_dir+'logs'  # need to change
sampel_save_dir = '/media/yang/F/ubuntu/My_Code/My_GAN' + save_dir+'samples'  # need to change
checkpoint_dir = '/media/yang/F/ubuntu/My_Code/My_GAN' + save_dir+'checkpoint'  # need to change
num_gpus = 2
epoch = 200
batch_size = 16  # need to change
z_dim = 64

conv_hidden_num = 128

data_format = 'NHWC'

g_lr = 0.00008  # need to change
d_lr = 0.00008  # need to change

lr_lower_boundary = 0.00002

gamma = 0.5
lambda_k = 0.5

saveimage_period = 120
savemodel_period = 480
# -----------------------------m4_BE_GAN_network-----------------------------
