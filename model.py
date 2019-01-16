import tensorflow as tf
import numpy as np
import os
import datetime
from utils import *
from ops import *
from networks import *
import time


class my_gan:
    def __init__(self, sess, cfg):
        self.sess = sess
        self.cfg = cfg
        self.images = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size * self.cfg.num_gpus, 64, 64, 3],
                                     name='real_image')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size * self.cfg.num_gpus, 10000],
                                     name='id')
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, self.cfg.z_dim], name='noise_z')

        '''
        #-----------------------------m4_gan_network-----------------------------
        my_gan_model = m4_gan_network(self.cfg)
        my_gan_model.build_model(self.images, self.labels, self.z)
        self.g_optim = my_gan_model.g_optim
        self.d_optim = my_gan_model.d_optim
        self.g_loss = my_gan_model.g_loss
        self.d_loss = my_gan_model.d_loss
        self.image_fake_sum = my_gan_model.image_fake_sum
        self.g_loss_sum = my_gan_model.g_loss_sum
        self.d_loss_sum = my_gan_model.d_loss_sum
        self.global_step = my_gan_model.global_step
        self.sampler = my_gan_model.sampler
        #-----------------------------m4_gan_network-----------------------------
        '''

        # -----------------------------m4_BE_GAN_network-----------------------------
        m4_BE_GAN_model = m4_BE_GAN_network(self.cfg)
        m4_BE_GAN_model.build_model(self.images, self.labels, self.z)
        self.g_optim = m4_BE_GAN_model.g_optim
        self.d_optim = m4_BE_GAN_model.d_optim
        self.g_loss = m4_BE_GAN_model.g_loss
        self.d_loss = m4_BE_GAN_model.d_loss

        self.image_fake_sum = m4_BE_GAN_model.image_fake_sum
        self.g_loss_sum = m4_BE_GAN_model.g_loss_sum
        self.d_loss_sum = m4_BE_GAN_model.d_loss_sum
        self.global_step = m4_BE_GAN_model.global_step
        self.sampler = m4_BE_GAN_model.sampler
        self.k_update = m4_BE_GAN_model.k_update
        self.k_t = m4_BE_GAN_model.k_t
        self.Mglobal = m4_BE_GAN_model.measure
        self.d_lr_update = m4_BE_GAN_model.d_lr_update
        self.g_lr_update = m4_BE_GAN_model.g_lr_update
        self.d_lr = m4_BE_GAN_model.d_lr
        self.g_lr = m4_BE_GAN_model.g_lr
        self.g_lr_ = self.cfg.g_lr
        self.d_lr_ = self.cfg.d_lr
        # -----------------------------m4_BE_GAN_network-----------------------------

    def train(self):
        try:
            self.saver = tf.train.Saver()
        except:
            print('one model save error....')
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.writer = tf.summary.FileWriter(
            '{}/{}'.format(self.cfg.log_dir, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))),
            self.sess.graph)
        merged = tf.summary.merge_all()
        could_load, counter = self.load(self.cfg.checkpoint_dir, self.cfg.dataset_name)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        names = np.loadtxt(os.path.join(self.cfg.datalabel_dir, self.cfg.datalabel_name), dtype=np.str)
        dataset_size = names.shape[0]
        names, labels = m4_get_file_label_name(os.path.join(self.cfg.datalabel_dir, self.cfg.datalabel_name),
                                               os.path.join(self.cfg.dataset_dir, self.cfg.dataset_name))
        filenames = tf.constant(names)
        filelabels = tf.constant(labels)
        try:
            dataset = tf.data.Dataset.from_tensor_slices((filenames, filelabels))
        except:
            dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, filelabels))

        dataset = dataset.map(m4_parse_function)
        dataset = dataset.shuffle(buffer_size=10000).batch(self.cfg.batch_size * self.cfg.num_gpus).repeat(
            self.cfg.epoch)
        iterator = dataset.make_one_shot_iterator()
        one_element = iterator.get_next()
        batch_idxs = dataset_size // (self.cfg.batch_size * self.cfg.num_gpus)
        batch_images_G, batch_labels_G = self.sess.run(one_element)
        batch_z_G = np.random.uniform(-1, 1, [self.cfg.batch_size * self.cfg.num_gpus, self.cfg.z_dim]).astype(
            np.float32)
        m4_image_save_cv(batch_images_G,
                         '{}/x_fixed.jpg'.format(self.cfg.sampel_save_dir))
        print('save one image.')
        try:
            for epoch in range(1,self.cfg.epoch+1):
                for idx in range(batch_idxs):
                    starttime = datetime.datetime.now()
                    batch_images, batch_labels = self.sess.run(one_element)
                    batch_z = np.random.uniform(-1, 1, [self.cfg.batch_size * self.cfg.num_gpus, self.cfg.z_dim]).astype(
                        np.float32)
                    if batch_images.shape[0] < self.cfg.batch_size * self.cfg.num_gpus:
                        continue


                    # get measure stand
                    k_update, k_t, Mglobal = self.sess.run([self.k_update, self.k_t, self.Mglobal],
                                                           feed_dict={self.images: batch_images,
                                                                      self.z: batch_z})
                    # # Upadate D network
                    # d_loss, merged_ = self.sess.run(
                    #     [self.d_loss, merged],
                    #     feed_dict={self.images: batch_images,
                    #                self.z: batch_z})

                    # Update G network
                    d_loss, g_loss, counter, merged_ = self.sess.run(
                        [self.d_loss, self.g_loss, self.global_step,merged],
                        feed_dict={self.images: batch_images,
                                   self.z: batch_z})
                    if epoch % 1000 == 0 and idx == 1:
                        _, _, self.g_lr_, self.d_lr_, = self.sess.run(
                            [self.g_lr_update, self.d_lr_update, self.g_lr, self.d_lr],
                            feed_dict={self.images: batch_images,
                                       self.z: batch_z})

                    self.writer.add_summary(merged_, counter)
                    endtime = datetime.datetime.now()
                    timediff = (endtime - starttime).total_seconds()
                    print(
                        "Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, k_t: %.8f, Mglobal: %.8f, g_lr: %.8f, d_lr: %.8f" \
                        % (epoch, self.cfg.epoch, idx, batch_idxs, timediff, d_loss, g_loss, k_t, Mglobal, self.g_lr_, self.d_lr_))
                    try:
                        if counter % self.cfg.saveimage_period == 0:
                            samples = self.sess.run([self.sampler], feed_dict={self.z: batch_z_G})
                            m4_image_save_cv(samples[0],
                                             '{}/train_{}_{}.jpg'.format(self.cfg.sampel_save_dir, epoch, counter))
                            print('save one image.')
                    except:
                        print('oen pic error')

                    try:
                        if counter % self.cfg.savemodel_period == 0:
                            self.save(self.cfg.checkpoint_dir, counter, self.cfg.dataset_name)
                    except:
                        print('one model save error....')

        except:
            print('Mission complete!')


    def save(self, checkpoint_dir, step, model_file_name):
        model_name = "GAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, model_file_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir, model_folder_name):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, model_folder_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
