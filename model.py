from utils import *
from data import *
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import re

class CNN:

    def __init__(self,sess,depth,num_classes,load_size,channels,data_dir,val_dir,test_dir):
        self.sess = sess
        self.depth = depth
        self.num_classes = num_classes
        self.load_size = load_size
        self.channels = channels
        self.data_dir = data_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.build_net()

    def build_net(self):
        self.X_img = tf.placeholder(tf.float32, shape = [None,self.load_size,self.load_size,self.channels], name = "Input_IMG")
        self.Y = tf.placeholder(tf.float32, shape = [None,self.num_classes], name = "Label")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        net = leaky(bn(conv(self.X_img, self.depth[0], name = "Conv1"), is_training = self.is_training, name = "bn1"))
        net = leaky(bn(conv(net, self.depth[1], name = "Conv2"), is_training = self.is_training, name="bn2"))
        net = strided_conv(net, name = "strided_conv1")
        net = leaky(bn(conv(net, self.depth[2], name="Conv3"), is_training = self.is_training, name = "bn3"))
        net = leaky(bn(conv(net, self.depth[3], name="Conv4"), is_training = self.is_training, name="bn4"))
        net = strided_conv(net, name = "strided_conv2")
        net = leaky(bn(conv(net, self.depth[4], name="Conv5"), is_training= self.is_training, name = "bn5"))
        net = leaky(bn(conv(net, self.depth[5], name="Conv6"), is_training= self.is_training, name = "bn6"))
        net = maxpool(net, name = "max_pool")

        net = tf.reduce_mean(net, axis = (1,2), name ="GAP")
        net = tf.layers.flatten(net)
        self.net = dense(net,self.num_classes,name = "Dense1")
        self.hypothesis = tf.nn.softmax(self.net)

    def get_loss(self,lamda):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = self.net, labels = self.Y)
        self.entropy_loss = tf.reduce_mean(cross_entropy)

        total_vars = tf.trainable_variables()
        weights_name_list = [var for var in total_vars if "kernel" in var.name]
        loss_holder = []
        for w in range(len(weights_name_list)):
            l2_loss = tf.nn.l2_loss(weights_name_list[w])
            loss_holder.append(l2_loss)
        self.regular_loss = tf.reduce_mean(loss_holder)*lamda
        self.loss = self.entropy_loss + self.regular_loss

    def training_op(self,lr,decay_rate,decay_step):
        self.global_step = tf.Variable(0, trainable=False, name ="global_step")
        self.lr_decayed = lr*decay_rate**(self.global_step/decay_step)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimization = tf.train.AdamOptimizer(beta1 = 0.5, learning_rate = self.lr_decayed).\
                minimize(self.loss,global_step = self.global_step)

        ################### code for Tensorboard ###################
        self.scalar_to_write = tf.placeholder(tf.float32)
        self.loss_summary = tf.summary.scalar("Loss",self.scalar_to_write)
        self.lr_summary = tf.summary.scalar("learing_rate", self.lr_decayed)

        self.writer1 = tf.summary.FileWriter("./logs/total_loss")
        self.writer2 = tf.summary.FileWriter("./logs/entropy_loss")
        self.writer3 = tf.summary.FileWriter("./logs/reg_loss")
        self.writer4 = tf.summary.FileWriter("./logs/lr")
        ################### code for Tensorboard ###################

    def training_run(self,batch_size,num_epoch, multiple):

        self.train_ob = ImageData(batch_size,self.load_size,self.channels,augment_flag = True)
        batch_xs, batch_ys = self.train_ob.get_batch(self.data_dir,aug_multiple = multiple, name = "Training")
        val_ob = ImageData(5000,self.load_size,self.channels,augment_flag = False)
        val_xs, val_ys = val_ob.get_batch(self.val_dir,aug_multiple = 1, name = "Validation")

        with tf.Session() as sess:

            saver = tf.train.Saver()
            save_dir = "./saver"

            if not os.path.exists(os.getcwd() + save_dir[1:]):
                os.makedirs(os.getcwd() + save_dir[1:])
            self.recent_ckpt_job_path = tf.train.latest_checkpoint("saver")

            if self.recent_ckpt_job_path is not None:
                sess.run(tf.initialize_all_variables())
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                print("\nWe find ckpt name: ", self.recent_ckpt_job_path)
                print("restore parameters...")
                saver.restore(sess,self.recent_ckpt_job_path)
                print("complete to restore!\n")
                base_name = os.path.basename(self.recent_ckpt_job_path)
                counter = int(next(re.finditer("(\d+)(?!.*\d)", base_name)).group(0)) +1
            else:
                sess.run(tf.initialize_all_variables())
                print("we fall to restore parameters.")
                print("we will Train network from zero!\n")
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess = sess, coord = coord)
                counter = 0

            ################### code for Tensorboard ###################
            # self.writer3.add_graph(sess.graph)
            ################### code for Tensorboard ###################

            total_batch = int(self.train_ob.total_size/self.train_ob.new_batch_size)
            image_val, label_val = sess.run([val_xs, val_ys])

            print("Start Training!")
            print("____________________________________________________________________________\n")
            start_time = time.time()
            for i in range(counter,num_epoch,1):
                loss_total = 0
                loss_entro = 0
                loss_reg = 0
                for j in range(total_batch+1):
                    image_batch, label_batch = sess.run([batch_xs, batch_ys])
                    tl,rl,el,_,decayed,g, = sess.run([self.loss,self.regular_loss,self.entropy_loss,self.optimization, self.lr_decayed,self.global_step],
                                     feed_dict = {self.X_img:image_batch, self.Y:label_batch, self.is_training:True})
                    loss_total += tl/total_batch
                    loss_entro += el/total_batch
                    loss_reg += rl/total_batch

                ################### code for Tensorboard ###################
                s = sess.run(self.loss_summary, feed_dict = {self.scalar_to_write: loss_total})
                self.writer1.add_summary(s,global_step = g)
                s = sess.run(self.loss_summary, feed_dict = {self.scalar_to_write: loss_entro})
                self.writer2.add_summary(s,global_step = g)
                s = sess.run(self.loss_summary, feed_dict = {self.scalar_to_write: loss_reg})
                self.writer3.add_summary(s,global_step = g)
                s = sess.run(self.lr_summary)
                self.writer4.add_summary(s,global_step = g)
                ################### code for Tensorboard ###################

                hypothesis_val = sess.run(self.hypothesis, feed_dict = {self.X_img: image_val, self.is_training: False})
                A = get_Accuracy(hypothesis = hypothesis_val,labels = label_val)
                hour = int((time.time() - start_time) / 3600)
                min = int(((time.time() - start_time) - 3600 * hour) / 60)
                sec = int((time.time() - start_time) - 3600 * hour - 60 * min)
                print("\nEpoch: %.4d    Time: %d hour %d min %d sec" %(i,hour,min,sec))
                print("lr: %f   Loss: %.4f    Accuracy_val: %f" % (decayed,loss_total, A), end = "%\n")

                if ((i % 5 == 0) or (i == (num_epoch -1))):
                    ckpt_path = saver.save(sess,save_dir + "/parameter",i)

                    print("saving_dir: " + os.getcwd() + ckpt_path[1:] + "\n")

            coord.request_stop()
            coord.join(threads)
            print("\n____________________________________________________________________________")
            print("Complete Training!")

    def testing_run(self):
        batch_size = 1000
        print("load Test Data...")
        test_ob = ImageData(batch_size, self.load_size, self.channels, augment_flag=False)
        test_xs, test_ys = test_ob.get_batch(self.test_dir, aug_multiple=1, name="Test")
        print("you should input batch_size which can divide total number of data!")

        if (test_ob.total_size % batch_size) == 0:
            hypothesis_holder =[]
            label_holder = []

            with tf.Session() as sess:
                saver = tf.train.Saver()
                sess.run(tf.initialize_all_variables())
                saver.restore(sess, self.recent_ckpt_job_path)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                total_batch = int(test_ob.total_size / test_ob.new_batch_size)

                for i in range(total_batch):
                    image_test, label_test = sess.run([test_xs, test_ys])
                    h = sess.run(self.hypothesis, feed_dict={self.X_img: image_test, self.is_training: False})
                    hypothesis_holder.append(h)
                    label_holder.append(label_test)
                hypothesis_holder = np.concatenate(hypothesis_holder, axis = 0)
                label_holder = np.concatenate(label_holder, axis = 0)
                A_test = get_Accuracy(hypothesis_holder, label_holder)
                print("Final Test Accuracy : %f" % (A_test), end ='%\n')

                coord.request_stop()
                coord.join(threads)
        else:
            print("you shoul change batch_size in model.py, line 159")
