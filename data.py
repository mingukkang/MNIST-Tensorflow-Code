import tensorflow as tf
import random
from glob import glob
import os
import math

class ImageData:

    def __init__(self,batch_size,load_size,channels,augment_flag):
        self.batch_size = batch_size
        self.load_size = load_size
        self.channels = channels
        self.augment_flag = augment_flag

    def get_batch(self,data_dir,aug_multiple,name):
        #data_dir = directory of the data, in the directory there should be folders whose name is order of classes.
        #if aug_multiple is 4 and the number of data set is 10000, then we will make get 40,000 pieces of data.
        class_list = os.listdir(data_dir)
        class_list.sort()
        n_classes = len(class_list)
        image_list = []
        label_list = []

        for i in range(n_classes):
            img_class = glob(data_dir + class_list[i] + '/*.*') 
            image_list += img_class 
            for j in range(len(img_class)):
                label_list += [i]
        images = tf.cast(image_list, tf.string)
        labels = tf.cast(label_list, tf.int32)

        # make an input queue
        input_queue = tf.train.slice_input_producer([images,labels], shuffle = True)
        label = input_queue[1]
        image_contents = tf.read_file(input_queue[0])
        extension = [os.path.splitext(x)[1] for x in image_list]

        for e in range(len(extension)):
            if (extension[e] == ".jpeg" or extension[e] == ".jpg"):
                image = tf.image.decode_jpeg(image_contents, channels=self.channels)
            elif (extension[e] == ".png"):
                image = tf.image.decode_png(image_contents, channels=self.channels)
            else:
                print("Error! Extension of Data is not Png,Jpg or Jpeg!")

        image = tf.image.resize_images(image, [self.load_size, self.load_size])
        image = tf.cast(image, tf.float32)/255 # Normalizing
        mod = self.batch_size % aug_multiple
        sub_batch_size = int(self.batch_size/aug_multiple)
        self.image_batch, self.label_batch = tf.train.batch([image,label],
                                                  batch_size = sub_batch_size,
                                                  num_threads = 8,
                                                  capacity = sub_batch_size*2)

        if self.augment_flag is True:
            for f in range(aug_multiple - 1):
                augment_size = self.load_size + (30 if self.load_size ==256 else 8)
                for g in range(sub_batch_size):
                    aug_img = self.augmentation(self.image_batch[g,:,:,:],augment_size)
                    self.image_batch = tf.concat([self.image_batch,aug_img], axis = 0)
                    self.label_batch = tf.concat([self.label_batch, self.label_batch[g:g+1]], axis =0)
            
            if mod !=0:
                for m in range(mod):
                    aug_img = self.augmentation(self.image_batch[m],augment_size)
                    self.image_batch = tf.concat([self.image_batch, aug_img], axis = 0)
                    self.label_batch = tf.concat([self.label_batch, self.label_batch[m:m+1]], axis = 0)

        self.label_batch = tf.one_hot(self.label_batch, depth = 10)
        self.total_size = len(image_list)*aug_multiple

        print("\nData type: ", name)
        print("Batch size : %d" % (self.new_batch_size))
        print("Number of Original data: %d\nAugmentation rate: %d" % (len(image_list), (aug_multiple -1)*100), '%')
        print("Number of total data: %d" % (self.total_size))
        print("Shape of images = ",self.image_batch.get_shape(), "    Shape of Labels = ", self.label_batch.get_shape())
        print("________________________ Batch is prepared in queue ________________________\n")

        return self.image_batch, self.label_batch

    def augmentation(self,img,augment_size):
        seed = random.randint(0,2**31-1)
        angle = random.randint(-15, 15)
        radian = angle*(math.pi/180)
        ori_shape = img.get_shape()
        p = random.random()
        if p > 0.75:
            noise = tf.random_normal(shape = ori_shape, mean =0.0, stddev = 0.2, dtype = tf.float32)
            img = tf.add(img,noise)

        # img = tf.image.random_flip_left_right(img,seed = seed)
        # img = tf.image.random_hue(img, max_delta=0.05)
        img = tf.image.random_contrast(img, lower = 0.3, upper = 1.0)
        img = tf.image.random_brightness(img, max_delta = 0.2)
        # img = tf.image.random_saturation(img,lower = 0.0, upper = 2.0)
        img = tf.image.resize_images(img, [augment_size,augment_size])
        img = tf.random_crop(img, ori_shape, seed = seed)
        img = tf.contrib.image.rotate(img,radian)
        img = tf.reshape(img, [1,self.load_size,self.load_size,self.channels])
        img = tf.minimum(img, 1.0)
        img = tf.maximum(img, 0.0)

        return img
