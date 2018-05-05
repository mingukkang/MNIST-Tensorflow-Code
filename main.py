from model import *
import tensorflow as tf
import zipfile

if __name__ =="__main__":
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer("num_classes",10, "number of class")
    flags.DEFINE_integer("load_size", 32, "size of load image")
    flags.DEFINE_integer("channels", 1, "channel of image")
    flags.DEFINE_integer("multiple", 5, "augmentation multiple")
    flags.DEFINE_string("depth", "64,128,256,384,512,768,1024", "depth list, ex) 64,128,256,512,768,1024")
    flags.DEFINE_string("data_container", "./mnist_png", "Data container")
    flags.DEFINE_string("training_dir","./mnist_png/training/","training data directory")
    flags.DEFINE_string("val_dir","./mnist_png/validation/","validation data directory")
    flags.DEFINE_string("test_dir","./mnist_png/testing/","testing data directory")
    flags.DEFINE_float("lamda", 0.05, "lamda value for regularization loss")
    flags.DEFINE_float("lr",0.0001,"learning_rate")
    flags.DEFINE_float("decay_rate",0.95,"learning rate decay rate" )
    flags.DEFINE_integer("decay_step",5000,"learning rate decay step")
    flags.DEFINE_integer("batch_size",64, "batch size of traing data")
    flags.DEFINE_integer("num_epoch", 150, "number of epoch for training")
    depth = list(map(int,FLAGS.depth.split(',')))

    zip_extractor(FLAGS.data_container)
    sess = tf.Session()
    Model_1 = CNN(sess = sess,
               depth = depth,
               num_classes = FLAGS.num_classes,
               load_size = FLAGS.load_size,
               channels= FLAGS.channels,
               data_dir = FLAGS.training_dir,
               val_dir = FLAGS.val_dir,
               test_dir = FLAGS.test_dir)


    Model_1.get_loss(lamda = FLAGS.lamda)

    Model_1.training_op(lr = FLAGS.lr, decay_rate = FLAGS.decay_rate, decay_step = FLAGS.decay_step)
    Model_1.training_run(batch_size = FLAGS.batch_size, num_epoch = FLAGS.num_epoch, multiple = FLAGS.multiple)
    Model_1.testing_run()
