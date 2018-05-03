## MNIST-Tensorflow 99.xx%

**I write a Tensorflow code for Classification of MNIST Data.**

You can get the results with following command:
```
python main.py --num_epoch 150
```

**This code has following features**

1. Data Augmentation is used (Training data: 50,000 -> 250,000)

2. 3x3 Conv with He_initializer, Strided Conv, batch_norm with decay rate 0.9, Max_Pooling are used

3. Activation Function is tf.nn.leaky_relu

4. Global Average Pooling is used instead of MLP

5. L2 Regularization Loss, Learning Rate Decay, Adam Optimization with beta1 = 0.5 are used

6. It Contains Codes for Tensorboard, Save, Restore 


## Enviroment
**- OS: Ubuntu 16.04**

**- Python 3.5**

**- Tensorflow-gpu version:  1.4.0rc2**

## Schematic of Network Architecture
![사진1](site 주소)

## Code

**1. Network**
```python
X_img = tf.placeholder(tf.float32, shape = [None,load_size,load_size,channels], name = "Input_IMG")
Y = tf.placeholder(tf.float32, shape = [None,num_classes], name = "Label")
is_training = tf.placeholder(tf.bool, name="is_training")

net = leaky(bn(conv(X_img, depth[0], name = "Conv1"), is_training = is_training, name = "bn1"))
net = leaky(bn(conv(net, depth[1], name = "Conv2"), is_training = is_training, name="bn2"))
net = strided_conv(net, name = "strided_conv1")
net = leaky(bn(conv(net, depth[2], name="Conv3"), is_training = is_training, name = "bn3"))
net = leaky(bn(conv(net, depth[3], name="Conv4"), is_training = is_training, name="bn4"))
net = strided_conv(net, name = "strided_conv2")
net = leaky(bn(conv(net, depth[4], name="Conv5"), is_training= is_training, name = "bn5"))
net = leaky(bn(conv(net, depth[5], name="Conv6"), is_training= is_training, name = "bn6"))
net = maxpool(net, name = "max_pool")

net = tf.reduce_mean(net, axis = (1,2), name ="GAP")
net = tf.layers.flatten(net)
net = dense(net,num_classes,name = "Dense1")
hypothesis = tf.nn.softmax(self.net)
