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
- OS: Ubuntu 16.04

- Python 3.5

- Tensorflow-gpu version:  1.4.0rc2

## Schematic of Network Architecture
![사진1](site 주소)

## Code

**1. Network**
```python
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
```

**2. Learning Rate Decay**
```python
self.global_step = tf.Variable(0, trainable=False, name ="global_step")
self.lr_decayed = lr*decay_rate**(self.global_step/decay_step)
```

**3. Optimization**
```python
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    self.optimization = tf.train.AdamOptimizer(beta1 = 0.5, learning_rate = self.lr_decayed).\
                        minimize(self.loss,global_step = self.global_step)
```

**4. Tensorboard Code**
```python
self.scalar_to_write = tf.placeholder(tf.float32)
self.loss_summary = tf.summary.scalar("Loss",self.scalar_to_write)
self.lr_summary = tf.summary.scalar("learing_rate", self.lr_decayed)

self.writer1 = tf.summary.FileWriter("./logs/total_loss")
self.writer2 = tf.summary.FileWriter("./logs/entropy_loss")
self.writer3 = tf.summary.FileWriter("./logs/reg_loss")
self.writer4 = tf.summary.FileWriter("./logs/lr")

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    self.writer3.add_graph(sess.graph)
    
    s = sess.run(self.loss_summary, feed_dict = {self.scalar_to_write: loss_total})
    self.writer1.add_summary(s,global_step = g)
    
    s = sess.run(self.loss_summary, feed_dict = {self.scalar_to_write: loss_entro})
    self.writer2.add_summary(s,global_step = g)
    
    s = sess.run(self.loss_summary, feed_dict = {self.scalar_to_write: loss_reg})
    self.writer3.add_summary(s,global_step = g)
    
    s = sess.run(self.lr_summary)
    self.writer4.add_summary(s,global_step = g)
```

## Augmentation Code
you can see Full Data Pipe line in data.py

```python
def augmentation(self,img,augment_size):
# In this code augment_size = 40, self.load_size = 32
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
```
## Result
**1. Loss and Learning rate
<table align='center'>
<tr align='center'>
<td> Animation of DCGAN </td>
<td> DCGAN After 100 Epoch </td>
</tr>
<tr>
<td><img src = 'images/MNIST_Animation.gif' height = '400px'>
<td><img src = 'images/Mnist_canvas100.png' height = '400px'>
</tr>
</table>
