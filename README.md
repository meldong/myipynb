# tf-slim
tf-slim-mnist MNIST tutorial with Tensorflow Slim (tf.contrib.slim) a lightweight library over Tensorflow.

### Official resource

[tensorflow/tensorflow/contrib/slim/](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)

[tensorflow/models/research/slim/](https://github.com/tensorflow/models/tree/master/research/slim)

[tensorflow/models/research/inception/inception/slim/](https://github.com/tensorflow/models/tree/master/research/inception/inception/slim)

### English resource

[mnuke/tf-slim-mnist](https://github.com/mnuke/tf-slim-mnist)

### Chinese resource
[如何评价tf-slim库](https://www.zhihu.com/question/53113870)

[【Tensorflow】辅助工具篇——tensorflow slim(TF-Slim)介绍](http://blog.csdn.net/mao_xiao_feng/article/details/73409975)

# Tensorflow Object Detection API

### Woah, data input seems pretty different from what it used to be
Tensorflow has really changed the way they're doing data input (for the better!) and though the new way seems pretty complicated (with queue runners etc...) it isn't that bad and can potentially make everything much faster, better.
I'm trying to keep up with all the changes but if something seems off to you, then please open an issue or create a pull request!
### Where did you get all those files in `/dataset` ?
I took those files from the tensorflow/models repo in the tensorflow slim folder [here](https://github.com/tensorflow/models/blob/master/slim/). I modified `download_and_convert_mnist.py` just a little so it can be run as a standalone program, and took only the files you need to run a lenet architechture for mnist.
### How do I do more than MNIST?
Modify the model file with whatever model you want, change the data input (maybe look at the [datasets already available in slim](https://github.com/tensorflow/models/tree/master/slim/datasets)). 
