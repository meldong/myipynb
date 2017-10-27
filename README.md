# tf-slim
tf-slim-mnist MNIST tutorial with Tensorflow Slim (tf.contrib.slim) a lightweight library over Tensorflow.
[tensorflow/tensorflow/contrib/slim/](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)

[tensorflow/models/research/slim/](https://github.com/tensorflow/models/tree/master/research/slim)

[tensorflow/models/research/inception/inception/slim/](https://github.com/tensorflow/models/tree/master/research/inception/inception/slim)

[mnuke/tf-slim-mnist](https://github.com/mnuke/tf-slim-mnist) is a good ipython notebook about it

Chinese resource
[如何评价tf-slim库](https://www.zhihu.com/question/53113870)
[【Tensorflow】辅助工具篇——tensorflow slim(TF-Slim)介绍](http://blog.csdn.net/mao_xiao_feng/article/details/73409975)

## Setting up datarun `python datasets/download_and_convert_mnist.py` to create [train, test].tfrecords files containing MNIST databy default (unless you specify `--directory`) they will be put into /tmp/mnist
## RunningRun the training, validation, and tensorboard concurrently. The results of the training and validation should show up in tensorboard.
### Running the trainingrun `mnist_train.py` which will read train.tfrecords using an input queue and output its model checkpoints, and summaries to the log directory (you can specify it with `--log_dir`)
### Running the validationrun `mnist_eval.py` which will read test.tfrecords using an input queue, and also read the train models checkpoints from `log/train` (by default). It will then load the model at that checkpoint and run it on the testing examples, outputting the summaries and log to its own folder `log/eval` (you can specify it with `--log_dir`)
### Running tensorboardTensorboard allows you to keep track of your training in a nice and visual way. It will read the logs from the training and validation and should update on its own though you may have to refresh the page manually sometimes.
Make sure both training and validation output their summaries to one log directory and preferably under their own folder. Run `tensorboard --logdir=log` (replace log with your own log folder if you changed it).
If each process has its own folder then train and validation should have their own colour and checkbox
## Notes
### Woah, data input seems pretty different from what it used to be
Tensorflow has really changed the way they're doing data input (for the better!) and though the new way seems pretty complicated (with queue runners etc...) it isn't that bad and can potentially make everything much faster, better.
I'm trying to keep up with all the changes but if something seems off to you, then please open an issue or create a pull request!
### Where did you get all those files in `/dataset` ?
I took those files from the tensorflow/models repo in the tensorflow slim folder [here](https://github.com/tensorflow/models/blob/master/slim/). I modified `download_and_convert_mnist.py` just a little so it can be run as a standalone program, and took only the files you need to run a lenet architechture for mnist.
### How do I do more than MNIST?
Modify the model file with whatever model you want, change the data input (maybe look at the [datasets already available in slim](https://github.com/tensorflow/models/tree/master/slim/datasets)). 
