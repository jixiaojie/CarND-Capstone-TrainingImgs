import os.path
from glob import glob
import cv2
import numpy as np
import cnn_model
import tensorflow.contrib.slim as slim
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# uint8 UNKNOWN=4
# uint8 GREEN=2
# uint8 YELLOW=1
# uint8 RED=0


def get_traindata(img_path, image_shape=None):

    image_paths = glob(img_path)
    images = []
    image_labels = []

    images_count = {	'unknown':0,
			'green':0,
			'yellow':0,
			'red':0,
			'unclassified':0 }

    print "Find %s images"%(len(image_paths)), " Loading data ..."

    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_8)

        if image_shape:
            images.append(cv2.resize(image, image_shape))
        else:
            images.append(image)
        
        if 'unknown' in path.lower():
            image_labels.append([0, 0, 0, 1])
            images_count['unknown'] += 1

        elif 'green' in path.lower():
            image_labels.append([0, 0, 1, 0])
            images_count['green'] += 1

        elif 'yellow' in path.lower():
            image_labels.append([0, 1, 0, 0])
            images_count['yellow'] += 1

        elif 'red' in path.lower():
            image_labels.append([1, 0, 0, 0])
            images_count['red'] += 1

        else:
            images_count['unclassified'] += 1


    for key in images_count.keys():
        print key, ": ", images_count[key] 

    return shuffle(images, image_labels)



def train(X_train, X_test, Y_train, Y_test, epoch = 5):

    # tf Graph input i[-1, 128, 64, 1]
    x = tf.placeholder(tf.float32, [None, 128, 64, 3])
    y_ = tf.placeholder(tf.float32, [None, 4]) #answer
    keep_prob = tf.placeholder(tf.float32)

    # Predict
    y = cnn_model.CNN(x)

    with tf.name_scope("LOSS"):
        loss = slim.losses.softmax_cross_entropy(y,y_)

    with tf.name_scope("ADAM"):
        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0)
        learning_rate = 1e-4
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.name_scope("ACC"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_size = len(Y_train)
    batch_size = 50
    total_batch = int(train_size / batch_size)

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    num = 1


    for i in range(epoch):
        # Random shuffling
        X_train, Y_train = shuffle(X_train, Y_train)

        # Loop over all batches
        for i in range(total_batch):

            # Compute the offset of the current minibatch in the data.
            offset = (i * batch_size) % (train_size)
            batch_x = X_train[offset:(offset + batch_size)]
            batch_y = Y_train[offset:(offset + batch_size)]

            _, train_accuracy = sess.run([train_step, accuracy] , feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.8})
            print num, " train_accuracy: ", train_accuracy

            num += 1
        test_accuracy = sess.run([accuracy] , feed_dict={x: X_test, y_: Y_test, keep_prob: 1.0})
        print num, "-----Test_accuracy: ", test_accuracy


    save_path = saver.save(sess, 'model/model.ckpt')






if __name__ == '__main__':

    # choose a data set.
    # 1.SimulatorTrack1_Classified_Imgs 
    imgs_data, imgs_label = get_traindata('./SimulatorTrack1_Classified_Imgs/*/*.jpg',(64, 128))
    X_train, X_test, Y_train, Y_test = train_test_split(imgs_data, imgs_label, test_size=0.2, random_state=0)

    # 2.RealCarTrack_Unclassified_Imgs , Unclassified data , Don't use to training
    #imgs_data = get_traindata('./RealCarTrack_Unclassified_Imgs/*.jpg')


    # 3.FromNet_Traffic_Light_Imgs , a small traffic light data set
    #X_train, Y_train = get_traindata('./FromNet_Traffic_Light_Imgs/training/*/*.jpg',(64, 128))
    #X_test, Y_test = get_traindata('./FromNet_Traffic_Light_Imgs/test/*/*.jpg',(64, 128))
   

    print len(X_train), len(Y_train) 
    print len(X_test), len(Y_test) 
    
    #cv2.imwrite('test01.jpg', imgs_data[0])
    
    train(X_train, X_test, Y_train, Y_test, 5)

