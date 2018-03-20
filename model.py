import tensorflow as tf
import numpy as np
import utility
import os
import cv2
import random


TRAIN_ITERATIONS = 10000
BATCH_SIZE = 4
NUM_CLASSES = 2
EPOCHS = 50
H, W = 480, 640
FINETUNE = False
IMAGE_PATHS = 'train_image'
MASK_PATHS = 'train_mask'
VAL_IMAGE_PATH = 'val_image'
VAL_MASK_PATH = 'val_mask'
OUTPUT = 'outputs/'
MODEL = 'model/'


def conv_with_bn(x, no_of_filters, kernel_size, training, strides=[1, 1], activation=tf.nn.relu, use_bias=True, name=None):
    conv = tf.layers.conv2d(x, no_of_filters, kernel_size, strides, padding='SAME', activation=activation,
                            use_bias=use_bias, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=name)
    conv = tf.layers.batch_normalization(conv, training=training)

    return conv


def trans_conv_with_bn(x, no_of_filters, kernel_size, training, strides=[2, 2], activation=tf.nn.relu, use_bias=True, name=None):
    conv = tf.layers.conv2d_transpose(x, no_of_filters, kernel_size, strides, padding='SAME', activation=activation,
                                      use_bias=use_bias, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=name)
    conv = tf.layers.batch_normalization(conv, training=training)
    return conv


def xentropy_loss(mask, prediction, num_classes=2):
    """Calculates softmax cross entropy loss."""
    mask = tf.reshape(tf.one_hot(mask, num_classes, axis=3), [-1, num_classes])
    prediction = tf.reshape(prediction, shape=[-1, num_classes])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=prediction, labels=mask))
    return loss


def calculate_iou(mask, prediction, name=None, batch_size=5, num_classes=2):
    """Calculates mean IoU, the default eval metric of segmentation."""

    mask = tf.reshape(tf.one_hot(mask, num_classes, axis=3),
                      [batch_size, -1, num_classes])
    prediction = tf.reshape(prediction, shape=[batch_size, -1, num_classes])
    iou, update_op = tf.metrics.mean_iou(
        tf.argmax(prediction, 2), tf.argmax(mask, 2), num_classes, name=name)

    return iou, update_op


def inference(image_tensor, is_training):
    """Runs image through the network and returns predicted mask."""

    print('Building Network for Inference...')

    conv0 = conv_with_bn(image_tensor, 64, [3, 3], is_training, name='conv0')
    down0 = conv_with_bn(conv0, 64, [3, 3], is_training, [2, 2], name='down0')

    conv1 = conv_with_bn(down0, 128, [3, 3], is_training, name='conv1')
    down1 = conv_with_bn(conv1, 128, [3, 3], is_training, [2, 2], name='down1')

    conv2 = conv_with_bn(down1, 256, [3, 3], is_training, name='conv2')
    down2 = conv_with_bn(conv2, 256, [3, 3], is_training, [2, 2], name='down2')

    conv3 = conv_with_bn(down2, 512, [3, 3], is_training, name='conv3')
    down3 = conv_with_bn(conv3, 512, [3, 3], is_training, [2, 2], name='down3')

    up3 = trans_conv_with_bn(down3, 512, [3, 3], is_training, name='up3')
    unconv3 = conv_with_bn(up3, 512, [3, 3], is_training, name='unconv3')

    up2 = trans_conv_with_bn(unconv3, 256, [3, 3], is_training, name='up2')
    unconv2 = conv_with_bn(up2, 256, [3, 3], is_training, name='unconv2')

    up1 = trans_conv_with_bn(unconv2, 128, [3, 3], is_training, name='up1')
    unconv1 = conv_with_bn(up1, 128, [3, 3], is_training, name='unconv1')

    up0 = trans_conv_with_bn(unconv1, 64, [3, 3], is_training, name='up0')
    unconv0 = conv_with_bn(up0, 64, [3, 3], is_training, name='unconv0')

    pred = conv_with_bn(unconv0, NUM_CLASSES, [
                        3, 3], is_training, activation=None, use_bias=False, name='pred')

    print('Done, network built.')
    return pred


def train(train_image_paths, train_mask_paths, val_image_path, val_mask_path, lr=1e-4):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        data, init_op = utility.data_batch(
            train_image_paths, train_mask_paths, augment=True, batch_size=BATCH_SIZE)

        val_data, val_init_op = utility.data_batch(
            val_image_path, val_mask_path, augment=False, batch_size=3)

        image_tensor, mask_tensor = data
        val_image_tensor, val_mask_tensor = val_data

        image_placeholder = tf.placeholder(
            tf.float32, shape=[None, H, W, 3])
        mask_placeholder = tf.placeholder(
            tf.int32, shape=[None, H, W, 1])

        training_flag = tf.placeholder(tf.bool)

        logits = inference(image_placeholder, training_flag)
        cost = xentropy_loss(mask_placeholder, logits)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train = optimizer.minimize(cost)

        iou_metric, iou_update = calculate_iou(
            mask_placeholder, logits, "iou_metric", BATCH_SIZE, NUM_CLASSES)

        running_vars = tf.get_collection(
            tf.GraphKeys.LOCAL_VARIABLES, scope="iou_metric")

        running_vars_init = tf.variables_initializer(var_list=running_vars)

        seg_image = tf.argmax(logits, axis=3)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('Training with learning rate: ', lr)
            if FINETUNE:
                saver.restore(sess, tf.train.latest_checkpoint(MODEL))
            else:
                sess.run(tf.global_variables_initializer())
            for epoch in range(0, EPOCHS):
                try:

                    total_loss, total_iou = 0, 0
                    for step in range(0, TRAIN_ITERATIONS):
                        print('Step:', step)
                        sess.run([init_op, val_init_op])
                        sess.run(running_vars_init)
                        train_image, train_mask = sess.run(
                            [image_tensor, mask_tensor])
                        train_feed_dict = {
                            image_placeholder: train_image,
                            mask_placeholder: train_mask,
                            training_flag: True
                        }

                        _, loss, update_iou, pred_mask = sess.run(
                            [train, cost, iou_update, seg_image], feed_dict=train_feed_dict)

                        iou = sess.run(iou_metric)
                        total_loss += loss
                        total_iou += iou
                except tf.errors.OutOfRangeError:
                    pass
                finally:
                    sess.run(running_vars_init)
                    val_iou, val_loss = 0, 0
                    val_image, val_mask = sess.run(
                        [val_image_tensor, val_mask_tensor])
                    val_feed_dict = {
                        image_placeholder: val_image,
                        mask_placeholder: val_mask,
                        training_flag: False
                    }
                    update_iou, pred_mask, val_loss = sess.run(
                        [iou_update, seg_image, cost], feed_dict=val_feed_dict)
                    val_iou = sess.run(iou_metric)

                    print('Train loss: ', total_loss / TRAIN_ITERATIONS,
                          'Train iou: ', total_iou / TRAIN_ITERATIONS)
                    print('Val. loss: ', val_loss, 'Val. iou: ', val_iou)
                    saver.save(sess, MODEL + 'model.ckpt', global_step=epoch)
                    print('Starting epoch: ', epoch)


def infer(image_paths):
    image_tensor = tf.placeholder(tf.float32, [None, H, W, 3])
    training_flag = tf.placeholder(tf.bool)
    logits = inference(image_tensor, training_flag)
    seg_image = tf.argmax(logits, axis=3)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL))

        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path, 1)
            h, w = image.shape[:2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (W, H))
            image = np.expand_dims(image, axis=0)
            feed_dict = {image_tensor: image,
                         training_flag: True}
            mask = sess.run(seg_image, feed_dict=feed_dict)
            mask = np.squeeze(mask)
            cv2.imwrite(str(i) + '.png', mask * 255)
            mask = cv2.imread(str(i) + '.png', 1)
            mask = cv2.resize(mask, (w, h))
            cv2.imwrite(str(i) + '.png', mask)


if __name__ == '__main__':

    image_paths = [os.path.join(os.getcwd(), IMAGE_PATHS,
                                x) for x in os.listdir(IMAGE_PATHS) if x.endswith('.png')]
    mask_paths = [os.path.join(os.getcwd(), MASK_PATHS, 'mask-' + os.path.basename(x))
                  for x in image_paths]
    val_image_paths = [os.path.join(os.getcwd(), VAL_IMAGE_PATH, x)
                       for x in os.listdir(VAL_IMAGE_PATH) if x.endswith('.png')]
    val_mask_paths = [os.path.join(
        os.getcwd(), VAL_MASK_PATH, 'mask-' + os.path.basename(x)) for x in val_image_paths]

    indexes = [x for x in range(len(image_paths))]
    random.shuffle(indexes)
    image_paths = [image_paths[index] for index in indexes]
    mask_paths = [mask_paths[index] for index in indexes]

    #infer_paths = ['test_image_modded/' + x for x in os.listdir(
    #    'test_image_modded') if x.endswith('.jpg')]
    #print(infer_paths)
    #infer(infer_paths)
    print('Training inititated...')
    train(image_paths, mask_paths, val_image_paths, val_mask_paths)
