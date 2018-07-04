import numpy as np
import tensorflow as tf
from market1501 import Market1501
from losses import triplet_loss

# from tensorflow.contrib.framework.python.ops import add_arg_scope

tf.app.flags.DEFINE_string('inputs_path', '/DML/Market-1501-v15.09.15', 'path to input')
tf.app.flags.DEFINE_string('outputs_path', './graphs/vgg/DML', 'path to output')
tf.app.flags.DEFINE_string('dataset', 'Market1501', 'name of dataset')
tf.app.flags.DEFINE_integer('batch_size', 6, 'batch size')
tf.app.flags.DEFINE_integer('max_epochs', 1000, 'num of epochs')
tf.app.flags.DEFINE_integer('embedding_length', 128, 'embedding length')

FLAGS = tf.app.flags.FLAGS


# @add_arg_scope
def vgg_16(inputs, embedding_length):
    #TODO: implement VGG_16
    with tf.name_scope('vgg_16'):
        output = None
    return output


def read_and_prepare_data():
    dataset = Market1501(FLAGS.inputs_path, num_validation_y=0.1, seed=1234)
    X_train, Y_train, _ = dataset.read_train()
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val, Y_val, _ = dataset.read_validation()
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    return X_train, Y_train, X_val, Y_val


def create_opt(net, labels, global_step, decay_steps=30000, learning_rate_decay_factor=0.5, learning_rate=0.01):
    lr = tf.train.exponential_decay(learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True,
                                    name='exponential_decay_learning_rate')

    features = tf.squeeze(net, axis=[1, 2])
    dml_loss = triplet_loss(features, labels, create_summaries=True) # TODO: use DML loss
    l2_loss = tf.losses.get_regularization_loss()
    loss = dml_loss + l2_loss
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
    return opt, loss, lr


def predict(net, y):
    squeezed = tf.squeeze(net, axis=[1, 2])
    #TODO: implement predict
    pred, accuracy = None, None
    return pred, accuracy


def main():
    # placeholders and variables
    inputs = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, None, None, 3])
    y = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    is_training = tf.placeholder(tf.bool)
    gstep = tf.Variable(0, dtype=tf.int32,
                        trainable=False, name='global_step')

    # read dataset
    X_train, Y_train, X_val, Y_val = read_and_prepare_data()

    # create network and opt
    net = vgg_16(inputs, FLAGS.embedding_length)
    labels = y
    opt, loss, lr = create_opt(net, y, gstep)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', lr)
    pred, acc = predict(net, labels)

    # summary
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.outputs_path + '/train', tf.get_default_graph())
    test_writer = tf.summary.FileWriter(FLAGS.outputs_path + '/test')
    saver = tf.train.Saver()

    # running network
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        step = gstep.eval()
        for epoch in range(FLAGS.max_epochs):
            print('Epoch: {}'.format(epoch))
            permutation = np.random.permutation(Y_train.shape[0])
            x_train = X_train[permutation]
            y_train = Y_train[permutation]
            for batch in range(X_train.shape[0] / FLAGS.batch_size):
                b = x_train[batch * FLAGS.batch_size: (batch + 1) * FLAGS.batch_size]

                l_b = y_train[batch * FLAGS.batch_size: (batch + 1) * FLAGS.batch_size]
                _, loss_value, _merged = sess.run([opt, loss, merged], feed_dict={inputs: b, y: l_b, is_training: True})
                if batch % 100 == 0:
                    print("step {} - loss: {}".format(step, loss_value))
                step += 1
                train_writer.add_summary(_merged)

            if (epoch % 20) == 0:
                saver.save(sess,FLAGS.outputs_path+"/weights",step)


if __name__ == "__main__":
    main()
