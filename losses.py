# vim: expandtab:ts=4:sw=4
import tensorflow as tf


def _pdist(a, b=None):
    sq_sum_a = tf.reduce_sum(tf.square(a), reduction_indices=[1])
    if b is None:
        return -2 * tf.matmul(a, tf.transpose(a)) + \
               tf.reshape(sq_sum_a, (-1, 1)) + tf.reshape(sq_sum_a, (1, -1))
    sq_sum_b = tf.reduce_sum(tf.square(b), reduction_indices=[1])
    return -2 * tf.matmul(a, tf.transpose(b)) + \
           tf.reshape(sq_sum_a, (-1, 1)) + tf.reshape(sq_sum_b, (1, -1))


def triplet_loss(features, labels, create_summaries=True):
    """Softmargin triplet loss.

    See::

        Hermans, Beyer, Leibe: In Defense of the Triplet Loss for Person
        Re-Identification. arXiv, 2017.

    Parameters
    ----------
    features : tf.Tensor
        A matrix of shape NxM that contains the M-dimensional feature vectors
        of N objects (floating type).
    labels : tf.Tensor
        The one-dimensional array of length N that contains for each feature
        the associated class label (integer type).
    create_summaries : Optional[bool]
        If True, creates summaries to monitor training behavior.

    Returns
    -------
    tf.Tensor
        A scalar loss tensor.

    """
    eps = tf.constant(1e-5, tf.float32)
    nil = tf.constant(0., tf.float32)
    almost_inf = tf.constant(1e+10, tf.float32)

    squared_distance_mat = _pdist(features)
    distance_mat = tf.sqrt(tf.maximum(nil, eps + squared_distance_mat))
    label_mat = tf.cast(tf.equal(
        tf.reshape(labels, (-1, 1)), tf.reshape(labels, (1, -1))), tf.float32)

    positive_distance = tf.reduce_max(label_mat * distance_mat, axis=1)
    negative_distance = tf.reduce_min(
        (label_mat * almost_inf) + distance_mat, axis=1)
    loss = tf.nn.softplus(positive_distance - negative_distance)

    if create_summaries:
        fraction_invalid_pdist = tf.reduce_mean(
            tf.cast(tf.less_equal(squared_distance_mat, -eps), tf.float32))
        tf.summary.scalar("fraction_invalid_pdist", fraction_invalid_pdist)

        fraction_active_triplets = tf.reduce_mean(
            tf.cast(tf.greater_equal(loss, 1e-5), tf.float32))
        tf.summary.scalar("fraction_active_triplets", fraction_active_triplets)

        embedding_squared_norm = tf.reduce_mean(
            tf.reduce_sum(tf.square(features), axis=1))
        tf.summary.scalar("mean squared feature norm", embedding_squared_norm)

        mean_distance = tf.reduce_mean(distance_mat)
        tf.summary.scalar("mean feature distance", mean_distance)

        mean_positive_distance = tf.reduce_mean(positive_distance)
        tf.summary.scalar("mean positive distance", mean_positive_distance)

        mean_negative_distance = tf.reduce_mean(negative_distance)
        tf.summary.scalar("mean negative distance", mean_negative_distance)

    return tf.reduce_mean(loss)

