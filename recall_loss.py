import tensorflow as tf
from keras import backend as K

def binary_recall_specificity(y_true, y_pred, recall_weight, spec_weight, from_logits=False):
    y_pred_continuous = y_pred

    #Compute from logits
    if from_logits:
        y_pred_continuous = tf.math.sigmoid(y_pred)

    # Create binary masks for TN, TP, FP, FN
    TP = tf.reduce_sum(y_true * y_pred_continuous)
    FP = tf.reduce_sum((1 - y_true) * y_pred_continuous)
    FN = tf.reduce_sum(y_true * (1 - y_pred_continuous))
    TN = tf.reduce_sum((1 - y_true) * (1 - y_pred_continuous))

    # Cast to float for mathematical operations
    TN = tf.cast(TN, tf.float32)
    FP = tf.cast(FP, tf.float32)
    TP = tf.cast(TP, tf.float32)
    FN = tf.cast(FN, tf.float32)

    # Calculate specificity and recall
    specificity = tf.reduce_sum(TN) / (tf.reduce_sum(TN) + tf.reduce_sum(FP) + K.epsilon())
    recall = tf.reduce_sum(TP) / (tf.reduce_sum(TP) + tf.reduce_sum(FN) + K.epsilon())

    return 1.0 - (recall_weight*recall + spec_weight*specificity)

def custom_loss(recall_weight, spec_weight, from_logits=False):

    def recall_spec_loss(y_true, y_pred):
        return binary_recall_specificity(y_true, y_pred, recall_weight, spec_weight, from_logits)

    # Returns the (y_true, y_pred) loss function
    return recall_spec_loss