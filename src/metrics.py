# coding: utf-8

import tensorflow as tf

def micro_metrics(tp, fp, tn, fn, eps):
    tp_sum = tf.reduce_sum(tp)
    fp_sum = tf.reduce_sum(fp)
    tn_sum = tf.reduce_sum(tn)
    fn_sum = tf.reduce_sum(fn)
    accuracy  = (tp_sum + tn_sum) / (tp_sum + fp_sum + tn_sum + fn_sum)
    precision = tp_sum / (tp_sum + fp_sum + eps)
    recall    = tp_sum / (tp_sum + fn_sum + eps)
    f_measure = 2 * precision * recall / (precision + recall + eps)
    
    family = 'micro_metrics'
    tf.summary.scalar('accuracy' , accuracy , family=family)
    tf.summary.scalar('precision', precision, family=family)
    tf.summary.scalar('recall'   , recall   , family=family)
    tf.summary.scalar('f_measure', f_measure, family=family)

def macro_metrics(tp, fp, tn, fn, num_classes, eps):
    accuracies = (tp + tn) / (tp + fp + tn + fn)
    precisions = tp / (tp + fp + eps)
    recalls    = tp / (tp + fn + eps)
    f_measures = 2 * precisions * recalls / (precisions + recalls + eps)
    
    for i in range(num_classes):
        family = 'metric_{}'.format(i)
        tf.summary.scalar('accuracy' , accuracies[i], family=family)
        tf.summary.scalar('precision', precisions[i], family=family)
        tf.summary.scalar('recall'   , recalls[i]   , family=family)
        tf.summary.scalar('f_measure', f_measures[i], family=family)

    accuracy  = tf.reduce_mean(accuracies)
    precision = tf.reduce_mean(precisions)
    recall    = tf.reduce_mean(recalls)
    f_measure = tf.reduce_mean(f_measures)

    family = 'macro_metrics'
    tf.summary.scalar('accuracy' , accuracy , family=family)
    tf.summary.scalar('precision', precision, family=family)
    tf.summary.scalar('recall'   , recall   , family=family)
    tf.summary.scalar('f_measure', f_measure, family=family)

def add_summary(labels, classes, num_classes):
    cm = tf.confusion_matrix(labels, classes, num_classes=num_classes, dtype=tf.float32)
    ln = tf.reduce_sum(cm)
    tp = tf.diag_part(cm)
    fp = tf.reduce_sum(cm, axis=1) - tp
    fn = tf.reduce_sum(cm, axis=0) - tp
    tn = ln - tp - fp - fn
    eps = tf.convert_to_tensor(1e-7)
    micro_metrics(tp, fp, tn, fn, eps)
    macro_metrics(tp, fp, tn, fn, num_classes, eps)
