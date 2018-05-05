# coding: utf-8

"""
モデルの評価指針となるメトリクスを計算するモジュールです。

分類器でよく用いられるメトリックを計算します。まずクラスごとに4つの統計量を出します。

    True  Ppsitive   : 正の予測が正解だった数。
    True  Negative   : 負の予測が正解だった数。
    False Positive   : 正の予測が不正解だった数。
    False Negative   : 負の予測が不正解だった数。

メトリクスは以下の4つを計算します。

    Accuracy  : 正解率。(TP+TN)/(TP+TN+FP+FN)
    Precition : 適合率。予測が正のもののうち、正解も正だった割合。TP/(TP+FP)
    Recall    : 再現率。正解が正のもののうち、予測が正だった割合。TP/(TP+FN)
    F-Measure : F値。適合率と再現率の調和平均。

Accuracyは正解した割合です。わかりやすいので精度として扱われることの多い値です。
ただ、不正解には「負を正としてしまった(FP)」か、「正を負としてしまった(FN)」の二種類があります。
Accuracyではどちらで間違えているのかがわかりません。

2つの間違いの違いはよく病気の誤診で説明されます。
    - FP 「病気じゃないのに病気と診断した」
    - FN 「病気だったのに病気じゃないと診断した」
両者は明らかに深刻さが異なります。
前者は「間違えました」で済まされるかもしれませんが、後者は、もし重い病気であれば命に関わりますね。

そこで、適合率と再現率を見ます。

適合率は、上記例であれば「病気だと診断した時に、本当に病気だった割合」です。
再現率は、上記例であれば「病気の人を、病気と予測できた割合」です。

病気の患者を漏れなく発見することが重要な場合、再現率が重要となります。
適合率と再現率はトレードオフになることが多いです。
両者のどちらが重要なのかは、ケースによります。誤診の例では再現率でしたが、必ずしも再現率が重要とは限りません。
また、両者のバランスをとった指針としてF値というものがあります。

それぞれ、マクロ平均とマイクロ平均を計算します。

    Macro Average : クラスごとに計算した値の平均
    Micro Average : 全クラスの結果を合計して計算した値

マイクロ平均は全体の結果なので計算しやすくわかりやすいですが、データ数の多いクラスの重みが大きくなります。
クラスごとに出現頻度が異なるケースで、それを加味した値を計算したい場合はマイクロ平均での結果が必要ですが、
クラスごとに平等に値を計算したい場合は、マクロ平均を用います。
"""

import six
import tensorflow as tf

def merge(xs):
    y = dict()
    for x in xs:
        y.update(x)
    return y

def calculate(labels, classes, num_classes, add_summary=True):
    cm = tf.confusion_matrix(labels, classes, num_classes=num_classes, dtype=tf.float32)
    ln = tf.reduce_sum(cm)
    tp = tf.diag_part(cm)
    fp = tf.reduce_sum(cm, axis=1) - tp
    fn = tf.reduce_sum(cm, axis=0) - tp
    tn = ln - tp - fp - fn
    eps = tf.convert_to_tensor(1e-7)
    metric_ops = merge([
        micro_metrics(tp, fp, tn, fn, eps),
        macro_metrics(tp, fp, tn, fn, num_classes, eps),
    ])

    if add_summary:
        for name, metric in six.iteritems(metric_ops):
            tf.summary.scalar(name, metric[1])

    return metric_ops

def micro_metrics(tp, fp, tn, fn, eps):
    with tf.name_scope('micro_average'):
        tp_sum = tf.reduce_sum(tp)
        fp_sum = tf.reduce_sum(fp)
        tn_sum = tf.reduce_sum(tn)
        fn_sum = tf.reduce_sum(fn)
        accuracy  = (tp_sum + tn_sum) / (tp_sum + fp_sum + tn_sum + fn_sum)
        precision = tp_sum / (tp_sum + fp_sum + eps)
        recall    = tp_sum / (tp_sum + fn_sum + eps)
        f_measure = 2 * precision * recall / (precision + recall + eps)

        return {
            'micro_average/accuracy' : tf.metrics.mean_tensor(accuracy),
            'micro_average/precision': tf.metrics.mean_tensor(precision),
            'micro_average/recall'   : tf.metrics.mean_tensor(recall),
            'micro_average/f_measure': tf.metrics.mean_tensor(f_measure),
        }

def macro_metrics(tp, fp, tn, fn, num_classes, eps):
    with tf.name_scope('macro_average'):
        accuracies = (tp + tn) / (tp + fp + tn + fn)
        precisions = tp / (tp + fp + eps)
        recalls    = tp / (tp + fn + eps)
        f_measures = 2 * precisions * recalls / (precisions + recalls + eps)

        accuracy  = tf.reduce_mean(accuracies)
        precision = tf.reduce_mean(precisions)
        recall    = tf.reduce_mean(recalls)
        f_measure = tf.reduce_mean(f_measures)
        
        metrics = {
            'macro_average/accuracy' : tf.metrics.mean_tensor(accuracy),
            'macro_average/precision': tf.metrics.mean_tensor(precision),
            'macro_average/recall'   : tf.metrics.mean_tensor(recall),
            'macro_average/f_measure': tf.metrics.mean_tensor(f_measure),
        }

    for i in range(num_classes):
        family = 'macro_class_{}'.format(i)
        with tf.name_scope(family):
            metrics = merge([
                metrics,
                {
                    family+'/accuracy' : tf.metrics.mean_tensor(accuracies[i]),
                    family+'/precision': tf.metrics.mean_tensor(precisions[i]),
                    family+'/recall'   : tf.metrics.mean_tensor(recalls[i]),
                    family+'/f_measure': tf.metrics.mean_tensor(f_measures[i]),
                },
            ])

    return metrics
