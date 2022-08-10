# 划分滑动窗口
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import torch
# def check_my_get_data():




def sliding_window(seq, win_width, win_step):
    """
    :param seq: [seq_len, dimension]
    :param win_width: int
    :param win_step: int
    :return:
    """
    return sliding_window_view(seq, win_width)[::win_step, :]


def prediction2supervised(seq, before_step, future_step, win_step):
    """
    transform the prediction problem to supervised learning
    :param seq: time series sequence, [seq_len, dimension]
    :param before_step: int, 7
    :param future_step: int, 1
    :param win_step: sliding step length
    :return: X [num_samples, before_step, dimension]
             Y [num_samples, future_step, dimension]
    """
    seq_array = sliding_window_view(seq, before_step+future_step, axis=0)[::win_step, :]
    X = seq_array[..., :before_step]
    Y = seq_array[..., -future_step:]
    X = np.transpose(X, [0, 2, 1])
    Y = np.transpose(Y, [0, 2, 1])
    return X, Y

