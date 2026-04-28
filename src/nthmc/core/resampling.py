"""Bootstrap and jackknife resampling helpers."""

from __future__ import annotations

import numpy as np
import gvar as gv


def bin_data(data, bin_size, axis=0):
    """Bin data by averaging every bin_size samples along an axis."""
    shape = data.shape
    bin_length = shape[axis] // bin_size
    truncated_length = bin_length * bin_size
    truncated_data = np.take(data, range(truncated_length), axis=axis)
    new_shape = list(truncated_data.shape)
    new_shape[axis] = bin_length
    new_shape.insert(axis + 1, bin_size)
    binned_data = truncated_data.swapaxes(0, axis).reshape(new_shape).mean(axis=axis + 1)
    return binned_data.swapaxes(0, axis)


def bootstrap(data, samp_times, samp_size=None, axis=0, bin=1, seed=1984):
    """Bootstrap-resample data and return sample means plus sampled indices."""
    data = np.array(data)
    np.random.seed(seed)

    if bin > 1:
        data = bin_data(data, bin, axis=axis)

    n_conf = data.shape[axis]
    if samp_size is None:
        samp_size = n_conf
    conf_bs = np.random.choice(n_conf, (samp_times, samp_size), replace=True)
    bs_ls = np.take(data, conf_bs, axis=axis)
    bs_ls = np.mean(bs_ls, axis=axis + 1)

    return bs_ls, conf_bs


def jackknife(data, axis=0):
    """Jackknife-resample data by dropping one sample at a time."""
    data = np.array(data)
    n_conf = data.shape[axis]
    total_sum = np.sum(data, axis=axis, keepdims=True)
    return (total_sum - data) / (n_conf - 1)


def jk_ls_avg(jk_ls, axis=0):
    """Average jackknife samples and return gvar values."""
    jk_ls = np.array(jk_ls)
    if axis != 0:
        jk_ls = np.swapaxes(jk_ls, 0, axis)

    shape = np.shape(jk_ls)
    jk_ls = np.reshape(jk_ls, (shape[0], -1))

    n_sample = len(jk_ls)
    mean = np.mean(jk_ls, axis=0)

    if len(shape) == 1:
        sdev = np.std(jk_ls, axis=0) * np.sqrt(n_sample - 1)
        gv_ls = gv.gvar(mean, sdev)[0]
    else:
        cov = np.cov(jk_ls, rowvar=False) * (n_sample - 1)
        gv_ls = gv.gvar(mean, cov)
        gv_ls = np.reshape(gv_ls, shape[1:])

    return gv_ls


def jk_dic_avg(dic):
    """Average a dictionary of jackknife lists."""
    key_ls = list(dic.keys())
    l_dic = {key: len(dic[key][0]) for key in key_ls}
    n_conf = len(dic[key_ls[0]])

    conf_ls = []
    for n in range(n_conf):
        temp = []
        for key in dic:
            temp.append(list(dic[key][n]))
        conf_ls.append(sum(temp, []))

    gv_ls = list(jk_ls_avg(conf_ls))

    gv_dic = {}
    for key in l_dic:
        gv_dic[key] = []
        for _ in range(l_dic[key]):
            gv_dic[key].append(gv_ls.pop(0))

    return gv_dic


def bs_ls_avg(bs_ls, axis=0):
    """Average bootstrap samples and return gvar values."""
    bs_ls = np.array(bs_ls)
    if axis != 0:
        bs_ls = np.swapaxes(bs_ls, 0, axis)
    shape = np.shape(bs_ls)
    bs_ls = np.reshape(bs_ls, (shape[0], -1))

    mean = np.mean(bs_ls, axis=0)

    if len(shape) == 1:
        sdev = np.std(bs_ls, axis=0)
        gv_ls = gv.gvar(mean, sdev)[0]
    else:
        cov = np.cov(bs_ls, rowvar=False)
        gv_ls = gv.gvar(mean, cov)
        gv_ls = np.reshape(gv_ls, shape[1:])

    return gv_ls


def bs_dic_avg(dic):
    """Average a dictionary of bootstrap lists."""
    key_ls = list(dic.keys())
    l_dic = {key: len(dic[key][0]) for key in key_ls}
    n_conf = len(dic[key_ls[0]])

    conf_ls = []
    for n in range(n_conf):
        temp = []
        for key in dic:
            temp.append(list(dic[key][n]))
        conf_ls.append(sum(temp, []))

    gv_ls = list(bs_ls_avg(conf_ls))

    gv_dic = {}
    for key in l_dic:
        gv_dic[key] = []
        for _ in range(l_dic[key]):
            gv_dic[key].append(gv_ls.pop(0))

    return gv_dic


def gv_ls_to_samples_corr(gv_ls, n_samp):
    """Convert correlated gvar values to Gaussian samples."""
    mean = np.array([gv_value.mean for gv_value in gv_ls])
    cov = gv.evalcov(gv_ls)
    rng = np.random.default_rng()
    return rng.multivariate_normal(mean, cov, size=n_samp)


def gv_dic_to_samples_corr(gv_dic, n_samp):
    """Convert a gvar dictionary to correlated Gaussian samples."""
    l_dic = {key: len(gv_dic[key]) for key in gv_dic}

    flatten_ls = []
    for key in gv_dic:
        flatten_ls.append(list(gv_dic[key]))
    flatten_ls = sum(flatten_ls, [])

    samp_all = gv_ls_to_samples_corr(flatten_ls, n_samp)
    samp_all = list(np.swapaxes(samp_all, 0, 1))

    samp_dic = {}
    for key in l_dic:
        samp_ls = []
        for _ in range(l_dic[key]):
            samp_ls.append(samp_all.pop(0))
        samp_dic[key] = np.swapaxes(np.array(samp_ls), 0, 1)

    return samp_dic
