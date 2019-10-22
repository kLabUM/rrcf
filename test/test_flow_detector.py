# -*- coding: utf-8 -*-

import os
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import rrcf

# Read data
def data_explore(is_show=True):
    abs_path = os.path.abspath(__file__)
    data_paths = abs_path.strip().split("/")[:-1]
    data_paths.extend(["..", "resources", "website_flow.csv"])
    data_path = os.path.join(*data_paths)
    if not data_path.startswith("/"):
        data_path = "/" + data_path
    data_frame = pd.read_csv(data_path)
    data_frame.sort_values(["time"], inplace=True)
    data_frame["time"] = pd.to_datetime(data_frame["time"], unit="s")
    data_frame.set_index(["time"], inplace=True)
    print(data_frame)

    if is_show:
        fig, ax = plt.subplots(3, figsize=(12, 7))
        data_frame["bytes"].plot(ax=ax[0])
        data_frame["request"].plot(ax=ax[1])
        data_frame["num"].plot(ax=ax[2])
        ax[0].set_ylabel("bytes")
        ax[1].set_ylabel("request")
        ax[2].set_ylabel("num")
        plt.tight_layout()
        plt.show()

    return data_frame


def flow_detector():
    # Set tree parameters
    num_trees = 10
    tree_size = 6000
    history_queue = deque([], maxlen=tree_size)

    n = 5000
    data_frame = data_explore(is_show=False)
    history_samples = data_frame[:n]
    testing_samples = data_frame[n:]

    anomaly_score = pd.Series(0.0, index=data_frame.index)

    forest = []
    for _ in range(num_trees):
        tree = rrcf.RCTree(history_samples.values, index_labels=list(history_samples.index))
        forest.append(tree)

    for idx in history_samples.index:
        cur_point = history_samples.ix[idx].values
        avg_codisp = 0.0
        for tree in forest:
            avg_codisp += tree.codisp(idx) / num_trees
        print('CoDisp for point ({index}) is {avg_codisp}'.format(index=idx, avg_codisp=avg_codisp))
        anomaly_score[idx] = avg_codisp

    for idx in list(history_samples.index):
        history_queue.append(idx)

    for idx in testing_samples.index:
        cur_point = testing_samples.ix[idx].values
        if len(history_queue) == tree_size:
            old_index = history_queue.popleft()
        else:
            old_index = None
        history_queue.append(idx)

        avg_codisp = 0.0
        for tree in forest:
            if old_index is not None:
                tree.forget_point(old_index)
            tree.insert_point(point=cur_point, index=idx)
            avg_codisp += tree.codisp(idx) / num_trees
        print('CoDisp for point ({index}) is {avg_codisp}'.format(index=idx, avg_codisp=avg_codisp))
        anomaly_score[idx] = avg_codisp

    if True:
        fig, ax = plt.subplots(4, figsize=(20, 8))
        data_frame["bytes"].plot(ax=ax[0])
        data_frame["request"].plot(ax=ax[1])
        data_frame["num"].plot(ax=ax[2])
        anomaly_score.plot(ax=ax[3])

        ax[0].set_ylabel("bytes")
        ax[1].set_ylabel("request")
        ax[2].set_ylabel("num")
        ax[3].set_ylabel("codisp_score")
        plt.tight_layout()
        plt.show()


flow_detector()
