import cameratransform as ct
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from fast_pytorch_kmeans import KMeans
import matplotlib.pyplot as plt
import pt_autograph as ptag

import torch


class ClusterSearch:
    def __init__(self, sizes: List[int] = [3, 2], device="cpu"):
        self._sizes = sizes
        self._device = device
        self._kmeans_objects = dict()
        self._cluster_label_ids = dict()
        self._largest_cluster_label = dict()
        self._largest_cluster_id_set = dict()
        self._cluster_counts = dict()
        self.reset_clusters()

    def reset_clusters(self):
        self._cluster_label_ids.clear()
        self._largest_cluster_label.clear()
        self._cluster_counts.clear()
        self._largest_cluster_id_set.clear()

    def calculate_clusters(
        self, center_points: torch.Tensor, ids: torch.Tensor, num_clusters: int
    ):
        self._cluster_label_ids[num_clusters] = dict()
        if len(center_points) < num_clusters:
            # Minimum clusters that we can have is the minimum number of objects
            max_clusters = len(center_points)
            if not max_clusters:
                return
            elif max_clusters < num_clusters:
                return
        self._largest_cluster_label[num_clusters] = None
        labels = None
        if num_clusters not in self._kmeans_objects:
            self._kmeans_objects[num_clusters] = KMeans(
                n_clusters=num_clusters,
                mode="euclidean",
            )
        # torch_tensors = []
        # for n in center_points:
        #     # print(n)
        #     torch_tensors.append(n.to(device))
        # tt = torch.cat(torch_tensors, dim=0)
        # tt = torch.reshape(tt, (len(torch_tensors), 2))
        # print(tt)
        labels = self._kmeans_objects[num_clusters].fit_predict(
            center_points.to(self._device)
        )
        # print(labels)
        # self._cluster_counts[num_clusters] = [0 for i in range(num_clusters)]
        # cluster_counts = self._cluster_counts[num_clusters]
        # cluster_label_ids = self._cluster_label_ids[num_clusters]
        # id_count = len(ids)
        # trunk-ignore(bandit/B101)
        # assert id_count == labels.shape[0]
        # for i in range(id_count):
        #     id = ids[i].item()
        #     cluster_label = labels[i].item()
        #     cluster_counts[cluster_label] += 1
        #     if cluster_label not in cluster_label_ids:
        #         cluster_label_ids[cluster_label] = [id]
        #     else:
        #         cluster_label_ids[cluster_label].append(id)

        cluster_ids = []
        for i in range(num_clusters):
            cids = ids[labels == i]
            cluster_ids.append(cids)
            self._cluster_label_ids[num_clusters][i] = cids

        self._cluster_counts[num_clusters] = [len(t) for t in cluster_ids]
        index_of_max_count = np.argmax(self._cluster_counts[num_clusters])
        self._largest_cluster_label[num_clusters] = index_of_max_count
        self._largest_cluster_id_set[num_clusters] = set(
            self._cluster_label_ids[num_clusters][
                self._largest_cluster_label[num_clusters]
            ].tolist()
        )

        # for cluster_label, cluster_id_list in self._cluster_label_ids[
        #     num_clusters
        # ].items():
        #     if self._largest_cluster_label[num_clusters] is None:
        #         self._largest_cluster_label[num_clusters] = cluster_label
        #     elif len(cluster_id_list) > len(
        #         cluster_label_ids[self._largest_cluster_label[num_clusters]]
        #     ):
        #         self._largest_cluster_label[num_clusters] = cluster_label

    def calculate_all_clusters(self, center_points: torch.Tensor, ids: torch.Tensor):
        self.reset_clusters()
        for i in self._sizes:
            self.calculate_clusters(
                center_points=center_points, ids=ids, num_clusters=i
            )

    def get_largest_cluster_id_set(self, num_clusters):
        if num_clusters not in self._largest_cluster_label:
            return set()
        return set(
            self._cluster_label_ids[num_clusters][
                self._largest_cluster_label[num_clusters]
            ]
        )

    def prune_not_in_largest_cluster(self, num_clusters, ids):
        largest_cluster_set = self._largest_cluster_id_set[num_clusters]
        result_ids = []
        for id in ids:
            id_item = id.item()
            if id_item in largest_cluster_set:
                result_ids.append(id)
        if not result_ids:
            return []
        return torch.tensor(result_ids, dtype=torch.int64)
