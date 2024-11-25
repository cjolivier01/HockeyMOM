from typing import Dict, List, Optional, Tuple

import torch
from fast_pytorch_kmeans import KMeans


class ClusterSnapshot:

    def __init__(
        self,
        num_clusters: int,
        device: str,
        centroids: Optional[torch.Tensor] = None,
        minibatch: Optional[int] = None,
        init_method: Optional[str] = None,
    ):
        self._device = device
        self._num_clusters = num_clusters
        self._centroids = centroids
        self._kmeans_object = KMeans(
            n_clusters=num_clusters,
            mode="euclidean",
            minibatch=minibatch,
            init_method=init_method,
        )
        self.reset()

    def reset(self):
        self._cluster_label_ids = None
        self._largest_cluster_label = None
        self._largest_cluster_id_set = None
        self._cluster_counts = None

    def calculate(self, center_points: torch.Tensor, ids: torch.Tensor):
        assert len(center_points) == len(ids)
        self.reset()
        if len(center_points) < self._num_clusters:
            # Minimum clusters that we can have is the minimum number of objects
            max_clusters = len(center_points)
            if not max_clusters:
                return
            elif max_clusters < self._num_clusters:
                return
        labels = self._kmeans_object.fit_predict(
            center_points.to(self._device), centroids=self._centroids
        )

        self._cluster_label_ids = list()
        for i in range(self._num_clusters):
            cids = ids[labels == i]
            self._cluster_label_ids.append(cids)

        self._cluster_counts = torch.tensor(
            [len(t) for t in self._cluster_label_ids], dtype=torch.int64
        )
        index_of_max_count = torch.argmax(self._cluster_counts)
        self._largest_cluster_label = index_of_max_count
        self._largest_cluster_id_set = set(
            self._cluster_label_ids[self._largest_cluster_label].tolist()
        )

    def prune_not_in_largest_cluster(self, ids):
        if not self._largest_cluster_id_set:
            return []
        result_ids = []
        for id in ids:
            id_item = id.item()
            if id_item in self._largest_cluster_id_set:
                result_ids.append(id)
        if not result_ids:
            return []
        return torch.tensor(result_ids, dtype=torch.int64, device=ids.device)


class ClusterMan:

    def __init__(self, sizes: List[int], device="cpu", init_method: Optional[str] = None):
        self._sizes = sizes
        self._device = device
        self.cluster_snapshots = dict()
        for i in sizes:
            self.cluster_snapshots[i] = ClusterSnapshot(
                num_clusters=i,
                device=device,
                init_method="random" if not init_method else init_method,
            )

    @property
    def cluster_counts(self):
        return self._sizes

    def reset_clusters(self):
        for cs in self.cluster_snapshots.values():
            cs.reset()

    def calculate_all_clusters(self, center_points: torch.Tensor, ids: torch.Tensor):
        self.reset_clusters()
        if isinstance(ids, list):
            if not ids:
                # No items are currently being tracked
                # This should always be the case when 'ids' is of type 'list'
                # Allowin it to fall-through in order to catch as an error if it is not empty.
                return
        if ids.device != self._device:
            ids = ids.to(self._device)
        for cluster_snapshot in self.cluster_snapshots.values():
            cluster_snapshot.calculate(center_points=center_points, ids=ids)

    def prune_not_in_largest_cluster(self, num_clusters, ids):
        return self.cluster_snapshots[num_clusters].prune_not_in_largest_cluster(ids)
