import math
from collections import OrderedDict
from typing import Any, Dict, Iterable

from torch.utils.data import Dataset


class MultiDatasetWrapper(Dataset):
    def __init__(self, *args, **kwargs):
        super(MultiDatasetWrapper, self).__init__(*args, **kwargs)
        self._datasets: OrderedDict[str, Dataset] = {}
        self._iters: OrderedDict[str, Iterable[Any]] = {}
        self._len = None

    def append_dataset(self, name: str, dataset: Dataset):
        assert name not in self._datasets
        assert len(self._iters) == 0
        assert self._len is None
        self._datasets[name] = dataset

    def close(self):
        for _, ds_item in self._datasets.items():
            if hasattr(ds_item, "close"):
                ds_item.close()

    def has_dataset(self, name: str) -> bool:
        return name in self._datasets

    def _get_iterator(self, name: str) -> Iterable[Any]:
        return self._iters[name]

    def __len__(self):
        if self._len is None:
            min_length = math.inf
            for _, ds_items in self._datasets.items():
                min_length = min(min_length, len(ds_items))
            self._len = min_length if min_length != math.inf else 0
        return self._len

    def __iter__(self):
        self._iters.clear()
        for key, ds_item in self._datasets.items():
            self._iters[key] = iter(ds_item)
        return self

    def __next__(self) -> OrderedDict[str, Any]:
        data: OrderedDict[str, Any] = {}
        for key, iter_item in self._datasets.items():
            data[key] = next(iter_item)
        return data

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped object if it's not found in Wrapper."""
        try:
            if not len(self._datasets):
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            checked_count: int = 0
            last_attr_value = None
            for obj in self._datasets.values():
                attr_value = getattr(obj, name)
                if not checked_count:
                    last_attr_value = attr_value
                else:
                    if attr_value != last_attr_value:
                        raise AssertionError(
                            f"For getattr across multipel datasets, all getattr values must be the same ({attr_value} vs {last_attr_value})"
                        )
                checked_count += 1
            return last_attr_value

        except AttributeError as e:
            # Optionally, you can handle the case where the
            # attribute doesn't exist in __wrapped_class either
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from e
