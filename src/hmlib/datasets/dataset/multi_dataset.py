import math
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Set, Union

from torch.utils.data import Dataset


class MultiDatasetWrapper(Dataset):

    def __init__(self, *args, forgive_missing_attributes: List[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._datasets: OrderedDict[str, Dataset] = {}
        self._iters: OrderedDict[str, Iterable[Any]] = {}
        self._len = None
        self._forgive_missing_attributes: Set[str] = (
            set() if not forgive_missing_attributes else set(forgive_missing_attributes)
        )

    def add_forgive_missing_attribute(self, name: str) -> None:
        assert name
        self._forgive_missing_attributes.add(name)

    def get_max_attribute(self, name: str, forgiving: bool = True) -> Union[Any, None]:
        val: Any = None
        for _, ds_item in self._datasets.items():
            if hasattr(ds_item, name):
                new_value = getattr(ds_item, name)
                val = new_value if val is None else max(val, new_value)
        return val

    def append_dataset(self, name: str, dataset: Dataset):
        assert isinstance(dataset, Dataset)
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

    def __iter__(self) -> Iterable[Any]:
        self._iters.clear()
        for key, ds_item in self._datasets.items():
            self._iters[key] = iter(ds_item)
        return self

    def __next__(self) -> OrderedDict[str, Any]:
        data: OrderedDict[str, Any] = {}
        for key, iter_item in self._iters.items():
            data[key] = next(iter_item)
        return data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for key, ds in self._datasets.items():
            results[key] = ds[idx]
        return results

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped object if it's not found in Wrapper."""
        try:
            if not len(self._datasets):
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            checked_count: int = 0
            last_attr_value = None
            for obj in self._datasets.values():
                attr_value = getattr(obj, name, None)
                if attr_value is None:
                    if name in self._forgive_missing_attributes:
                        continue
                    raise AttributeError(f"'{type(obj).__name__}' object has no attribute '{name}'")
                if not checked_count:
                    last_attr_value = attr_value
                else:
                    if attr_value != last_attr_value:
                        print(
                            f"For getattr across multiple datasets, all getattr values must be the same ({attr_value} vs {last_attr_value})"
                        )
                checked_count += 1
            if not checked_count:
                # Didn't find any, may have been a forgiveable, but at least one needs to have it
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            return last_attr_value

        except AttributeError as ex:
            # Optionally, you can handle the case where the
            # attribute doesn't exist in __wrapped_class either
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}', caused by: {str(ex)}"
            )
