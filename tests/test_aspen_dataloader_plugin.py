import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from hmlib.aspen import AspenNet


def _make_graph(params):
    return {
        "plugins": {
            "dataloader": {
                "class": "hmlib.aspen.plugins.dataloader_plugin.DataLoaderPlugin",
                "depends": [],
                "params": params,
            }
        }
    }


def should_emit_batches_from_shared_dataloader():
    dataset = [{"x": torch.tensor(1)}, {"x": torch.tensor(2)}]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    net = AspenNet("dataloader_shared", _make_graph({}), shared={"dataloader": loader})

    out1 = net({})
    assert out1["x"].item() == 1

    out2 = net({})
    assert out2["x"].item() == 2

    with pytest.raises(StopIteration):
        net({})


def should_build_dataloader_from_spec():
    dataloader_spec = {
        "class": "torch.utils.data.DataLoader",
        "params": {
            "dataset": {
                "class": "torch.utils.data.TensorDataset",
                "args": [torch.tensor([3, 4])],
            },
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 0,
        },
    }
    params = {"dataloader": dataloader_spec, "wrap_key": "batch"}
    net = AspenNet("dataloader_spec", _make_graph(params))

    out = net({})
    assert out["batch"][0].item() == 3
