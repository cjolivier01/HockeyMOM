import torch


# class StreamTensor:
    
#     def __int__(tensor: torch.Tensor)
    
#     def ref(self):
#         assert False and "Not implemented"
    
#     def get(self):
#         assert 


class StreamTensorToGpu:
    def __init__(self, tensor: torch.Tensor, device: torch.device):
        assert tensor.device.type != device.type
        self._stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream=self._stream):
            self._tensor = tensor.to(device, non_blocking=True)

    def ref(self):
        return self._tensor

    def get(self):
        self._stream.synchronize()
        t = self._tensor
        self._tensor = None
        return t


class StreamTensorToDtype:
    def __init__(self, tensor: torch.Tensor, dtype: torch.dtype):
        assert tensor.dtype != dtype
        self._stream = torch.cuda.Stream(tensor.device)
        with torch.cuda.stream(stream=self._stream):
            self._tensor = tensor.to(dtype=dtype, non_blocking=True)

    def ref(self):
        return self._tensor

    def get(self):
        self._stream.synchronize()
        t = self._tensor
        self._tensor = None
        return t
