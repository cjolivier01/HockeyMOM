import importlib


class LazyImport:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def __getattr__(self, name):
        if self.module is None:
            self.module = importlib.import_module(self.module_name)
        return getattr(self.module, name)


# from .hm_transforms import *
# import hmlib.hm_transforms
# import hmlib.models.end_to_end
