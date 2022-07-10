import torch


class KDCache:
    
    def __init__(self):
        self.cache = None
        self.do_cache = True

    def remember(self, obj):
        if self.do_cache:
            self.cache = obj


class KDReadyModule:

    def __init__(
        self,
        model: torch.nn.Module,
        mapping_function: callable,
    ):
        self.model = model
        self.mapping_function = mapping_function
    
    def generate_all_kd_modules(self):
        for module in self.model.modules():
            if isinstance(module, KDCache):
                yield module
    
    def set_kd_modules_do_cache(self):
        for i, module in enumerate(self.generate_all_kd_modules()):
            module.do_cache = self.mapping_function(i)
    
    def get_all(self):
        return [module.cache for module in self.generate_all_kd_modules() if module.do_cache]
