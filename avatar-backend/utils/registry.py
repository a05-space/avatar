import inspect


def build_from_cfg(cfg, registry, default_args=None):
    args = cfg.copy()
    obj_type = args.pop("type")
    obj_cls = registry.get(obj_type)
    return obj_cls(**args)

class Registry:

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()
        self._scope = self.infer_scope()
        self.build_func = build_from_cfg

    @staticmethod
    def infer_scope():
        filename = inspect.getmodule(inspect.stack()[2][0]).__name__
        split_filename = filename.split(".")
        return split_filename[0]

    def get(self, key):
        if key in self._module_dict:
            return self._module_dict[key]

    def build(self, *args, **kwargs):
        return self.build_func(*args, **kwargs, registry=self)

    def _register_module(self, module_class, module_name=None, force=False):
        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            self._module_dict[name] = module_class

    def register_module(self, name=None, force=False, module=None):
        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register
