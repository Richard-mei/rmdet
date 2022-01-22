import inspect


class Registry:
    def __init__(self, name):
        # 可实现注册类细分功能
        self._name = name
        # 内部核心内容，维护所有的已经注册好的 class
        self._module_dict = dict()

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, '
                            f'but got {type(module_class)}')

        if module_name is None:
            module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError(f'{module_name} is already registered '
                           f'in {self._name}')
        # 最核心代码
        self._module_dict[module_name] = module_class

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        if key in self._module_dict:
            return self._module_dict[key]
        else:
            raise KeyError
    # 装饰器函数

    def register_module(self, name=None, force=False, module=None):
        if module is not None:
            # 如果已经是 module，那就知道 增加到字典中即可
            self._register_module(
                module_class=module, module_name=name, force=force)
            return module

        # 最标准用法
        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(
                module_class=cls, module_name=name, force=force)
            return cls

        return _register
