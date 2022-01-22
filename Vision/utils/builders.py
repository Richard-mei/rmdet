from .registry import Registry
import inspect
from torch import nn


GENERATORS = Registry('generator')
DATASETS = Registry('dataset')
LOSSES = Registry('loss')
DETECTORS = Registry('detector')
OPTIMIZERS = Registry('optimizer')


def build_from_cfg(cfg, registry, default_args=None, **kwargs):
    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('cfg')  # 注册 str 类名
    if isinstance(obj_type, str):
        # 相当于 self._module_dict[obj_type]
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')

    # 如果已经实例化了，那就直接返回
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')

    # 最终初始化对于类，并且返回，就完成了一个类的实例化过程
    return obj_cls(**args, **kwargs)


def build_from_cfg_(cfg, registry, **kwargs):
    if isinstance(cfg, str):
        obj = registry.get(cfg)
        if obj is None:
            raise KeyError(
                f'{obj} is not in the {registry.name} registry'
            )
    elif inspect.isclass(cfg):
        obj = cfg
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(cfg)}'
        )
    return obj(**kwargs)


def build(cfg, registry, **kwargs):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, **kwargs) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, **kwargs)


def build_generator(cfg, **kwargs):
    """Build generator."""
    return build(cfg, GENERATORS, **kwargs)


def build_dataset(cfg, **kwargs):
    return build(cfg, DATASETS, **kwargs)


def build_loss(cfg, **kwargs):
    """Build loss."""
    return build(cfg, LOSSES, **kwargs)


def build_detector(cfg, **kwargs):
    """Build detector."""
    return build(cfg, DETECTORS, **kwargs)


def build_optimizer(cfg, **kwargs):
    """Build backbone."""
    return build(cfg, OPTIMIZERS, **kwargs)
