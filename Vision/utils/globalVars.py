def init_global_dict():
    global _global_dict
    _global_dict = {}


def set_value(key, value, logger=None):
    _global_dict[key] = value
    if logger:
        logger.info(f'Set {key}: {value}')


def get_value(key, default_value=None):
    try:
        return _global_dict[key]
    except KeyError:
        return default_value
