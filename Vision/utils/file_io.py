import json
from abc import ABC
import abc
import csv

import yaml


class FileBase(ABC):
    @staticmethod
    @abc.abstractmethod  # 将方法标记为抽象的==>在子类中必须实现
    def load(inputs):
        """Retrieve data from the input source and return an object."""
        return

    @staticmethod
    @abc.abstractmethod
    def save(output, data):
        """Save the data object to the output."""
        return


class json_file(FileBase):

    @staticmethod
    def save(dict_config: dict, save_path: str):
        cfg = json.dumps(dict_config)
        with open(r'{}'.format(save_path), 'w') as f:
            f.write(cfg)

    @staticmethod
    def load(file_path: str):
        cfg = json.load(open(file_path))
        return cfg

    @staticmethod
    def show(cfg: dict or str):
        if isinstance(cfg, dict):
            print(json.dumps(cfg, indent=4))
        elif isinstance(cfg, str) and cfg.endswith('json'):
            print(json.dumps(json_file.load(cfg), indent=2))
        else:
            raise TypeError(f'File to show must be a dict or json file! But got {type(cfg)} instead.')


class yaml_file(FileBase):

    @staticmethod
    def save(dict_config: dict, save_path: str):
        cfg = yaml.dump(dict_config)
        with open(save_path, 'w') as f:
            f.write(cfg)

    @staticmethod
    def load(file_path: str):
        cfg = yaml.load(open(file_path, 'r'), Loader=yaml.FullLoader)
        return cfg


class csv_file(FileBase):

    @staticmethod
    def save(output, data):
        pass

    @staticmethod
    def load(file_path: str):
        cfg = csv.reader(file_path)
        return cfg


class file(FileBase):
    @staticmethod
    def load(file_path: str):
        if file_path.endswith('json'):
            cfg = json_file.load(file_path)
        elif file_path.endswith('yaml'):
            cfg = yaml_file.load(file_path)
        elif file_path.endswith('csv'):
            cfg = csv_file.load(file_path)
        else:
            raise TypeError('File format must be .json .yaml .csv!')
        return cfg

    @staticmethod
    def save(dict_config: dict, save_path: str):
        if save_path.endswith('json'):
            json_file.save(dict_config, save_path)
        elif save_path.endswith('yaml'):
            yaml_file.save(dict_config, save_path)
        else:
            raise TypeError('File format must be .json .yaml!')

    @staticmethod
    def show(cfg: dict or str):
        if isinstance(cfg, dict) or isinstance(cfg, str) and cfg.endswith('json'):
            json_file.show(cfg)

