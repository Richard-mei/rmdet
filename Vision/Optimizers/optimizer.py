from ..utils import OPTIMIZERS
from torch import optim


@OPTIMIZERS.register_module()
class SGD(optim.SGD):
    def __init__(self, **kwargs):
        super(SGD, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return SGD
# @OPTIMIZERS.register_module()
# class SGD(optim.SGD):
#     def __init__(self, params=None, lr=None, momentum=None, weight_decay=None):
#         super(SGD, self).__init__(params, lr, momentum, weight_decay)
#         self.params = params
#         self.lr = lr
#         self.momentum = momentum
#         self.weight_decay = weight_decay
#
#     def __call__(self):
#         return optim.SGD(params=self.params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)


@OPTIMIZERS.register_module()
class Adam(optim.Adam):
    def __init__(self, **kwargs):
        super(Adam, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return Adam


@OPTIMIZERS.register_module()
class Adamw(optim.AdamW):
    def __init__(self, **kwargs):
        super(Adamw, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return Adamw


@OPTIMIZERS.register_module()
class RMSprop(optim.RMSprop):
    def __init__(self, **kwargs):
        super(RMSprop, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return RMSprop

