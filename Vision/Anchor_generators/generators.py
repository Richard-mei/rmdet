from ..utils import GENERATORS
import numpy as np
import torch.nn as nn
import torch
from torch.nn.modules.utils import _pair


def coords_fmap2orig(image_shape, stride):
    """
    transfor one fmap coords to orig coords
    Args
    featurn [batch_size,h,w,c]
    stride int
    Returns
    coords [n,2]
    """
    h, w = image_shape
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    coords = torch.stack([shift_x, shift_y, shift_x, shift_y], -1) + stride // 2
    return coords


@GENERATORS.register_module()
class Horizontal_Generator(nn.Module):
    def __init__(self, ratios, scales, strides, sizes):
        super().__init__()
        # if config is None:
        #     self.config = {'strides': [8, 16, 32, 64, 128], 'pyramid_levels': [3, 4, 5, 6, 7],
        #                    'sizes': [32, 64, 128, 256, 512], 'ratios': [0.5, 1, 2],
        #                    'scales': [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]}
        # else:
        #     self.config = config

        self.ratios = np.array(ratios)
        self.scales = np.array(eval(scales))
        self.size = sizes
        self.strides = strides

    # def forward(self, featmap_sizes):
    def forward(self, image):
        """Get anchors according to feature map sizes.

        Arg:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.

        Return:
            Tensor:
                anchors (Tensor): Anchors of one image (concatenate all level anchors).
        """
        H, W = image.size(2), image.size(3)  # (ori_H, ori_W)
        print(H, W)
        feature_size = [(H / stride, W / stride) for stride in self.strides]
        print(feature_size)

        # feature_size = featmap_sizes
        all_anchors = []
        for i in range(len(feature_size)):
            anchors = self.generate_anchors(self.size[i], self.ratios, self.scales)  # (9, 4)
            shift_anchors = self.shift(anchors, feature_size[i], self.strides[i])  # (H*W*9, 4)  4: x1, y1, x2, y2
            all_anchors.append(shift_anchors)
        all_anchors = torch.cat(all_anchors, dim=0)
        return all_anchors

    @staticmethod
    def generate_anchors(base_size=16, ratios=None, scales=None):
        if ratios is None:
            ratios = np.array([0.5, 1, 2])
        if scales is None:
            scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

        num_anchors = len(ratios) * len(scales)  # 9
        anchors = np.zeros((num_anchors, 4))
        anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T  # np.tile(ori, (y, x)) 将ori沿着y和x轴分别扩大y、x倍
        # 此时计算出的每个anchor的W=H，需要按照ratios进行调整
        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]  # (9,)
        # fix the ratios of w, h
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))  # (9,) np.repeat(c, n) 将c逐个元素复制n次，返回行向量
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))  # (9,)

        # transfrom from(0 ,0, w, h ) to ( x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
        anchors = torch.from_numpy(anchors).float().cuda() if torch.cuda.is_available() else torch.from_numpy(
            anchors).float()
        return anchors

    @staticmethod
    def shift(anchors, image_shape, stride):
        """
        anchors : Tensor(num, 4)
        image_shape : (H, W)
        return shift_anchor: (H*W*num,4)
        """

        ori_coords = coords_fmap2orig(image_shape, stride)  # (H*W, 4) 4:(x,y,x,y)
        ori_coords = ori_coords.to(device=anchors.device)
        shift_anchor = ori_coords[:, None, :] + anchors[None, :, :]
        return shift_anchor.reshape(-1, 4)


@GENERATORS.register_module()
class YOLO_Generator(object):
    def __init__(self, anchor_sizes=None, anchor_masks=None, strides=None, num_classes=None):
        if isinstance(anchor_sizes, str):
            self.anchor_sizes = torch.tensor(eval(anchor_sizes)).cuda()
        elif isinstance(anchor_sizes, list):
            self.anchor_sizes = torch.tensor(anchor_sizes).cuda()
        else:
            raise ValueError(f'Anchors must a list or str, got {type(anchor_sizes)} instead!')

        if isinstance(anchor_masks, str):
            self.anchor_masks = eval(anchor_masks)
        elif isinstance(anchor_masks, list):
            self.anchor_masks = anchor_masks
        else:
            raise ValueError(f'anchor_mask must be list or str, got {type(anchor_masks)}.')
        self.strides = strides
        self.num_classes = num_classes
    # def __init__(self, ratios, scales, strides, sizes):
    #     super().__init__()
    #     # if config is None:
    #     #     self.config = {'strides': [8, 16, 32, 64, 128], 'pyramid_levels': [3, 4, 5, 6, 7],
    #     #                    'sizes': [32, 64, 128, 256, 512], 'ratios': [0.5, 1, 2],
    #     #                    'scales': [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]}
    #     # else:
    #     #     self.config = config
    #
    #     self.ratios = np.array(ratios)
    #     self.scales = np.array(eval(scales))
    #     self.size = sizes
    #     self.strides = strides
    #
    # def forward(self, image):
    #     H, W = image.size(2), image.size(3)  # (ori_H, ori_W)
    #     feature_size = [(H / stride, W / stride) for stride in self.strides]
    #     all_anchors = []
    #     for i in range(len(feature_size)):
    #         anchors = self.generate_anchors(self.size[i], self.ratios, self.scales)  # 特定尺寸特征图上的9种不同尺度anchor
    #         shift_anchors = self.shift(anchors, feature_size[i], self.strides[i])  # (H*W, A, 4)  4: x1, y1, x2, y2
    #         all_anchors.append(shift_anchors)
    #     # all_anchors = torch.cat(all_anchors, dim=0)
    #     return all_anchors
    #
    # @staticmethod
    # def generate_anchors(base_size=16, ratios=None, scales=None):
    #     if ratios is None:
    #         ratios = np.array([0.5, 1, 2])
    #     if scales is None:
    #         scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    #
    #     num_anchors = len(ratios) * len(scales)  # 9
    #     anchors = np.zeros((num_anchors, 4))
    #     anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T  # np.tile(ori, (y, x)) 将ori沿着y和x轴分别扩大y、x倍
    #     # 此时计算出的每个anchor的W=H，需要按照ratios进行调整
    #     # compute areas of anchors
    #     areas = anchors[:, 2] * anchors[:, 3]  # (9,)
    #     # fix the ratios of w, h
    #     anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))  # (9,) np.repeat(c, n) 将c逐个元素复制n次，返回行向量
    #     anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))  # (9,)
    #
    #     # transfrom from(0 ,0, w, h ) to ( x1, y1, x2, y2)
    #     anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    #     anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    #     anchors = torch.from_numpy(anchors).float().cuda() if torch.cuda.is_available() else torch.from_numpy(
    #         anchors).float()
    #     return anchors
    #
    # @staticmethod
    # def shift(anchors, image_shape, stride):
    #     """
    #     anchors : Tensor(num, 4)
    #     image_shape : (H, W)
    #     return shift_anchor: (H*W*num,4)
    #     """
    #
    #     ori_coords = coords_fmap2orig(image_shape, stride)  # (H*W, 4) 4:(x,y,x,y)
    #     ori_coords = ori_coords.to(device=anchors.device)
    #     shift_anchor = ori_coords[:, None, :] + anchors[None, :, :]
    #     return shift_anchor.reshape(-1, 4)


@GENERATORS.register_module()
class Anchor_Gen:
    """Standard anchor generator for 2D anchor-based detectors.

    Args:
        strides (list[int] or list[tuple[int, int]] or torch.Tensor or str): Strides of anchors
            in multiple feature levels in order (w, h).
        ratios (list[float] | torch.Tensor): The list of ratios between the height and width
            of anchors in a single level.
        scales (list[int] | None): Anchor scales for anchors in a single level.
            It cannot be set at the same time if `octave_base_scale` and
            `scales_per_octave` are set.
        base_sizes (list[int] | None): The basic sizes
            of anchors in multiple levels.
            If None is given, strides will be used as base_sizes.
            (If strides are non square, the shortest stride is taken.)
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. By default it is True in V2.0
        octave_base_scale (int): The base scale of octave.
        scales_per_octave (int): Number of scales for each octave.
            `octave_base_scale` and `scales_per_octave` are usually used in
            retinanet and the `scales` should be None when they are set.
        centers (list[tuple[float, float]] | None): The centers of the anchor
            relative to the feature grid center in multiple feature levels.
            By default it is set to be None and not used. If a list of tuple of
            float is given, they will be used to shift the centers of anchors.
        center_offset (float): The offset of center in proportion to anchors'
            width and height. By default it is 0 in V2.0.

    Examples:
        >>> self = Anchors_Gen([16], [1.], [1.], [9])
        >>> all_anchors = self.grid_anchors([(2, 2)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]])]
        >>> self = AnchorGenerator([16, 32], [1.], [1.], [9, 18])
        >>> all_anchors = self.grid_anchors([(2, 2), (1, 1)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]]), \
        tensor([[-9., -9., 9., 9.]])]
    """

    def __init__(self,
                 strides,
                 ratios,
                 scales=None,
                 base_sizes=None,
                 scale_major=True,
                 centers=None,
                 center_offset=0.,
                 dynamic_image_size=True
                 ):

        # calculate base sizes of anchors
        self.strides = [_pair(stride) for stride in strides]
        self.base_sizes = [min(stride) for stride in self.strides
                           ] if base_sizes is None else base_sizes
        assert len(self.base_sizes) == len(self.strides), \
            'The number of strides should be the same as base sizes, got ' \
            f'{self.strides} and {self.base_sizes}'

        self.scales = torch.Tensor(scales) if isinstance(scales, list) else torch.Tensor(eval(scales))
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.dynamic_image_size = dynamic_image_size

        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    def gen_base_anchors(self):
        """Generate base anchors.

        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    center=center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        """Generate base anchors of a single level.

        Args:
            base_size (int | float): Basic size of an anchor.
            scales (torch.Tensor): Scales of the anchor.
            ratios (torch.Tensor): The ratio between between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)  # x1, y1, x2, y2

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        """Generate mesh grid of x and y.

        Args:
            x (torch.Tensor): Grids of x dimension.
            y (torch.Tensor): Grids of y dimension.
            row_major (bool, optional): Whether to return y grids first.
                Defaults to True.

        Returns:
            tuple[torch.Tensor]: The mesh grids of x and y.
        """
        # use shape instead of len to keep tracing while exporting to onnx
        xx = x.repeat(y.shape[0])
        yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_sizes, device='cuda'):
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple] or torch.tensor): List of feature map sizes in
                multiple feature levels.
            device (str): Device where the anchors will be put on.

        Return:
            list[torch.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature level, \
                num_base_anchors is the number of anchors for that level.
        """
        # for image input
        if self.dynamic_image_size:
            H, W = featmap_sizes.size(2), featmap_sizes.size(3)  # (ori_H, ori_W)
            featmap_sizes = [(H / stride[1], W / stride[0]) for stride in self.strides]

        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                self.base_anchors[i].to(device),
                featmap_sizes[i],
                self.strides[i],
                device=device)
            multi_level_anchors.append(anchors)
        multi_level_anchors = torch.cat(multi_level_anchors, dim=0)
        return multi_level_anchors

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16),
                                  device='cuda'):
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_anchors``.

        Args:
            base_anchors (torch.Tensor): The base anchors of a feature grid.
            featmap_size (tuple[int]): Size of the feature maps.
            stride (tuple[int], optional): Stride of the feature map in order
                (w, h). Defaults to (16, 16).
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """
        # keep as Tensor, so that we can covert to ONNX correctly
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride[0]
        shift_y = torch.arange(0, feat_h, device=device) * stride[1]

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)
        # 这里的base_anchor中心点应该在(0, 0)，即Anchor_offset应该设置为0，然后再进行整张图上的复制

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)  # x1, y1, x2, y2
        # 每9行表示一个点的9各Anchor
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
        """Generate valid flags of anchors in multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels.
            pad_shape (tuple): The padded shape of the image.
            device (str): Device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of anchors in multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / anchor_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  self.num_base_anchors[i],
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 num_base_anchors,
                                 device='cuda'):
        """Generate the valid flags of anchor in a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps.
            valid_size (tuple[int]): The valid size of the feature maps.
            num_base_anchors (int): The number of base anchors.
            device (str, optional): Device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each anchor in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(valid.size(0),
                                      num_base_anchors).contiguous().view(-1)
        return valid


@GENERATORS.register_module()
class YOLOAnchorGen(Anchor_Gen):
    """Anchor generator for YOLO.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels.
        base_sizes (list[list[tuple[int, int]]]): The basic sizes
            of anchors in multiple levels.
    """
    def __init__(self, strides, base_sizes, dynamic_image_size=True):
        if isinstance(base_sizes, str):
            base_sizes = eval(base_sizes)
        self.strides = [_pair(stride) for stride in strides]
        self.centers = [(stride[0] / 2., stride[1] / 2.)
                        for stride in self.strides]
        self.base_sizes = []
        num_anchor_per_level = len(base_sizes[0])  # 3
        for base_sizes_per_level in base_sizes:
            assert num_anchor_per_level == len(base_sizes_per_level)
            self.base_sizes.append(
                [_pair(base_size) for base_size in base_sizes_per_level])
        self.dynamic_image_size = dynamic_image_size
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.base_sizes)

    def gen_base_anchors(self):
        """Generate base anchors.

        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_sizes_per_level in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(base_sizes_per_level,
                                                   center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self, base_sizes_per_level, center=None):
        """Generate base anchors of a single level.

        Args:
            base_sizes_per_level (list[tuple[int, int]]): Basic sizes of
                anchors.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        x_center, y_center = center
        base_anchors = []
        for base_size in base_sizes_per_level:
            w, h = base_size

            # use float anchor and the anchor's center is aligned with the
            # pixel center
            base_anchor = torch.Tensor([
                x_center - 0.5 * w, y_center - 0.5 * h, x_center + 0.5 * w,
                y_center + 0.5 * h
            ])
            base_anchors.append(base_anchor)
        base_anchors = torch.stack(base_anchors, dim=0)

        return base_anchors

    def responsible_flags(self, featmap_sizes, gt_bboxes, device='cuda'):
        """Generate responsible anchor flags of grid cells in multiple scales.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in multiple
                feature levels.
            gt_bboxes (Tensor): Ground truth boxes, shape (n, 4).
            device (str): Device where the anchors will be put on.

        Return:
            list(torch.Tensor): responsible flags of anchors in multiple level
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_responsible_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            flags = self.single_level_responsible_flags(
                featmap_sizes[i],
                gt_bboxes,
                anchor_stride,
                self.num_base_anchors[i],
                device=device)
            multi_level_responsible_flags.append(flags)
        return multi_level_responsible_flags

    def single_level_responsible_flags(self,
                                       featmap_size,
                                       gt_bboxes,
                                       stride,
                                       num_base_anchors,
                                       device='cuda'):
        """Generate the responsible flags of anchor in a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps.
            gt_bboxes (Tensor): Ground truth boxes, shape (n, 4).
            stride (tuple(int)): stride of current level
            num_base_anchors (int): The number of base anchors.
            device (str, optional): Device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each anchor in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        gt_bboxes_cx = ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) * 0.5).to(device)
        gt_bboxes_cy = ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) * 0.5).to(device)
        gt_bboxes_grid_x = torch.floor(gt_bboxes_cx / stride[0]).long()
        gt_bboxes_grid_y = torch.floor(gt_bboxes_cy / stride[1]).long()

        # row major indexing
        gt_bboxes_grid_idx = gt_bboxes_grid_y * feat_w + gt_bboxes_grid_x

        responsible_grid = torch.zeros(
            feat_h * feat_w, dtype=torch.uint8, device=device)
        responsible_grid[gt_bboxes_grid_idx] = 1

        responsible_grid = responsible_grid[:, None].expand(
            responsible_grid.size(0), num_base_anchors).contiguous().view(-1)
        return responsible_grid
















