import math
import torch.nn.functional
import numpy as np
import torch
import cv2
from torchvision import transforms
import random

# class Augmenter(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample, flip_x=0.5):
#         if np.random.rand() < flip_x:
#             image, annots = sample['img'], sample['annot']
#             image = image[:, ::-1, :]
#
#             rows, cols, channels = image.shape
#             x1 = annots[:, 0].copy()
#             x2 = annots[:, 2].copy()
#
#             x_tmp = x1.copy()
#
#             annots[:, 0] = cols - x2
#             annots[:, 2] = cols - x_tmp
#
#             sample['img'], sample['annot'] = image, annots
#             # sample = {'img': image, 'annot': annots}
#
#         return sample
#


class Augmenter(object):
    """
    随机颜色抖动、随机旋转、随机水平翻转、随机裁剪、随机擦除。
    0.5的概率随机左右翻转"""

    def __call__(self, sample, flip_x=0.5):
        sample = self.random_flip(sample, flip_x)
        # if np.random.rand() < 0.5:
        #     sample = self.random_rotation(sample, degree=10)
        # if np.random.rand() < 0.5:
        #     sample = self.random_colorJitter(sample)
        # if np.random.rand() < 0.5:
        #     sample = self.random_crop(sample)
        if np.random.rand() < 0.5:
            sample = self.random_Erasing(sample)
        return sample

    @staticmethod
    def random_flip(sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            img, annots = sample['img'], sample['annot']
            img = img[:, ::-1, :]

            rows, cols, channels = img.shape
            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample['img'], sample['annot'] = img, annots
        return sample

    @staticmethod
    def random_rotation(sample, degree=10):
        img, annots = sample['img'], sample['annot']

        d = random.uniform(-degree, degree)
        h, w, _ = img.shape
        rx0, ry0 = w / 2.0, h / 2.0
        img = img.rotate(d)
        a = -d / 180.0 * math.pi
        annots = torch.from_numpy(annots)
        new_boxes = torch.zeros_like(annots)
        new_boxes[:, 0] = annots[:, 1]
        new_boxes[:, 1] = annots[:, 0]
        new_boxes[:, 2] = annots[:, 3]
        new_boxes[:, 3] = annots[:, 2]
        for i in range(annots.shape[0]):
            ymin, xmin, ymax, xmax = new_boxes[i, :]
            xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
            x0, y0 = xmin, ymin
            x1, y1 = xmin, ymax
            x2, y2 = xmax, ymin
            x3, y3 = xmax, ymax
            z = torch.FloatTensor([[y0, x0], [y1, x1], [y2, x2], [y3, x3]])
            tp = torch.zeros_like(z)
            tp[:, 1] = (z[:, 1] - rx0) * math.cos(a) - (z[:, 0] - ry0) * math.sin(a) + rx0
            tp[:, 0] = (z[:, 1] - rx0) * math.sin(a) + (z[:, 0] - ry0) * math.cos(a) + ry0
            ymax, xmax = torch.max(tp, dim=0)[0]
            ymin, xmin = torch.min(tp, dim=0)[0]
            new_boxes[i] = torch.stack([ymin, xmin, ymax, xmax])
        new_boxes[:, 1::2].clamp_(min=0, max=w - 1)
        new_boxes[:, 0::2].clamp_(min=0, max=h - 1)
        annots[:, 0] = new_boxes[:, 1]
        annots[:, 1] = new_boxes[:, 0]
        annots[:, 2] = new_boxes[:, 3]
        annots[:, 3] = new_boxes[:, 2]
        annots = annots.numpy()
        sample['img'], sample['annot'] = img, annots
        return sample

    @staticmethod
    def random_colorJitter(sample, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        img = sample['img']
        img = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)(img)
        sample['img'] = img
        return sample

    @staticmethod
    def random_crop(sample, crop_scale_min=0.2, aspect_ratio=[3. / 4, 4. / 3], remain_min=0.7, attempt_max=10):
        img, annots = sample['img'], sample['annot']
        success = False
        annots = torch.from_numpy(annots)
        for attempt in range(attempt_max):
            # choose crop size
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(crop_scale_min, 1.0) * area
            aspect_ratio_ = random.uniform(aspect_ratio[0], aspect_ratio[1])
            w = int(round(math.sqrt(target_area * aspect_ratio_)))
            h = int(round(math.sqrt(target_area / aspect_ratio_)))
            if random.random() < 0.5:
                w, h = h, w
            # if size is right then random crop
            if w <= img.shape[1] and h <= img.shape[0]:
                x = random.randint(0, img.shape[1] - w)
                y = random.randint(0, img.shape[0] - h)
                # check
                crop_box = torch.FloatTensor([[x, y, x + w, y + h]])
                inter = _box_inter(crop_box, annots)  # [1,N] N can be zero
                box_area = (annots[:, 2] - annots[:, 0]) * (annots[:, 3] - annots[:, 1])  # [N]
                mask = inter > 0.0001  # [1,N] N can be zero
                inter = inter[mask]  # [1,S] S can be zero
                box_area = box_area[mask.view(-1)]  # [S]
                box_remain = inter.view(-1) / box_area  # [S]
                if box_remain.shape[0] != 0:
                    if bool(torch.min(box_remain > remain_min)):
                        success = True
                        break
                else:
                    success = True
                    break
        if success:
            img = img.crop((x, y, x + w, y + h))
            annots -= torch.Tensor([x, y, x, y])
            annots[:, 1::2].clamp_(min=0, max=h - 1)
            annots[:, 0::2].clamp_(min=0, max=w - 1)
            # ow, oh = (size, size)
            # sw = float(ow) / img.size[0]
            # sh = float(oh) / img.size[1]
            # img = img.resize((ow,oh), Image.BILINEAR)
            # boxes *= torch.FloatTensor([sw,sh,sw,sh])
        annots = annots.numpy()
        sample['img'], sample['annot'] = img, annots
        return sample

    @staticmethod
    def random_Erasing(sample, EPSILON=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        img = sample['img']
        # print(img.shape)  # 480,640,3

        if random.uniform(0, 1) > EPSILON:
            sample['img'] = img

            return sample

        for attempt in range(100):
            # area = img.size()[1] * img.size()[2]
            area = img.shape[0] * img.shape[1]

            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1 / r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                if img.shape[2] == 3:
                    # img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    # img[1, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    # img[2, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    img[x1:x1 + h, y1:y1 + w, 0] = mean[0]
                    img[x1:x1 + h, y1:y1 + w, 1] = mean[1]
                    img[x1:x1 + h, y1:y1 + w, 2] = mean[2]
                    # img[:, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(3, h, w))
                else:
                    img[x1:x1 + h, y1:y1 + w, 0] = mean[1]
                    # img[0, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(1, h, w))\
                sample['img'] = img
                return sample
        sample['img'] = img
        return sample

    @staticmethod
    def CenterCrop(sample, size: int or list):
        img, annots = sample['img'], sample['annot']
        h, w, _ = img.shape
        center = (int(h/2), int(w/2))
        crop = transforms.TenCrop(size=size)
        cropped_img = crop(img)
        local_x1, local_y1, local_x2, local_y2 = center


class Normalizer(object):

    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        sample['image']= transforms.Normalize(self.mean, self.std, inplace=True)(image)
        sample['annot'] = annots
        return sample

        # return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


class Resizer(object):
    def __init__(self, img_sizes=None):
        if img_sizes is None:
            img_sizes = [480, 640]
        self.img_sizes = img_sizes

    def __call__(self, sample):
        h_side, w_side = self.img_sizes
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape  # (480, 640, 3)
        scale = h_side / height
        if width * scale > w_side:
            scale = w_side / width
        # smallest_side = min(width, height)  # 短边
        # largest_side = max(width, height)  # 长边
        # scale = min_side / smallest_side  # 缩放比例
        # # scale = max_side / largest_side  # 保证最大边等于右限
        # if largest_side * scale > max_side:
        #     scale = max_side / largest_side
        nw, nh = int(scale * width), int(scale * height)
        assert (nw <= w_side) and (nh <= w_side)
        image_resized = cv2.resize(image, (nw, nh))  # nh, nw ,3
        # if nh > nw:
        #     image_paded = np.zeros(shape=[max_side, min_side, 3], dtype=np.uint8)
        #     image_paded[:nh, :nw, :] = image_resized
        # else:
        #     image_paded = np.zeros(shape=[min_side, max_side, 3], dtype=np.uint8)
        #     image_paded[:nh, :nw, :] = image_resized
        image_paded = np.zeros(shape=[h_side, w_side, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized
        # if min_side == max_side:
        #     pad_w = math.ceil(min_side / 32) * 32 - nw
        #     pad_h = math.ceil(min_side / 32) * 32 - nh
        # else:
        #     pad_w = math.ceil(nw / 32) * 32 - nw
        #     pad_h = math.ceil(nh / 32) * 32 - nh
        # image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        # image_paded[:nh, :nw, :] = image_resized
        if annots is None:
            return image_paded
        else:
            annots[..., [0, 2]] = annots[..., [0, 2]] * scale
            annots[..., [1, 3]] = annots[..., [1, 3]] * scale
            sample['img'], sample['annot'], sample['scale'] = transforms.ToTensor()(image_paded), torch.from_numpy(annots), scale
            # sample['img'], sample['annot'], sample['scale'] = \
            #     torch.from_numpy(image_paded).to(torch.float32), torch.from_numpy(annots), scale
            return sample
            # return {'img': torch.from_numpy(image_paded).to(torch.float32), 'annot': torch.from_numpy(annots),
            #         'scale': scale}


def collate_fn(sample):
    imgs_list = [s['img'] for s in sample]  # 3, 480, 640     0, 1, 2  => 2, 0, 1
    annots_list = [s['annot'] for s in sample]
    scales = [s['scale'] for s in sample]
    ids = [s['id'] for s in sample]
    assert len(imgs_list) == len(annots_list)
    batch_size = len(annots_list)
    pad_imgs_list = []
    pad_annots_list = []
    h_list = [int(s.shape[1]) for s in imgs_list]  # 480
    w_list = [int(s.shape[2]) for s in imgs_list]  # 640
    max_h = max(np.array(h_list))
    max_w = max(np.array(w_list))

    for i in range(batch_size):
        img = imgs_list[i]  # 3, 480, 640
        pad_imgs_list.append(
            torch.nn.functional.pad(img, (0, 0, 0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.))

    max_num = 0
    for i in range(batch_size):
        n = annots_list[i].shape[0]
        if n > max_num:
            max_num = n
    for i in range(batch_size):
        pad_annots_list.append(
            torch.nn.functional.pad(annots_list[i], (0, 0, 0, max_num - annots_list[i].shape[0]), value=-1))
    batch_imgs = torch.stack(pad_imgs_list)
    # batch_imgs = torch.stack(pad_imgs_list).permute(0, 3, 1, 2)
    batch_annots = torch.stack(pad_annots_list)
    return {'img': batch_imgs, 'annot': batch_annots, 'scale': scales, 'id': ids}


# def xyxy2xywh(x):
#     # Transform box coordinates from [x1, y1, x2, y2, cls] (where xy1=top-left, xy2=bottom-right) to [x, y, w, h, cls]
#     y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
#     y[:, 4:] = x[:, 4:]
#     y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
#     y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
#     y[:, 2] = x[:, 2] - x[:, 0]  # width
#     y[:, 3] = x[:, 3] - x[:, 1]  # height
#     return y
#
#
# def xywh2xyxy(x):
#     # Transform box coordinates from [x, y, w, h] to [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right)
#     y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
#     y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
#     y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
#     y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
#     y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
#     return y


