import torch
import xml.etree.ElementTree as ET
import os
from torchvision import transforms
from PIL import Image
import random
from torch.utils.data import Dataset
import torch.nn.functional
from ..utils import DATASETS, xyxy2xywh
import cv2
import numpy as np


def flip(img, boxes):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    w = img.width
    if boxes.shape[0] != 0:
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 2] = xmax
        boxes[:, 0] = xmin
    return img, boxes


class VOCDataset_(torch.utils.data.Dataset):
    CLASSES_NAME = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    def __init__(self, root_dir, train_set=None, resize_size=None, use_difficult=False, is_train=True,
                 augment=None):
        if resize_size is None:
            resize_size = [800, 1333]
        self.root = root_dir
        self.use_difficult = use_difficult

        self.train_set = train_set
        self.img_ids = list()
        for (year, name) in self.train_set:
            root_path = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(root_path, 'ImageSets/Main', name + '.txt')):
                self.img_ids.append((root_path, line.strip()))

        self.name2id = dict(zip(VOCDataset.CLASSES_NAME, range(len(VOCDataset.CLASSES_NAME))))
        self.id2name = {v: k for k, v in self.name2id.items()}
        self.resize_size = resize_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.train = is_train
        self.augment = augment
        print("INFO=====>voc dataset init finished  ! !")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):

        img_id = self.img_ids[index]
        img = Image.open(os.path.join(img_id[0], 'JPEGImages', img_id[1] + '.jpg'))
        anno = ET.parse(os.path.join(img_id[0], 'Annotations', img_id[1] + '.xml')).getroot()
        boxes = []
        classes = []
        for obj in anno.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.use_difficult and difficult:
                continue
            _box = obj.find("bndbox")
            box = [
                _box.find("xmin").text,
                _box.find("ymin").text,
                _box.find("xmax").text,
                _box.find("ymax").text,
            ]
            TO_REMOVE = 1
            box = tuple(
                map(lambda x: x - TO_REMOVE, list(map(float, box)))
            )
            boxes.append(box)

            name = obj.find("name").text.lower().strip()
            classes.append(self.name2id[name])

        boxes = np.array(boxes, dtype=np.float32)
        if self.train:
            if random.random() < 0.5:
                img, boxes = flip(img, boxes)
            if self.augment is not None:
                img, boxes = self.augment(img, boxes)
        img = np.array(img)
        img, boxes = self.preprocess_img_boxes(img, boxes, self.resize_size)

        img = transforms.ToTensor()(img)
        boxes = torch.from_numpy(boxes)
        classes = torch.LongTensor(classes)

        return {'img': img, 'boxes': boxes, 'classes': classes}

    @staticmethod
    def preprocess_img_boxes(image, boxes, input_ksize):
        """
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        """
        min_side, max_side = input_ksize
        h, w, _ = image.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w = 32 - nw % 32
        pad_h = 32 - nh % 32

        image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes

    def collate_fn(self, sample):
        imgs_list = [s['img'] for s in sample]
        boxes_list = [s['boxes'] for s in sample]
        classes_list = [s['classes'] for s in sample]
        assert len(imgs_list) == len(boxes_list) == len(classes_list)
        batch_size = len(boxes_list)
        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = max(np.array(h_list))
        max_w = max(np.array(w_list))
        for i in range(batch_size):
            img = imgs_list[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(
                torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))

        max_num = 0
        for i in range(batch_size):
            n = boxes_list[i].shape[0]
            if n > max_num: max_num = n
        for i in range(batch_size):
            pad_boxes_list.append(
                torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
            pad_classes_list.append(
                torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)

        return batch_imgs, batch_boxes, batch_classes


@DATASETS.register_module()
class VOCDataset(Dataset):
    def __init__(self, root_dir, set_=None, resize_size=None, use_difficult=False, is_train=True,
                 transform=None, annots_type=None, CLASSES_NAME=None, exclude_cls=None, **kwargs):
        if resize_size is None:
            resize_size = [800, 1333]
        self.root = root_dir
        self.use_difficult = use_difficult
        self.annot_type = annots_type

        self.exclude_cls = exclude_cls

        self.train_set = eval(set_) if isinstance(set_, str) else set_
        self.img_ids = list()
        for (year, name) in self.train_set:
            root_path = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(root_path, 'ImageSets/Main', name + '.txt')):
                self.img_ids.append((root_path, line.strip()))

        self.name2id = dict(zip(CLASSES_NAME, range(len(CLASSES_NAME))))
        # self.name2id = dict(zip(VOCDataset.CLASSES_NAME, range(len(VOCDataset.CLASSES_NAME))))
        self.id2name = {v: k for k, v in self.name2id.items()}
        self.resize_size = resize_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.train = is_train
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):

        img_id = self.img_ids[index]
        img = self.load_image(img_id)
        annot = self.load_annotations(img_id)
        if self.annot_type == 'xywh':
            annot = xyxy2xywh(annot)
        sample = {'img': img, 'annot': annot, 'id': img_id, 'scale': None}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_annotations(self, img_id):
        anno = ET.parse(os.path.join(img_id[0], 'Annotations', img_id[1] + '.xml')).getroot()
        annotations = np.zeros((0, 5))
        for obj in anno.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.use_difficult and difficult:
                continue
            anno = np.zeros((1, 5))
            box = np.zeros((1, 4))
            cls = np.zeros((1, 1))
            _box = obj.find("bndbox")
            name = obj.find("name").text.lower().strip()
            box[0, :4] = [_box.find("xmin").text,
                          _box.find("ymin").text,
                          _box.find("xmax").text,
                          _box.find("ymax").text,
                          ]
            TO_REMOVE = 1
            box = tuple(
                map(lambda x: x - TO_REMOVE, list(map(float, box[0])))
            )
            try:
                cls[0, 0] = self.name2id[name]
                # print(f'{name}:{self.name2id[name]}')
                anno[0, :4] = np.array(box, dtype=np.float32)
                anno[0, 4] = cls
                annotations = np.append(annotations, anno, axis=0)
            # w, h = anno[0, 2]-anno[0, 0], anno[0, 3]-anno[0, 1]
            # with open('G:/pycharm_projects/MMM/wh.txt', 'a') as f:
            #     f.write(f'{w},{h}\n')
            except KeyError:
                continue
        return annotations

    def load_image(self, img_id):
        img_path = os.path.join(img_id[0], 'JPEGImages', img_id[1] + '.jpg')
        img = cv2.imread(img_path)  # (480, 640, 3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    import matplotlib.patches as patches
    from matplotlib.ticker import NullLocator
    from Vision import file, build_dataset, get_by_key
    from collections import ChainMap

    config = file.load('G:/pycharm_projects/MMM/Vision/configs/retinanet_fpn_r50.yaml')

    data_cfg = config.get('dataset')
    train_cfg = ChainMap(data_cfg.get('base'), data_cfg.get('train'))
    eval_cfg = ChainMap(data_cfg.get('base'), data_cfg.get('eval'))
    names = get_by_key(config, 'CLASSES_NAME')

    train_set = [('2007', 'trainval')]
    # eval_dataset = build_dataset(cfg=eval_cfg, transform=Resizer(img_sizes=eval_cfg.get('img_sizes')))
    eval_dataset = VOCDataset(root_dir='I:/DataSets/VOCdevkit_wp', set_=[('2007', 'trainval')], resize_size=None,
                              use_difficult=False, is_train=True,
                              transform=Resizer(img_sizes=eval_cfg.get('img_sizes')), annots_type=None,
                              CLASSES_NAME=names)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_cfg.get('batch_size'),
                                              collate_fn=collate_fn)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, len(names))]

    for num, data in enumerate(eval_loader):
        image = np.asarray(data['img'].squeeze(0).permute(1, 2, 0), dtype='uint8')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        boxes = data['annot'][..., :4].squeeze(0)
        classes = data['annot'][..., -1].squeeze(0)

        for i, box in enumerate(boxes):
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(image, pt1, pt2, tuple(255 * j for j in list(colors[int(classes[i])][:3])))
            # cv2.rectangle(image, pt1, pt2, (255, 0, 0))
            cls = "%s" % (names[int(classes[i])])
            cv2.putText(image, cls, (int(box[0]), int(box[1]) + 20), 0, 1,
                        tuple(255 * j for j in list(colors[int(classes[i])][:3])), 2)
        cv2.imshow('img', image)
        cv2.waitKey(0)

        # plt.figure()
        # fig, ax = plt.subplots(1)
        # ax.imshow(image)
        # for i, box in enumerate(boxes):
        #     pt1 = (int(box[0]), int(box[1]))
        #     pt2 = (int(box[2]), int(box[3]))
        #     img_pad = cv2.rectangle(image, pt1, pt2, (0, 255, 0))
        #     b_color = colors[int(classes[i])]
        #     bbox = patches.Rectangle((box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1], linewidth=1,
        #                              facecolor='none', edgecolor=b_color)
        #     ax.add_patch(bbox)
        #     plt.text(box[0], box[1], s="%s" % (VOCDataset.CLASSES_NAME[int(classes[i])]), color='white',
        #              verticalalignment='top',
        #              bbox={'color': b_color, 'pad': 0})
        # plt.axis('off')
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        # plt.savefig('./out_images/{}'.format(num), bbox_inches='tight', pad_inches=0.0)
        # plt.show()
        # if 0xff == 'q':
        #     plt.close()
