import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image, ImageDraw as D
from src.utils import transform
from pycocotools.coco import COCO
import torchvision.transforms.functional as FT
from torchvision import transforms as T


class MILDataset(Dataset):
    """
    Class to prepare MIL data
    """

    def __init__(self, datafolder, split):
        self.split =split
        self.datafolder = datafolder
        self.coco = coco_test = COCO(f'{self.datafolder}/{split}.json')
        self.cat_ids = coco_test.getCatIds(catNms=['text', 'figure'])
        self.img_ids = coco_test.getImgIds(catIds=self.cat_ids)

    def __getitem__(self, i):

        img = self.coco.loadImgs(self.img_ids[i])[0]
        image = Image.open(f'{self.datafolder}/data/' + img['file_name'], 'r')
        image = image.convert('RGB')
        ann_ids = self.coco.getAnnIds(imgIds=self.img_ids[i], catIds=self.cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for i in range(len(anns)):
            xmin = anns[i]['bbox'][0]
            ymin = anns[i]['bbox'][1]
            xmax = xmin + anns[i]['bbox'][2]
            ymax = ymin + anns[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(anns[i]['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        #
        # image = FT.resize(image, (300, 300))
        #
        # transform = T.Compose([
        #     T.PILToTensor()
        # ])

        # transform = transforms.PILToTensor()
        # Convert the PIL image to Torch tensor
        # img_tensor = transform(image)

        # print the converted Torch tensor
        # print(img_tensor)

        # image = self.get_transform(True)(image)
        image, boxes, labels = transform(image, boxes, labels, split=self.split)

        return image, boxes, labels #, difficulties

    def __len__(self):
        return len(self.img_ids)

    def get_transform(self, train):

        transforms = []
        # random flip
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        # normalization and resize to SSD input shape
        transforms.extend(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        return T.Compose(transforms)

    def plot_image_w_bb(self, i, resized=False):

        cat_color = {1: (255, 0, 0), 2: (0, 0, 255)}

        img = self.coco.loadImgs(self.img_ids[i])[0]
        W = img['width']
        H = img['height']


        image = Image.open(f'{self.datafolder}/data/' + img['file_name'], 'r')
        image = image.convert('RGB')
        if resized:
            image = FT.resize(image, (300, 300))
            W = 300
            H = 300
        print(f'hight = {H}, wight={W}, img_id={self.img_ids[i]}')
        ann_ids = self.coco.getAnnIds(imgIds=self.img_ids[i], catIds=self.cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        draw = D.Draw(image)
        for k in range(len(anns)):
            xmin = anns[k]['bbox'][0]
            ymin = anns[k]['bbox'][1]
            xmax = xmin + anns[k]['bbox'][2]
            ymax = ymin + anns[k]['bbox'][3]
            draw.rectangle([(int(xmin * W), int(ymin * H)), (int(xmax * W), int(ymax * H))], outline=cat_color[anns[k]['category_id']],
                           width=5)

        return image

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()


        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])


        images = torch.stack(images, dim=0)

        return images, boxes, labels  # tensor (N, 3, 300, 300), 3 lists of N tensors each






