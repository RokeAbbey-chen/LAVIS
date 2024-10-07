
"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re
import os
import random as rd
import copy
import os.path as osp
import json
import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionInstructDataset, CaptionEvalDataset
from lavis.datasets.datasets.caption_datasets import CaptionEvalDataset2 as COCOCapEvalDataset2

COCOCapDataset = CaptionDataset
COCOCapInstructDataset = CaptionInstructDataset
from glob import glob


class LatinCapDataset(CaptionDataset):
    LABEL_DELLIMITER= "<del>"
    def __init__(self, vis_processor, text_processor, paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.annotation = []

        pattern = "*/"
        annotations = []
        def get_label_path(bathp: str, imgp: str):
            matcher = re.search(r'(\d+)(?:\.jpg|\.png)', imgp)
            if matcher:
                labp = osp.join(bp, 'labels', f"{matcher.group(1)}.json")
                if osp.exists(labp):
                    return labp
            return None

        def get_label(labp: str):
            with open(labp, 'r', encoding='utf-8') as f:
                d = json.loads(f)
                # label = self.LABEL_DELLIMITER.join(d['text'])
                return d['text']

        image_id = 0
        for basepath in paths:
            dirs = [d for d in os.listdir(basepath) if osp.isdir(osp.join(basepath, d, f'sub{d}'))]
            for d in dirs:
                bp = osp.join(basepath, d)
                img_paths = glob(osp.join(bp, 'imgs/*'))
                anns = []
                for imgp in img_paths:
                    labp = get_label_path(bp, imgp)
                    if labp is None:
                        continue
                    labels = get_label(labp)
                    captions = []
                    for i in range(min(5, len(labels))):
                        copy_ = copy.copy(labels)
                        if i > 0:
                            rd.shuffle(copy_)
                        captions.append(self.LABEL_DELLIMITER.join(copy_))
                        anns.append({'image_path': imgp, 'lable_path': labp, 'caption': copy_, 'captions': captions, 'image_id': image_id})
                    image_id += 1
            annotations.extend(anns)
        self.annotation = annotations
                
                
    def __getitem__(self, index):
        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image_path"])
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            print("image doesn't exist:", image_path)
            return None # image does not exist

        image = self.vis_processor(image)

        return {
            "image": image,
            "text_input": ann['caption'],
            "image_id": ann["image_id"],
            "all_text_input": ann['captions']
        }

