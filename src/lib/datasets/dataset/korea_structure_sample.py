from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

class KOREA_STRUCTURE(data.Dataset): 
    num_classes = 23
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                    dtype=np.float32).reshape(1, 1, 3)
                    
    def __init__(self, opt, split):
        super(KOREA_STRUCTURE, self).__init__() 
        self.data_dir = opt.data_dir
        if split == 'test' or split == 'val':
            self.annot_path = opt.eval_json_dir
            self.img_dir = os.path.join(self.data_dir, 'Eval')
        else:
            self.annot_path = opt.json_dir
            self.img_dir = os.path.join(self.data_dir, 'Train')

        self.max_objs = 128 
        self.class_name = ['__background__', 'HF010580', 'HF019337', 'HF019291', 
                            'HF030004', 'HF019257', 'HF019251', 'HF019248', 
                            'HF010566', 'HF010431', 'HF030019', 'HF010211', 
                            'HF019036', 'HF010623', 'HF030029', 'HF030060', 
                            'HF010490', 'HF019298', 'HF010377', 'HF010632', 
                            'HF010221', 'HF010570', 'HF030048', 'HF010609']

        # self.class_name = ['__background__', '대웅전', '경주귀후재', '경주사마소', 
        #                     '롯데월드타워', '불국사당간지주', '삼괴정', '이씨삼강묘비', 
        #                     '경교장', '정구중가옥', '국립세종도서관', '여주신륵사보제존자석종앞석등', 
        #                     '경주석빙고', '숭례문', '백남준아트센터', '메타폴리스', 
        #                     '거돈사지원공국사탑', '분황사석정', '이후원묘역', '명동성당', 
        #                     '서희장군묘', '흥화문', '스카이타워', '계동배렴가옥']

        self._valid_ids = [1, 2, 3, 
                           4, 5, 6, 7, 
                           8, 9, 10, 11, 
                           12, 13, 14, 15, 
                           16, 17, 18, 19, 
                           20, 21, 22, 23]
        
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
    
        self.split = split
        self.opt = opt

        print('==> initializing xray {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    score = bbox[4]
                    if score < 0.5: continue
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    bbox_out  = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))
  
    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()