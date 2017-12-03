#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import json
import random
import cv2

import src.utils.visualize

parser = argparse.ArgumentParser(description='AI Challenger')
parser.add_argument('test_results', metavar='FILE',
                    help='path to test_results')
parser.add_argument('test_data_dir', metavar='DIR',
                    help='path to test_data_dir')
parser.add_argument('output_dir', metavar='DIR',
                    help='path to output_dir')

def main():
    global args
    args = parser.parse_args()

    with open(args.test_results, 'rb') as fr:
        data = json.load(fr)
        random.shuffle(data)

        for item in data:

            img_path = os.path.join(args.test_data_dir, item['image_id'] + '.jpg')
            img_np = cv2.imread(img_path)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

            keypoint_dict = item['keypoint_annotations']

            src.utils.visualize.draw_point_on_img_dict(img_np, keypoint_dict)

            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.output_dir, item['image_id']+".jpg"), img_np)



if __name__ == '__main__':
    main()
