#!/usr/bin/env python
# encoding: utf-8

import argparse
import json
import math

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import src.models.hourglass_ae
import src.dataset.reader
import src.utils.aux

parser = argparse.ArgumentParser(description='AI Challenger Keypoints Detection')
parser.add_argument('test_data_dir', metavar='DIR',
                    help='path to test_data_dir')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--checkpoint_path', default='./results/hg_ae.pth.tar', type=str, metavar='PATH',
                    help='path to checkpoint')
parser.add_argument('--test_results', default='./results/multi_test_results.json', type=str, metavar='PATH',
                    help='path to test results.')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to checkpoint')

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    model = src.models.hourglass_ae.hg(f=256, num_stacks=3, num_classes=14, embedding_len=14)

    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])

    all_model = (model, )

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.4798, 0.4517, 0.4220],
                                    std=[0.2558, 0.2481, 0.2468])
    test_dataset = src.dataset.reader.TestReader(
        data_dir=args.test_data_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test(test_loader, all_model)

def test(test_loader, models):
    # switch to evaluate mode
    for model in models:
        model.eval()

    out_data = list()
    print "Testing..."
    for i, (img, cur_size, ori_size, img_id, img_path) in enumerate(test_loader):
        img_id = img_id[0]

        ori_h, ori_w = ori_size
        ori_h = ori_h.numpy()[0]
        ori_w = ori_w.numpy()[0]

        cur_h, cur_w = cur_size
        cur_h = cur_h.numpy()[0]
        cur_w = cur_w.numpy()[0]

        detect_list = list()
        group_list = list()
        for model in models:
            n = len(img)
            for img_idx, input_img in enumerate(img):
                input_img = input_img.cuda(async=True)
                img_var = torch.autograd.Variable(input_img, volatile=True)

                output = model(img_var)
                output = output[-1]
                out_detect = output[0]
                out_group = output[1]

                out_detect = out_detect.data.cpu().numpy()[0]
                out_group = out_group.data.cpu().numpy()[0]
                if img_idx >= n/2:
                    # The back half of the images is flipped
                    out_detect = src.utils.aux.flip_fm(out_detect)
                    out_group = src.utils.aux.flip_fm(out_group)
                detect_list.append(out_detect)
                group_list.append(out_group)

        out_detect, out_group = src.utils.aux.integrate_fm_group(detect_list, group_list, cur_h, cur_w)

        out_list = src.utils.aux.nms_fm(out_detect, out_group, threshold=0.3, extra_space=5)
        keypoint_annos = src.utils.aux.group_with_keypoint(out_list, ori_h, ori_w,
                                                       cur_h, cur_w, threshold=0.5 * math.sqrt(n * len(models)))
        if (i + 1) % 1000 == 0:
            print ("finish %d images" % (i+1))

        item = dict()
        item['image_id'] = img_id
        item['keypoint_annotations'] = keypoint_annos
        out_data.append(item)

    print ("Saving results at %s" % args.test_results)
    with open(args.test_results, 'wb') as fw:
        json.dump(out_data, fw)

    return

if __name__ == '__main__':
    main()
