import cv2
import numpy as np
from nms import nms


def nms_fm(detect_fm, group_fm, point_num=14, threshold=0.2, extra_space=9):
    """
    :param detect_fm: fm of detection, shape=[C1, H, W], numpy array
    :param group_fm: fm of grouping, shape=[C2, H, W], numpy array
    :param point_num: the number of considered keypoint
    :param threshold: threshold to select activation, float
    :param extra_space:  extra_space of the nms, int
    :return: list of tuple, ((y, x), score, tag), (y, x) -> coordinate, score -> confidence score, tag -> vector for grouping
    """
    out_list = list()
    for idx in range(point_num):
        out_list.append([])

    cc, yy, xx = nms(detect_fm, threshold, extra_space)
    cc = cc.tolist()
    yy = yy.tolist()
    xx = xx.tolist()
    assert len(yy) == len(xx)
    assert len(cc) == len(xx)
    tmp_len = len(xx)
    for i in xrange(tmp_len):
        c = cc[i]
        y = yy[i]
        x = xx[i]
        score = detect_fm[c, y, x]
        tag = group_fm[c, y, x, :]
        out_list[c].append(((y, x), score, tag))

    return out_list


def group_with_keypoint(in_list, ori_h, ori_w, cur_h, cur_w, threshold=1, min_part_num=3, point_num=14):
    def any_true(check_list):
        for item in check_list:
            for subitem in item:
                if subitem:
                    return True
        return False

    def resize_coord(coord):
        y, x = coord
        y = int(float(y) / cur_h * ori_h + ori_h / cur_h / 2)
        x = int(float(x) / cur_w * ori_w + ori_w / cur_w / 2)
        return (y, x)

    def tag_dis(lhs, rhs):
        return np.linalg.norm(lhs - rhs)

    check_seq = [12, 13, 0, 3, 6, 9, 1, 2, 4, 5, 7, 8, 10, 11]
    check_list = list()
    out_dict = dict()
    human_count = 0
    for idx in range(point_num):
        item_len = len(in_list[idx])
        check_list.append([True] * item_len)

    while (any_true(check_list)):
        human_count += 1
        human_name = "human%d" % (human_count)
        tmp_coords = np.zeros(point_num * 3, dtype=np.int32).reshape(point_num, 3)
        part_count = 0

        finish = False
        for i in check_seq:
            if finish:
                break
            for j in range(len(in_list[i])):
                if check_list[i][j]:
                    cur_coord, score, tag = in_list[i][j]
                    y, x = resize_coord(cur_coord)
                    tmp_coords[i][0] = x
                    tmp_coords[i][1] = y
                    tmp_coords[i][2] = 1
                    check_list[i][j] = False
                    others = [k for k in range(point_num) if k != i]
                    for ii in others:
                        max_score = 0.
                        for jj in range(len(in_list[ii])):
                            if check_list[ii][jj]:
                                cur_coord, sub_score, sub_tag = in_list[ii][jj]
                                yy, xx = resize_coord(cur_coord)
                                if tag_dis(tag, sub_tag) < threshold and check_list[ii][jj] and sub_score > max_score:
                                    max_score = sub_score
                                    tmp_coords[ii][0] = xx
                                    tmp_coords[ii][1] = yy
                                    tmp_coords[ii][2] = 1
                                    check_list[ii][jj] = False
                                    part_count += 1
                    finish = True
                    break
        if part_count >= min_part_num:
            out_dict[human_name] = tmp_coords.reshape(-1).tolist()
    if len(out_dict) == 0:
        out_dict['human1'] = [0] * point_num * 3
    return out_dict


def flip_fm(fm):
    left_right_pair = (
        (0, 3),
        (1, 4),
        (2, 5),
        (6, 9),
        (7, 10),
        (8, 11),
    )
    out_fm = np.empty_like(fm)
    out_fm[12, :, :] = fm[12, :, :]
    out_fm[13, :, :] = fm[13, :, :]

    for idx1, idx2 in left_right_pair:
        out_fm[idx1, :, :] = fm[idx2, :, :]
        out_fm[idx2, :, :] = fm[idx1, :, :]

    out_fm = out_fm[:, :, ::-1]
    return out_fm


def integrate_fm_group(detect_fm_list, group_fm_list, height, width):
    """
    :param detect_fm_list:  list of detect_fm
    :param group_fm_list:  list of group_fm
    :return: integrated detect_fm and group_fm
    """
    resized_detect_list = list()
    resized_group_list = list()

    for fm in detect_fm_list:
        resized_detect_list.append(resize_fm(fm, height, width))

    for fm in group_fm_list:
        tmp = list()
        resized_group_list.append(resize_fm(fm, height, width))
    out_detect_fm = sum(resized_detect_list) / len(resized_detect_list)
    out_group_fm = np.stack(resized_group_list, axis=-1)

    return out_detect_fm, out_group_fm


def resize_fm(fm, dst_h, dst_w):
    """
    :param fm: [C, H, W]
    :param dst_h:
    :param dst_w:
    :return:
    """
    out_list = list()
    for item in fm:
        out_list.append(cv2.resize(item, (dst_w, dst_h)))
    output = np.stack(out_list, axis=0)
    return output

