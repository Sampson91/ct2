#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/8/24 下午2:46
# @Author : wangyangyang
import os

from PIL import Image


def concat_images(name, input_path, output_path, counts=12, ROW=2, COL=6, item_width=192):
    """
    input image name,need concat image path and output image path,
    concat image orientation way is from left to right and up to down.
    """
    image_files = []
    for index in range(counts):
        image_files.append(Image.open(input_path + name + "-" + str(item_width) + "-" + str(index) + ".png"))
    target = Image.new('RGB', (item_width * COL, item_width * ROW))

    for row in range(ROW):
        for col in range(COL):
            # image concat
            target.paste(image_files[COL * row + col], (0 + item_width * col, 0 + item_width * row))
    target.save(output_path + "/" + name + '.png', quality=100)


def cut_image_save(image, split_image_saving_path, save_name, item_width=192, gray_image_save_path=None):
    """
    input image file,need split image path and new file name,
    split image orientation way is from left to right and up to down.
    """
    width, height = image.size
    box_list = []
    width_count = int(width / item_width)
    height_count = int(height / item_width)
    # (left, upper, right, lower)
    for i in range(0, height_count):
        for j in range(0, width_count):
            box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
            box_list.append(box)

    image_list = [image.crop(box) for box in box_list]
    count = height_count * width_count
    for i in range(count):
        save_image_path = split_image_saving_path + "/" + str(int(save_name)) + "-" + str(item_width) + "-" + str(i) + ".png"
        image_list[i].save(save_image_path, quality=100)
        if gray_image_save_path:
            save_gray_image_path = gray_image_save_path + save_name + str(item_width) + str(i) + ".png"
            image_gray = image_list[i].convert('L')
            image_gray.save(save_gray_image_path, quality=100)


def process_image_split(origin_image_path, image_split_save_path, item_width=192):
    obj_files = os.listdir(origin_image_path)
    if not os.path.exists(image_split_save_path):
        os.mkdir(image_split_save_path)
    for obj_file in obj_files:
        img = Image.open(origin_image_path + obj_file)
        save_name = obj_file.split(".")[0]
        cut_image_save(img, image_split_save_path, save_name, item_width)


def process_test_image_concat(origin_image_path, dataset_dir, save_image_concat_dir, item_width=185):
    if not os.path.exists(save_image_concat_dir):
        os.mkdir(save_image_concat_dir)
    image_names = os.listdir(dataset_dir)
    name_end = image_names[0].split(".")[1]
    image_name = list(set([name.split("-" + str(item_width) + "-")[0] for name in image_names]))
    image = Image.open(origin_image_path + image_name[0] + "." + name_end)
    width, height = image.size
    COL = int(width / item_width)
    ROW = int(height / item_width)
    for name in image_name:
        name_list = [n for n in image_names if name == n.split("-" + str(item_width) + "-")[0]]
        counts = len(name_list)
        counts == ROW * COL
        if counts == ROW * COL:
            concat_images(name, dataset_dir, save_image_concat_dir, counts, ROW, COL,item_width)
        else:
            continue


if __name__ == '__main__':
    path = "/media/yang/sys/kitt2_train_data/data/img_00/img/"
    image_split_saving_path = "/media/yang/sys/kitt2_train_data/data/img_00/img_split/"
    gray_image_saving_path = "/media/yang/sys/kitt2_train_data/data/img_00/gray_img/"

    # obj_files = os.listdir(path)
    # for obj_file in obj_files:
    #     img=Image.open(path + obj_file)
    #     save_name = obj_file.split(".")[0]
    #     cut_image_save(img,image_split_saving_path, gray_image_saving_path, save_name)

    # image concat
    input_path = "/media/yang/sys/kitt2_train_data/data/img_00/output_color_img/"
    save_concat = "/media/yang/sys/kitt2_train_data/data/img_00/img_concat/"
    image_names = os.listdir(input_path)
    UNIT_WIDTH_SIZE = 192
    UNIT_HEIGHT_SIZE = 192
    process_test_image_concat(path, image_split_saving_path, save_concat)

    image_name = list(set([name.split("-")[0] for name in image_names]))
    image = Image.open(path + image_name[0] + ".png")
    width, height = image.size
    COL = int(width / UNIT_WIDTH_SIZE)
    ROW = int(height / UNIT_HEIGHT_SIZE)
    for name in image_name:
        name_list = [n for n in image_names if name in n]
        counts = len(name_list)
        if counts == ROW * COL:
            concat_images(name, input_path, save_concat, counts, ROW, COL)
        else:
            continue
