import os
import cv2
import shutil
import glob as gb

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image


def clahe_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[3] == 1)  # check the channel is 1
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, :, :, 0] = clahe.apply(np.array(imgs[i, :, :, 0], dtype=np.uint8))
    return imgs_equalized


def enhance(image_path, clip_limit=3):
    image = cv2.imread(image_path)
    # convert image to LAB color model
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # split the image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    # apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L channel with the original A and B channel
    merged_channels = cv2.merge((cl, a_channel, b_channel))

    # convert iamge from LAB color model back to RGB color model
    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    # return cv2_to_pil(final_image)
    return final_image


def kaggle_processing(a):

    scale = np.size(a, 0)
    b = np.zeros(a.shape)
    cv2.circle(b, (int(a.shape[1] / 2), int(a.shape[0] / 2)),
               int(scale / 2 * 1.6), (1, 1, 1), -1, 8, 0)
    aa = cv2.addWeighted(a, 4, cv2.GaussianBlur(
        a, (0, 0), scale / 30), -4, 128) * b + 128 * (1 - b)
    return aa


def enhance_cv2(image_path, clip_limit=3):
    return pil_to_cv2(enhance(image_path, clip_limit=clip_limit))


def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def pil_to_cv2(pil_image):
    cv2_image = np.array(pil_image)
    cv2_image = cv2_image[:, :, ::-1].copy()
    return cv2_image
##


def mkdir_if_not_exist(dir_name, is_delete=False):
    """
    创建文件夹
    create dir
    :param dir_name: 文件夹列表
    :param is_delete: 是否删除
    :return: 是否成功
    """
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print(u'[INFO] Dir "%s" exists, deleting.' % dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(u'[INFO] Dir "%s" not exists, creating.' % dir_name)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False


def write_images(imgs, save_path, orgList, clear=True):
    for index in range(len(orgList)):
        FileName = orgList[index]
        (f_name, fe_name) = os.path.splitext(FileName)
        (img_dir, temp_filename) = os.path.split(FileName)

        cv2.imwrite(save_path + temp_filename, imgs[index, :, :, :])

    if clear:
        del imgs


if __name__ == "__main__":
    ImgDir = './origin/'
    gray_path = './out/gray/'
    rg_path = './out/rg/'
    datanorm_gray_path = './out/norm_gray/'
    datanorm_rg_path = './out/norm_rg/'
    datanorm_col_path = './out/norm_color/'
    clahe_outPath = './out/clahe_color/'
    clahegray_outPath = './out/clahe_gray/'
    claherg_outPath = './out/clahe_rg/'
    gamma_rg_path = './out/gamma_rg/'
    gamma_gray_path = './out/gamma_gray/'
    gussian_path = './out/gussian_enhance/'

    orgList = gb.glob(ImgDir + "*." + 'jpg')  # 文件名列表 original image filename list

    mkdir_if_not_exist(gray_path)
    mkdir_if_not_exist(rg_path)
    mkdir_if_not_exist(datanorm_gray_path)
    mkdir_if_not_exist(datanorm_rg_path)
    mkdir_if_not_exist(datanorm_col_path)
    mkdir_if_not_exist(clahe_outPath)
    mkdir_if_not_exist(clahegray_outPath)
    mkdir_if_not_exist(claherg_outPath)
    mkdir_if_not_exist(gamma_rg_path)
    mkdir_if_not_exist(gamma_gray_path)
    mkdir_if_not_exist(gussian_path)

    imgs = []
    rg_imgs = []
    gray_imgs = []
    for index in range(len(orgList)):
        orgPath = orgList[index]
        orgImg_rgb = plt.imread(orgPath)
        # img_gray = cv2.cvtColor(orgImg,cv2.COLOR_RGB2GRAY)
        orgImg_bgr = cv2.imread(orgPath)
        img_gray = cv2.cvtColor(orgImg_bgr, cv2.COLOR_BGR2GRAY)

        imgs.append(orgImg_bgr.copy())
        rg_imgs.append((orgImg_rgb[:, :, 1] * 0.75 + orgImg_rgb[:, :, 0] * 0.25)[:, :, np.newaxis].copy())
        gray_imgs.append(img_gray[:, :, np.newaxis].copy())

    imgs = np.array(imgs)
    rg_imgs = np.array(rg_imgs)
    gray_imgs = np.array(gray_imgs)
    print(imgs.shape)
    print(rg_imgs.shape)
    print(gray_imgs.shape)

    # ===== clahe
    clahe_imgs_gray = clahe_equalized(rg_imgs)
    write_images(clahe_imgs_gray, clahegray_outPath, orgList)
    clahe_imgs_rg = clahe_equalized(gray_imgs)
    write_images(clahe_imgs_rg, claherg_outPath, orgList)

    for FileName in orgList:
        #fov(ImgDir, FileName, outPath)
        img = cv2.imread(FileName)
        (f_name, fe_name) = os.path.splitext(FileName)
        (img_dir, temp_filename) = os.path.split(FileName)

        # img_rgb_equalized = histo_equalized_rgb(img)
        clahe_img = enhance(FileName, clip_limit=3)
        cv2.imwrite(clahe_outPath + temp_filename, clahe_img)


# gussian
    for FileName in orgList:
        #fov(ImgDir, FileName, outPath)
        img = cv2.imread(FileName)
        (f_name, fe_name) = os.path.splitext(FileName)
        (img_dir, temp_filename) = os.path.split(FileName)

        # img_rgb_equalized = histo_equalized_rgb(img)
        gussian_img = kaggle_processing(img)
        cv2.imwrite(gussian_path + temp_filename, gussian_img)
