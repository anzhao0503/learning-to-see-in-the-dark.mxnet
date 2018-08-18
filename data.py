import os
import mxnet.gluon.data as data
import numpy as np
import rawpy
import mxnet.ndarray as F
# import utils
# from PIL import Image


def pack_raw(raw, dataset):
    '''
    :param dataset: Sony for Bayer
                    Fuji for X-Trans
    '''
    im = raw.raw_image_visible.astype(np.float32)
    if dataset == 'Sony':
        im = np.maximum(im - 512, 0) / (16383 - 512)  # substract the black level
        img_shape = im.shape
        im = im[:, :, np.newaxis]
        H, W = (img_shape[0]//2) * 2, (img_shape[1]//2) * 2
        out = np.concatenate((im[0:H:2, 0:W:2, :], im[0:H:2, 1:W:2, :],
                              im[1:H:2, 1:W:2, :], im[1:H:2, 0:W:2, :]),
                             axis=2)
    elif dataset == 'Fuji':
        im = np.maximum(im - 1024, 0) / (16383 - 1024)  # substract the black level
        img_shape = im.shape
        H, W = (img_shape[0]//6) * 6, (img_shape[1]//6) * 6
        out = np.zeros((H//3, W//3, 9))
        # 0 R
        out[0::2, 0::2, 0] = im[0:H:6, 0:W:6]
        out[0::2, 1::2, 0] = im[0:H:6, 4:W:6]
        out[1::2, 0::2, 0] = im[3:H:6, 1:W:6]
        out[1::2, 1::2, 0] = im[3:H:6, 3:W:6]

        # 1 G
        out[0::2, 0::2, 1] = im[0:H:6, 2:W:6]
        out[0::2, 1::2, 1] = im[0:H:6, 5:W:6]
        out[1::2, 0::2, 1] = im[3:H:6, 2:W:6]
        out[1::2, 1::2, 1] = im[3:H:6, 5:W:6]

        # 1 B
        out[0::2, 0::2, 2] = im[0:H:6, 1:W:6]
        out[0::2, 1::2, 2] = im[0:H:6, 3:W:6]
        out[1::2, 0::2, 2] = im[3:H:6, 0:W:6]
        out[1::2, 1::2, 2] = im[3:H:6, 4:W:6]

        # 4 R
        out[0::2, 0::2, 3] = im[1:H:6, 2:W:6]
        out[0::2, 1::2, 3] = im[2:H:6, 5:W:6]
        out[1::2, 0::2, 3] = im[5:H:6, 2:W:6]
        out[1::2, 1::2, 3] = im[4:H:6, 5:W:6]

        # 5 B
        out[0::2, 0::2, 4] = im[2:H:6, 2:W:6]
        out[0::2, 1::2, 4] = im[1:H:6, 5:W:6]
        out[1::2, 0::2, 4] = im[4:H:6, 2:W:6]
        out[1::2, 1::2, 4] = im[5:H:6, 5:W:6]

        out[:, :, 5] = im[1:H:3, 0:W:3]
        out[:, :, 6] = im[1:H:3, 1:W:3]
        out[:, :, 7] = im[2:H:3, 0:W:3]
        out[:, :, 8] = im[2:H:3, 1:W:3]
    else:
        pass
    return out


def process_img(img_path, dataset):
    '''
    :param dataset: Sony or Fuji
    :return: numpy array
    '''
    raw = rawpy.imread(img_path)
    img = pack_raw(raw, dataset)
    return img


def process_gt(gt_path):
    raw = rawpy.imread(gt_path)
    im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    im = np.float32(im / 65535.0)
    return im


def load_dataset_all(dataset_dir, dataset, mode, imgs, gts):
    filename = os.path.join(dataset_dir, '{0}_{1}_list.txt'.format(dataset, mode))
    fw = open(filename, 'r')
    lines = fw.readlines()
    for line in lines:
        img_path = line.split()[0]
        gt_path = line.split()[1]
        in_exposure = float(img_path.split('/')[-1][9:-5])
        gt_exposure = float(gt_path.split('/')[-1][9:-5])
        ratio = min(gt_exposure / in_exposure, 300)
        img = process_img(os.path.join(dataset_dir, img_path), dataset)
        img = img[np.newaxis, :] * ratio
        gt = process_gt(os.path.join(dataset_dir, gt_path))
        gt = gt[np.newaxis, :]
        imgs.append(img)
        gts.append(gt)
    fw.close()


def load_dataset(dataset_dir, dataset, mode):
    filename = os.path.join(dataset_dir, '{0}_{1}_list.txt'.format(dataset, mode))
    fw = open(filename, 'r')
    lines = fw.readlines()
    img_paths = []
    gt_paths = []
    for line in lines:
        img_paths.append(line.split()[0])
        gt_paths.append(line.split()[1])
    fw.close()
    return img_paths, gt_paths


class MyDataset(data.Dataset):
    '''
    :param dataset: Sony or Fuji
    :param mode: train or val or test
    :param transform: None for val or test
    :return: image and groundtruth with the type of numpy array
    '''
    def __init__(self, dataset, mode, transform=None):
        self.dataset = dataset
        self.dataset_dir = './dataset'
        self.mode = mode
        self.transform = transform
        self.img_paths = []
        self.gt_paths = []
        self.img_paths, self.gt_paths = load_dataset(self.dataset_dir, self.dataset, self.mode)
        # load_dataset_all(self.dataset_dir, self.dataset, self.mode, self.imgs, self.gts)
        # print 'Loading {} dataset to memory is ok...'.format(self.mode)

    def __getitem__(self, index):
        img_path, gt_path = self.img_paths[index], self.gt_paths[index]
        in_exposure = float(img_path.split('/')[-1][9:-5])
        gt_exposure = float(gt_path.split('/')[-1][9:-5])
        ratio = min(gt_exposure / in_exposure, 300)
        img = process_img(os.path.join(self.dataset_dir, img_path), self.dataset)
        img = img[np.newaxis, :] * ratio
        gt = process_gt(os.path.join(self.dataset_dir, gt_path))
        gt = gt[np.newaxis, :]
        if self.transform is not None:
            img, gt = self.transform(img, gt)
        img = F.minimum(img, 1.0)
        return img, gt

    def __len__(self):
        return len(self.img_paths)


# class MyDataset(data.Dataset):
#     '''
#     :param dataset: Sony or Fuji
#     :param mode: train or val or test
#     :param transform: None for val or test
#     :return: image and groundtruth with the type of numpy array
#
#     It takes a long time to load raw data. Keep them in memory.
#     '''
#     def __init__(self, dataset, mode, transform=None):
#         self.dataset = dataset
#         self.dataset_dir = './dataset'
#         self.mode = mode
#         self.transform = transform
#         self.imgs = []
#         self.gts = []
#         load_dataset_all(self.dataset_dir, self.dataset, self.mode, self.imgs, self.gts)
#         print 'Loading {} dataset to memory is ok...'.format(self.mode)
#
#     def __getitem__(self, index):
#         img, gt = self.imgs[index], self.gts[index]
#         if self.transform is not None:
#             img, gt = self.transform(img, gt)
#         img = F.minimum(img, 1.0)
#         return img, gt
#
#     def __len__(self):
#         return len(self.imgs)

# if __name__ == '__main__':
#     transform = utils.Compose([utils.RandomCrop(512, 3), utils.RandomFlipLeftRight(), utils.RandomFlipTopBottom(),
#                                utils.RandomTranspose(), ])
#     test_dataset = MyDataset('Fuji', 'test', transform=transform)
#     test_loader = data.DataLoader(test_dataset, batch_size=1, last_batch='discard')
#     for _, (img, gt) in enumerate(test_loader):
#         gt = gt[0, :, :, :].asnumpy() * 255
#         gt_array = np.zeros((gt.shape[1], gt.shape[2], gt.shape[3]))
#         gt_array = gt[0]
#         gt_img = Image.fromarray(np.uint8(gt_array))
#         gt_img.save('gt.png')
#         break
