import os
import math
import numpy as np
import scipy
import mxnet as mx
from mxnet import gluon, autograd, initializer
from mxnet import profiler
import mxnet.ndarray as F

import net
import utils
import data
from option import Option


metric = mx.metric.MSE()
profiler.set_config(profile_all=True, aggregate_stats=True, filename='profile_output.json')


def train(args):
    np.random.seed(args.seed)
    if args.gpu:
        ctx = [mx.gpu(0)]
    else:
        ctx = [mx.cpu(0)]
    if args.dataset == "Sony":
        out_channels = 12
        scale = 2
    else:
        out_channels = 27
        scale = 3

    # load data
    train_transform = utils.Compose([utils.RandomCrop(args.patch_size, scale), utils.RandomFlipLeftRight(),
                                     utils.RandomFlipTopBottom(), utils.RandomTranspose(), utils.ToTensor(), ])
    train_dataset = data.MyDataset(args.dataset, "train", transform=train_transform)
    val_transform = utils.Compose([utils.ToTensor()])
    val_dataset = data.MyDataset(args.dataset, "val", transform=val_transform)
    train_loader = gluon.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, last_batch='rollover')
    val_loader = gluon.data.DataLoader(val_dataset, batch_size=1, last_batch='discard')
    unet = net.UNet(out_channels, scale)
    unet.initialize(init=initializer.Xavier(), ctx=ctx)

    # optimizer and loss
    trainer = gluon.Trainer(unet.collect_params(), 'adam', {'learning_rate': args.lr})
    l1_loss = gluon.loss.L1Loss()

    print "Start training now.."
    for i in range(args.epochs):
        total_loss = 0
        count = 0
        profiler.set_state('run')
        for batch_id, (img, gt) in enumerate(train_loader):
            batch_size = img.shape[0]
            count += batch_size
            img_list = gluon.utils.split_and_load(img[0], ctx)
            gt_list = gluon.utils.split_and_load(gt[0], ctx)
            with autograd.record():
                preds = [unet(x) for x in img_list]
                losses = []
                for ii in range(len(preds)):
                    loss = l1_loss(gt_list[ii], preds[ii])
                    losses.append(loss)
            for loss in losses:
                loss.backward()
            total_loss += sum([l.sum().asscalar() for l in losses])
            avg_loss = total_loss / count
            trainer.step(batch_size)
            metric.update(gt_list, preds)
            F.waitall()
            profiler.set_state('stop')
            print profiler.dumps()
            break
            gt_save = gt_list[0]
            output_save = preds[0]

            if (batch_id + 1) % 100 == 0:
                message = "Epoch {}: [{}/{}]: l1_loss: {:.4f}".format(i+1, count, len(train_dataset),
                                                                      avg_loss)
                print message
        temp = F.concat(gt_save, output_save, dim=3)
        temp = temp.asnumpy().reshape(temp.shape[2], temp.shape[3], 3)
        scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255, mode='RGB').save(
            args.save_model_dir + '%04d_%05d_00_train.jpg' % (i + 1, count))

        # evaluate
        batches = 0
        avg_psnr = 0.
        for img, gt in val_loader:
            batches += 1
            imgs = gluon.utils.split_and_load(img[0], ctx)
            label = gluon.utils.split_and_load(gt[0], ctx)
            outputs = []
            for x in imgs:
                outputs.append(unet(x))
            metric.update(label, outputs)
            avg_psnr += 10 * math.log10(1 / metric.get()[1])
            metric.reset()
        avg_psnr /= batches
        print('Epoch {}: validation avg psnr: {:.3f}'.format(i+1, avg_psnr))

        # save model
        if (i + 1) % args.save_freq == 0:
            save_model_filename = "Epoch_" + str(i + 1) + ".params"
            save_model_path = os.path.join(args.save_model_dir, save_model_filename)
            unet.save_params(save_model_path)
            print("\nCheckpoint, trained model saved at", save_model_path)


    # save model
    save_model_filename = "Final_Epoch_" + str(i + 1) + ".params"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    unet.save_params(save_model_path)
    print("\nCheckpoint, trained model saved at", save_model_path)


def test(args):
    if args.gpu:
        ctx = [mx.gpu(0)]
    else:
        ctx = [mx.cpu(0)]
    if args.dataset == "Sony":
        out_channels = 12
        scale = 2
    else:
        out_channels = 27
        scale = 3

    # load data
    test_transform = utils.Compose([utils.ToTensor()])
    test_dataset = data.MyDataset(args.dataset, "test", transform=test_transform)
    test_loader = gluon.data.DataLoader(test_dataset, batch_size=1, last_batch='discard')
    unet = net.UNet(out_channels, scale)
    unet.load_params(args.model, ctx=ctx)
    batches = 0
    avg_psnr = 0.
    for img, gt in test_loader:
        batches += 1
        imgs = gluon.utils.split_and_load(img[0], ctx)
        label = gluon.utils.split_and_load(gt[0], ctx)
        outputs = []
        for x in imgs:
            outputs.append(unet(x))
        metric.update(label, outputs)
        avg_psnr += 10 * math.log10(1 / metric.get()[1])
        metric.reset()
    avg_psnr /= batches
    print('Test avg psnr: {:.3f}'.format(avg_psnr))


def main():
    args = Option().parse()
    if args.subparser is None:
        raise ValueError("ERROR: you should specify the type, train or test")
    elif args.subparser == "train":
        train(args)
    elif args.subparser == "test":
        test(args)
    else:
        raise ValueError("ERROR: Unkown type")


if __name__ == '__main__':
    main()