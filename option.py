import os
import argparse


class Option():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='parser for learning-to-see-in-the-dark-mxnet')
        subparsers = self.parser.add_subparsers(title='subparsers', dest='subparser')

        train_args = subparsers.add_parser("train", help="parser for training")
        train_args.add_argument("--epochs", type=int, default=4000,
                                help="number of training epochs, default is 4000")
        train_args.add_argument("--lr", type=float, default=1e-4,
                                help="learning rate, default is 1e-4")
        train_args.add_argument("--batch_size", type=int, default=1,
                                help="batch size for training, default is 1")
        train_args.add_argument("--dataset", type=str, choices=["Sony", "Fuji"], default="Sony",
                                help="name of dataset to train, Sony or Fuji")
        train_args.add_argument("--patch_size", type=int, default=512,
                                help="patch size for training, default is 512")
        train_args.add_argument("--save_model_dir", type=str, default="models/",
                                help="directory to save models, default is models/")
        train_args.add_argument("--save_freq", type=int, default=100,
                                help="number of epochs to save models, default is 500")
        train_args.add_argument("--gpu", type=bool, default=False,
                                help="set True for using gpu, False for cpu")
        train_args.add_argument("--seed", type=int, default=666,
                                help="random seed for training")

        test_args = subparsers.add_parser("test", help="parser for test")
        test_args.add_argument("--dataset", type=str, choices=["Sony", "Fuji"], default="Sony",
                                help="name of dataset to train, Sony or Fuji")
        test_args.add_argument("--model", type=str, default="models/",
                                help="path of model to load, default is models/")
        test_args.add_argument("--gpu", type=bool, default=False,
                                help="set True for using gpu, False for cpu")

    def parse(self):
        return self.parser.parse_args()