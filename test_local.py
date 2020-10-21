# toooest %matplotlib inline
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch import utils
from torchvision import datasets
# from network.localize_net import Localize_hanson
from util import util
from config import GlobalConfig
# from network.reversible_image_net_hide import RNet_test
from network.reversible_image_net_hanson import ReversibleImageNetwork_hanson

if __name__ =='__main__':
    # Setting
    config = GlobalConfig()
    isSelfRecovery = True
    skipTraining = config.skipTraining

    device = config.device
    print(device)
    # Hyper Parameters
    num_epochs = 1
    train_batch_size = config.train_batch_size
    test_batch_size = config.test_batch_size
    learning_rate = config.learning_rate
    use_Vgg = config.useVgg
    use_dataset = config.use_dataset
    # beta = config.beta
    # if use_Vgg:
    #     beta = 10

    MODELS_PATH = config.MODELS_PATH
    VALID_PATH = config.VALID_PATH
    TRAIN_PATH = config.TRAIN_PATH
    TEST_PATH = config.TRAIN_PATH

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)


    def test(net, train_loader, config):


        for epoch in range(1):
            # train
            for idx, train_batch in enumerate(train_loader):
                data, _ = train_batch
                train_covers = data.to(device)
                pred_label, CropoutWithCover = net.test_local(train_covers)

                for i in range(pred_label.shape[0]):
                    # util.save_images(p7_final[i].cpu(),
                    #                  'epoch-{0}-recovery-batch-{1}-{2}_after7.png'.format(epoch, idx, i),
                    #                  './Images/recovery',
                    #                  std=config.std,
                    #                  mean=config.mean)
                    # util.save_images(p5_final[i].cpu(),
                    #                  'epoch-{0}-recovery-batch-{1}-{2}_after5.png'.format(epoch, idx, i),
                    #                  './Images/recovery',
                    #                  std=config.std,
                    #                  mean=config.mean)
                    util.save_images(pred_label[i].cpu(),
                                     'localized-{1}-{2}.png'.format(epoch, idx, i),
                                     './Images/localized',)
                    util.save_images(CropoutWithCover[i].cpu(),
                                     'GroundTruth-{1}-{2}.png'.format(epoch, idx, i),
                                     './Images/localized',
                                     std=config.std,
                                     mean=config.mean)
                if idx>16:
                    break
                    # util.save_images(x_recover[i].cpu(),
                    #                  'epoch-{0}-recovery-batch-{1}-{2}.png'.format(epoch, idx, i),
                    #                  './Images/recovery',
                    #                  std=config.std,
                    #                  mean=config.mean)
                    # util.save_images(x_hidden[i].cpu(),
                    #                  'epoch-{0}-hidden-batch-{1}-{2}.png'.format(epoch, idx, i),
                    #                  './Images/hidden',
                    #                  std=config.std,
                    #                  mean=config.mean)
                    # util.save_images(train_covers[i].cpu(),
                    #                  'epoch-{0}-covers-batch-{1}-{2}.png'.format(epoch, idx, i),
                    #                  './Images/cover',
                    #                  std=config.std,
                    #                  mean=config.mean)

    # ------------------------------------ Begin ---------------------------------------
    # Creates net object
    net = ReversibleImageNetwork_hanson(username="hanson", config=config)

    # Creates training set
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            TEST_PATH,
            transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.mean,
                                     std=config.std),

            ])), batch_size=1, num_workers=1,
        pin_memory=True, shuffle=True, drop_last=True)

    # Creates test set
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(
    #         TEST_PATH,
    #         transforms.Compose([
    #             transforms.Scale(config.Width),
    #             transforms.RandomCrop(config.Width),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=config.mean,
    #                                  std=config.std)
    #         ])), batch_size=test_batch_size, num_workers=1,
    #     pin_memory=True, shuffle=True, drop_last=True)

    """ Begin Pre-Training """
    # if not config.skipPreTraining:
    #     net, hist_loss_localization, hist_loss_cover, hist_loss_recover, hist_loss_discriminator_enc, hist_loss_discriminator_recovery \
    #         = pretrain(net, train_loader, config)
    #     # Plot loss through epochs
    #     util.plt_plot(hist_loss_cover)
    #     util.plt_plot(hist_loss_recover)
    #     util.plt_plot(hist_loss_discriminator_enc)
    #     util.plt_plot(hist_loss_discriminator_recovery)
    # else:
    #     net.load_state_dict_pretrain(MODELS_PATH)
        # net.load_state_dict_Discriminator(torch.load(MODELS_PATH + 'Epoch N' + config.loadfromEpochNum))

    if not config.skipMainTraining:
        net.load_model(MODELS_PATH + 'Epoch N1')
        # net.load_state_dict_all(MODELS_PATH + 'Epoch N1')
        test(net, test_loader, config)
        # Plot loss through epochs
        # util.plt_plot(hist_loss_cover)
        # util.plt_plot(hist_loss_recover)
        # util.plt_plot(hist_loss_discriminator_enc)
        # util.plt_plot(hist_loss_discriminator_recovery)
