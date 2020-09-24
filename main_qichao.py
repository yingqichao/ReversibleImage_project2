# toooest %matplotlib inline
import os
from loss.vgg_loss import VGGLoss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import utils
from torchvision import datasets, utils
from network.reversible_image_net import ReversibleImageNetwork
from config import GlobalConfig
import torch.nn as nn
import torch.nn.functional as F
import util

# Directory path
# os.chdir("..")
if __name__ =='__main__':
    # Setting
    config = GlobalConfig()
    isSelfRecovery = True
    skipTraining = config.skipTraining

    device = config.device
    print(device)
    # Hyper Parameters
    num_epochs = config.num_epochs
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
    TEST_PATH = config.TEST_PATH

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    def train(net, train_loader, config):
        with open('./Train_result.txt', 'w') as f:
            train_loss_localization, train_loss_cover, train_loss_recover, \
                train_loss_discriminator_enc, train_loss_discriminator_recovery = [], [], [], [], []
            hist_loss_localization, hist_loss_cover, hist_loss_recover, hist_loss_discriminator_enc, \
                hist_loss_discriminator_recovery = [], [], [], [], []
            for epoch in range(num_epochs):
                # train
                for idx, train_batch in enumerate(train_loader):
                    data, _ = train_batch
                    train_covers = data.to(device)
                    losses, output = net.train_on_batch(train_covers, train_covers)
                    x_hidden, x_recover, pred_label, cropout_label = output
                    # losses
                    train_loss_discriminator_enc.append(losses['loss_discriminator_enc'])
                    train_loss_discriminator_recovery.append(losses['loss_discriminator_recovery'])
                    train_loss_localization.append(losses['loss_localization'])
                    train_loss_cover.append(losses['loss_cover'])
                    train_loss_recover.append(losses['loss_recover'])
                    if idx % 4 == 3:
                        str = 'Net 1 Epoch {0}/{1} Training: Batch {2}/{3}. Total Loss {4:.4f}, Localization Loss {5:.4f}, ' \
                              'Cover Loss {6:.4f}, Recover Loss {7:.4f}, Adversial Cover Loss {8:.4f}, Adversial Recovery Loss {9:.4f}' \
                            .format(epoch, num_epochs, idx + 1, len(train_loader), losses['loss_sum'],
                                    losses['loss_localization'], losses['loss_cover'], losses['loss_recover'],
                                    losses['loss_discriminator_enc'], losses['loss_discriminator_recovery'])
                        f.write(str + '\n')
                        print(str)
                    if idx % 128 == 127:
                        for i in range(x_recover.shape[0]):
                            util.save_images(x_recover[i].cpu(),
                                             'epoch-{0}-recovery-batch-{1}-{2}.png'.format(epoch, idx, i),
                                             './Images/recovery',
                                             std=config.std,
                                             mean=config.mean)
                            util.save_images(x_hidden[i].cpu(),
                                             'epoch-{0}-hidden-batch-{1}-{2}.png'.format(epoch, idx, i),
                                             './Images/hidden',
                                             std=config.std,
                                             mean=config.mean)
                            util.save_images(train_covers[i].cpu(),
                                             'epoch-{0}-covers-batch-{1}-{2}.png'.format(epoch, idx, i),
                                             './Images/original',
                                             std=config.std,
                                             mean=config.mean)


                #torch.save(net.state_dict(), MODELS_PATH + 'Epoch N{}.pkl'.format(epoch + 1))


                mean_train_loss_discriminator_enc = np.mean(train_loss_discriminator_enc)
                mean_train_loss_discriminator_recovery = np.mean(train_loss_discriminator_recovery)
                mean_train_loss_localization = np.mean(train_loss_localization)
                mean_train_loss_cover = np.mean(train_loss_cover)
                mean_train_loss_recover = np.mean(train_loss_recover)
                hist_loss_discriminator_enc.append(mean_train_loss_discriminator_enc)
                hist_loss_localization.append(mean_train_loss_cover)
                hist_loss_cover.append(mean_train_loss_localization)
                hist_loss_recover.append(mean_train_loss_recover)
                net.save_state_dict(MODELS_PATH + 'Epoch N{}'.format(epoch + 1))
                # Prints epoch average loss
                print('Epoch [{0}/{1}], Average_loss: Localization Loss {2:.4f}, Cover Loss {3:.4f}, Recover Loss {4:.4f}, '
                      'Adversial Cover Loss {5:.4f}, Adversial Recovery Loss {6:.4f}'.format(
                    epoch + 1, num_epochs, mean_train_loss_localization, mean_train_loss_cover, mean_train_loss_recover,
                    mean_train_loss_discriminator_enc, mean_train_loss_discriminator_recovery
                ))

                # validate
                # for idx, test_batch in enumerate(test_loader):
                #     data, _ = test_batch
                #     test_covers = data.to(device)
                #     losses, output = net.validate_on_batch(test_covers, test_covers)

        return net, hist_loss_localization, hist_loss_cover, hist_loss_recover, hist_loss_discriminator_enc, hist_loss_discriminator_recovery

    # ------------------------------------ Begin ---------------------------------------
    # Creates net object
    net = ReversibleImageNetwork(username="qichao", config=config) #.to(device)

    # Creates training set
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            TRAIN_PATH,
            transforms.Compose([
                transforms.Scale(config.Width),
                transforms.RandomCrop(config.Width),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.mean,
                                     std=config.std),

            ])), batch_size=train_batch_size, num_workers=1,
        pin_memory=True, shuffle=True, drop_last=True)

    # Creates test set
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            TEST_PATH,
            transforms.Compose([
                transforms.Scale(config.Width),
                transforms.RandomCrop(config.Width),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.mean,
                                     std=config.std)
            ])), batch_size=test_batch_size, num_workers=1,
        pin_memory=True, shuffle=True, drop_last=True)
    if not skipTraining:
        net, hist_loss_localization, hist_loss_cover, hist_loss_recover, hist_loss_discriminator_enc, hist_loss_discriminator_recovery \
            = train(net, train_loader, config)
        #net, mean_train_loss, loss_history = train_model(net, train_loader, beta, learning_rate, isSelfRecovery)
        # Plot loss through epochs
        plt.plot(hist_loss_localization)
        plt.title('hist_loss_localization')
        plt.ylabel('Loss')
        plt.xlabel('Batch')
        plt.show()
        plt.plot(hist_loss_cover)
        plt.title('hist_loss_cover')
        plt.ylabel('Loss')
        plt.xlabel('Batch')
        plt.show()
        plt.plot(hist_loss_recover)
        plt.title('hist_loss_recover')
        plt.ylabel('Loss')
        plt.xlabel('Batch')
        plt.show()
        plt.plot(hist_loss_discriminator_enc)
        plt.title('hist_loss_discriminator_enc')
        plt.ylabel('Loss')
        plt.xlabel('Batch')
        plt.show()
        plt.plot(hist_loss_discriminator_recovery)
        plt.title('hist_loss_discriminator_recovery')
        plt.ylabel('Loss')
        plt.xlabel('Batch')
        plt.show()
    else:
        net.load_state_dict(torch.load(MODELS_PATH + 'Epoch N10'))

    # test_model(net, test_loader, beta, learning_rate, isSelfRecovery=True)

    # def train_model(net, train_loader, beta, learning_rate,isSelfRecovery=True):
    #     # batch:3 epoch:2 data:2*3*224*224
    #     with open('./Train_result.txt', 'w') as f:
    #         # Save optimizer
    #         # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    #         optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    #         # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    #
    #         loss_history = []
    #         # Iterate over batches performing forward and backward passes
    #         for epoch in range(num_epochs):
    #             if epoch % 3 == 2:
    #                 skipLocalizationNetwork, skipRecoveryNetwork = True, False
    #             else:
    #                 skipLocalizationNetwork, skipRecoveryNetwork = False, True
    #
    #             # Train mode
    #             net.train()
    #
    #             train_losses = []
    #             train_recovered, train_hidden, train_covers = None, None, None
    #             # Train one epoch
    #             for idx, train_batch in enumerate(train_loader):
    #                 data, _ = train_batch
    #
    #                 train_covers = data[:]
    #
    #                 # Creates variable from secret and cover images
    #                 # train_cover作为tamper的图像
    #                 #train_secrets = train_secrets.to(device)
    #                 train_covers = train_covers.to(device)
    #
    #
    #                 # Forward + Backward + Optimize
    #                 optimizer.zero_grad()
    #                 train_hidden, train_recovered, pred_label, cropout_label, cropout_label_2, _ = \
    #                     net(train_covers, train_covers, skipLocalizationNetwork, skipRecoveryNetwork)
    #
    #
    #                 train_loss_all, train_loss_localization, train_loss_cover, train_loss_recover = \
    #                     localization_loss(pred_label, cropout_label, cropout_label_2, train_hidden, train_covers, train_recovered,
    #                                       skipLocalizationNetwork, skipRecoveryNetwork)
    #
    #                 # Calculate loss and perform backprop
    #                 # train_loss, train_loss_cover, train_loss_secret = customized_loss(train_output, train_hidden, train_secrets,
    #                 #                                                                   train_covers, beta)
    #                 train_loss_all.backward()
    #                 optimizer.step()
    #
    #                 # Saves training loss
    #                 train_losses.append(train_loss_all.data.cpu().numpy())
    #                 loss_history.append(train_loss_all.data.cpu().numpy())
    #
    #                 if idx % 4 == 3:
    #                     str = 'Net 1 Epoch {0}/{1} Training: Batch {2}/{3}. Total Loss {4:.4f}, Localization Loss {5:.4f}, Cover Loss {6:.4f}, Recover Loss {7:.4f} '\
    #                         .format(epoch, num_epochs, idx + 1, len(train_loader), train_loss_all.data, train_loss_localization.data, train_loss_cover.data,train_loss_recover.data)
    #                     f.write(str+'\n')
    #                     print(str)
    #
    #             torch.save(net.state_dict(), MODELS_PATH + 'Epoch N{}.pkl'.format(epoch + 1))
    #             # 保存图片
    #             for i in range(train_recovered.shape[0]):
    #
    #                 util.save_images((train_recovered[i]).mul(cropout_label_2).cpu(), 'epoch-recovery-{0}-{1}.png'.format(epoch,i), './Images', std=config.std,
    #                                  mean=config.mean)
    #                 util.save_images(train_hidden[i].cpu(), 'epoch-hidden-{0}-{1}.png'.format(epoch,i), './Images', std=config.std,
    #                                  mean=config.mean)
    #                 util.save_images(train_covers[i].cpu(), 'epoch-covers-{0}-{1}.png'.format(epoch,i), './Images', std=config.std,
    #                                  mean=config.mean)
    #
    #
    #             mean_train_loss = np.mean(train_losses)
    #
    #             # Prints epoch average loss
    #             print('Epoch [{0}/{1}], Average_loss: {2:.4f}'.format(
    #                 epoch + 1, num_epochs, mean_train_loss))
    #
    #             # Debug
    #             # imshow(utils.make_grid(train_covers), 0, learning_rate=learning_rate, beta=beta)
    #             # imshow(utils.make_grid(train_hidden), 0, learning_rate=learning_rate, beta=beta)
    #     return net, mean_train_loss, loss_history


    # def test_model(net, test_loader, beta, learning_rate, isSelfRecovery=True):
    #     # Switch to evaluate mode
    #
    #     net.eval()
    #
    #     test_losses = []
    #     # Show images
    #     for idx, test_batch in enumerate(test_loader):
    #         # Saves images
    #         data, _ = test_batch
    #
    #         test_cover = data[:]
    #
    #         # Creates variable from secret and cover images
    #         test_cover = test_cover.to(device)
    #         #test_secret = torch.tensor(test_secret, requires_grad=False).to(device)
    #         test_cover = torch.tensor(test_cover, requires_grad=False).to(device)
    #
    #         test_hidden, test_recovered, pred_label, cropout_label, cropout_label_2, selected_attack = \
    #             net(test_cover, test_cover, is_test=False)
    #         # MSE标签距离 loss
    #         test_loss_all, test_loss_localization, test_loss_cover, test_loss_recover = \
    #             localization_loss(pred_label, cropout_label, cropout_label_2, test_hidden, test_cover, test_recovered)
    #
    #         #     diff_S, diff_C = np.abs(np.array(test_output.data[0]) - np.array(test_secret.data[0])), np.abs(np.array(test_hidden.data[0]) - np.array(test_cover.data[0]))
    #
    #         #     print (diff_S, diff_C)
    #
    #         test_output = test_cover*(1-cropout_label_2)+test_recovered*cropout_label_2
    #
    #         if idx < 10:
    #             print('Test: Batch {0}/{1}. Total Loss {2:.4f}, Localization Loss {3:.4f}, Cover Loss {4:.4f}, Recover Loss {5:.4f} '
    #                 .format(idx + 1, len(train_loader), test_loss_all.data, test_loss_localization.data, test_loss_cover.data,test_loss_recover.data))
    #             print('Selected: ' + selected_attack)
    #             # Creates img tensor
    #             imgs = [test_cover.data, test_hidden.data, test_output.data]
    #
    #             # prints the whole tensor
    #             torch.set_printoptions(profile="full")
    #             print('----Figure {0}----'.format(idx + 1))
    #             print('[Expected]')
    #             print(pred_label.data)
    #
    #             print('[Real]')
    #             print(cropout_label.data)
    #             print('------------------')
    #             # Prints Images
    #             util.imshow(imgs, 'Example ' + str(idx) + ', lr=' + str(learning_rate) + ', B=' + str(beta),
    #                         std=config.std, mean=config.mean)
    #
    #         test_losses.append(test_loss_all.data.cpu().numpy())
    #
    #     mean_test_loss = np.mean(test_losses)
    #
    #     print('Average loss on test set: {:.2f}'.format(mean_test_loss))

    # def customized_loss(S_prime, C_prime, S, C, B):
    #     ''' Calculates loss specified on the paper.'''
    #
    #     loss_cover = F.mse_loss(C_prime, C)
    #     loss_secret = F.mse_loss(S_prime, S)
    #     loss_all = loss_cover + B * loss_secret
    #     return loss_all, loss_cover, loss_secret
    #
    # def localization_loss(pred_label, cropout_label, cropout_label_2, train_hidden, train_covers, train_recovered,
    #                       skipLocalizationNetwork, skipRecoveryNetwork, use_vgg=False):
    #
    #     loss_localization, loss_recover = None, None
    #     hyper1, hyper2, hyper3 = config.beta[0],config.beta[1],config.beta[2]
    #     # numpy_watch_groundtruth = cropout_label.data.clone().detach().cpu().numpy()
    #     # numpy_watch_predicted = pred_label.data.clone().detach().cpu().numpy()
    #     # if use_vgg:
    #     #     vgg_loss = VGGLoss(3, 1, False).to(device)
    #     #     vgg_on_cov = vgg_loss(train_hidden)
    #     #     vgg_on_enc = vgg_loss(train_covers)
    #     #     loss_cover = F.mse_loss(vgg_on_cov, vgg_on_enc)
    #     #     if cropout_label_2 is not None:
    #     #         vgg_loss_2 = VGGLoss(3, 1, False).to(device)
    #     #         vgg_on_cov_2 = vgg_loss(train_recovered)
    #     #         vgg_on_enc_2 = vgg_loss(train_covers)
    #     #         loss_recover = F.mse_loss(vgg_on_cov_2, vgg_on_enc_2)
    #     # else:
    #     # loss_fn = nn.MSELoss()
    #     loss_cover = F.mse_loss(train_hidden, train_covers)
    #     if not skipLocalizationNetwork:
    #         if config.num_classes == 2:
    #             loss_localization = F.binary_cross_entropy(pred_label, cropout_label)
    #         else:
    #             loss_localization = criterion(pred_label, cropout_label)
    #
    #     if not skipRecoveryNetwork:
    #         # imgs = [(train_recovered).mul(cropout_label_2[1]).data,(train_covers).mul(cropout_label_2[1])]
    #         # util.imshow(imgs,'(After Net 1) Fig.1 After EncodeAndAttacked Fig.2 Original',
    #         #             std=config.std, mean=config.mean)
    #         loss_recover = F.mse_loss((train_recovered).mul(cropout_label_2), (train_covers).mul(cropout_label_2))
    #
    #     if loss_localization is not None and loss_localization < 0.15:
    #         hyper1 = 0
    #     if loss_recover is not None and loss_recover > 1000:
    #         hyper2 *= 0.7
    #
    #     # 全部训练
    #     if not skipRecoveryNetwork and not skipLocalizationNetwork:
    #         loss_all = hyper1 * loss_localization + hyper2 * loss_cover + hyper3 * loss_recover
    #     elif not skipLocalizationNetwork:
    #         loss_all = hyper1 * loss_localization + loss_cover
    #     else:
    #         loss_all = hyper2 * loss_cover + hyper3 * loss_recover
    #     return loss_all, loss_localization, loss_cover, loss_recover
