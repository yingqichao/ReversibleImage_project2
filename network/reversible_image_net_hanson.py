# %matplotlib inline
import torch
import torch.nn as nn
from encoder.encoder_decoder import EncoderDecoder
from config import GlobalConfig
from localizer.localizer import LocalizeNetwork
from localizer.localizer_noPool import LocalizeNetwork_noPool
from decoder.extract_naive import Extract_naive
from encoder.hiding_naive import Hiding_naive
from noise_layers.cropout import Cropout
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.resize import Resize
from noise_layers.gaussian import Gaussian
from network.discriminator import Discriminator
from loss.vgg_loss import VGGLoss
from encoder.prep_unet import PrepNetwork_Unet
from decoder.revert_unet import Revert_Unet
from encoder.prep_naive import PrepNetwork_Naive

class ReversibleImageNetwork_hanson:
    def __init__(self, username, config=GlobalConfig()):
        super(ReversibleImageNetwork_hanson, self).__init__()
        self.config = config
        self.device = self.config.device
        self.username = username
        """ Generator Network"""
        self.encoder_decoder = EncoderDecoder(config=config).to(self.device)
        self.preprocessing_network = PrepNetwork_Unet(config=config)
        """ Recovery Network """
        self.revert_network = Revert_Unet(config=config)
        """Localize Network"""
        # if self.username=="qichao":
        #     self.localizer = LocalizeNetwork(config).to(self.device)
        # else:
        #     self.localizer = LocalizeNetwork_noPool(config).to(self.device)
        """Discriminator"""
        self.discriminator = Discriminator(config).to(self.device)
        self.cover_label = 1
        self.encoded_label = 0
        """Vgg"""

        self.vgg_loss = VGGLoss(3, 1, False)
        self.vgg_loss.to(self.device)

        """Loss"""
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.mse_loss = nn.MSELoss().to(self.device)

        """Optimizer"""
        self.optimizer_encoder_decoder_network = torch.optim.Adam(self.encoder_decoder.parameters())
        self.optimizer_preprocessing_network = torch.optim.Adam(self.preprocessing_network.parameters())
        self.optimizer_revert_network = torch.optim.Adam(self.revert_network.parameters())
        # self.optimizer_localizer = torch.optim.Adam(self.localizer.parameters())
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

        """Attack Layers"""
        self.cropout_layer = Cropout(config).to(self.device)
        self.jpeg_layer = JpegCompression(self.device).to(self.device)
        self.resize_layer = Resize((0.5, 0.7)).to(self.device)
        self.gaussian = Gaussian(config).to(self.device)

    def pretrain_on_batch(self, Cover, Another):
        """
            预训练：训练Hiding Images in Images论文的结果，也即把Secret图像隐藏到Cover图像中
            其中HidingNetwork和ExtractNetwork的训练主要是为了下面的train_on_batch
        """
        batch_size = Cover.shape[0]
        self.encoder_decoder.train()

        with torch.enable_grad():
            """ Run, Train the discriminator"""
            # self.optimizer_localizer.zero_grad()
            self.optimizer_discrim.zero_grad()
            Marked, Extracted = self.encoder_decoder(Cover, Another)
            """ Discriminate """
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_on_cover = self.discriminator(Cover)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            d_loss_on_cover.backward()
            d_on_encoded = self.discriminator(Marked.detach())
            d_on_recovered = self.discriminator(Extracted.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)
            d_loss_on_recovered = self.bce_with_logits_loss(d_on_recovered, d_target_label_encoded)
            d_loss_on_fake_total = (d_loss_on_encoded + d_loss_on_recovered) / 2
            d_loss_on_fake_total.backward()
            self.optimizer_discrim.step()

            """ Train PrepNetwork and RevertNetwork """
            if self.config.useVgg == False:
                loss_cover = self.mse_loss(Marked, Cover)
                loss_recover = self.mse_loss(Extracted, Another)
            else:
                vgg_on_cov = self.vgg_loss(Cover)
                vgg_on_another = self.vgg_loss(Another)
                vgg_on_enc = self.vgg_loss(Marked)
                loss_cover = self.mse_loss(vgg_on_cov, vgg_on_enc)
                vgg_on_recovery = self.vgg_loss(Extracted)
                loss_recover = self.mse_loss(vgg_on_another, vgg_on_recovery)
            d_on_encoded_for_enc = self.discriminator(Marked)
            g_loss_adv_enc = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)
            d_on_encoded_for_recovery = self.discriminator(Extracted)
            g_loss_adv_recovery = self.bce_with_logits_loss(d_on_encoded_for_recovery, g_target_label_encoded)
            """ Total loss for EncoderDecoder """
            loss_enc_dec = g_loss_adv_recovery * self.config.hyper_discriminator + loss_recover * self.config.hyper_recovery \
                           + loss_cover * self.config.hyper_cover + g_loss_adv_enc * self.config.hyper_discriminator
            # + loss_cover * self.config.hyper_cover\
            # + loss_localization_again * self.config.hyper_localizer\
            # + g_loss_adv_enc * self.config.hyper_discriminator \
            loss_enc_dec.backward()
            self.optimizer_encoder_decoder_network.step()

        losses = {
            'loss_sum': loss_enc_dec.item(),
            'loss_localization': 0,  # loss_localization.item(),
            'loss_cover': loss_cover.item(),
            'loss_recover': loss_recover.item(),
            'loss_discriminator_enc': g_loss_adv_enc.item(),
            'loss_discriminator_recovery': g_loss_adv_recovery.item()
        }
        return losses, (Marked, Extracted)


    def train_on_batch(self, Cover):
        """
            训练方法：先额外训练单个Secret图像向Cover图像嵌入和提取的网络（可以通过读取预训练结果），
            然后将Secret图像送入PrepNetwork(基于Unet)做处理（置乱），送进嵌入和提取网络，
            提取得到的图像再送进RevertNetwork得到近似原图（B），再填充到原图中
            Loss：B与原图的loss，Hidden与原图的loss
        """
        batch_size = Cover.shape[0]
        self.preprocessing_network.train()
        self.encoder_decoder.train()
        self.revert_network.train()

        with torch.enable_grad():
            """ Run, Train the discriminator"""
            #self.optimizer_localizer.zero_grad()
            self.optimizer_discrim.zero_grad()
            Secret_processed = self.preprocessing_network(Cover)
            Marked, Extracted = self.encoder_decoder(Cover, Secret_processed)

            Recovered = self.revert_network(Extracted)
            """ Discriminate """
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_on_cover = self.discriminator(Cover)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            d_loss_on_cover.backward()
            d_on_encoded = self.discriminator(Marked.detach())
            d_on_recovered = self.discriminator(Recovered.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)
            d_loss_on_recovered = self.bce_with_logits_loss(d_on_recovered, d_target_label_encoded)
            d_loss_on_fake_total = (d_loss_on_encoded + d_loss_on_recovered) / 2
            d_loss_on_fake_total.backward()
            self.optimizer_discrim.step()

            # x_1_crop, cropout_label, _ = self.cropout_layer(x_hidden, Cover)
            # x_1_gaussian = self.gaussian(x_1_crop)
            # x_1_resize = self.resize_layer(x_1_gaussian)
            # x_1_attack = self.jpeg_layer(x_1_crop)
            # pred_label = self.localizer(x_1_attack.detach())
            # loss_localization = self.bce_with_logits_loss(pred_label, cropout_label)
            # loss_localization.backward()
            # self.optimizer_localizer.step()
            """ Train PrepNetwork and RevertNetwork """
            self.optimizer_preprocessing_network.zero_grad()
            self.optimizer_revert_network.zero_grad()
            # pred_again_label = self.localizer(x_1_attack)
            # loss_localization_again = self.bce_with_logits_loss(pred_again_label, cropout_label)
            if self.config.useVgg == False:
                loss_cover = self.mse_loss(Marked, Cover)
                loss_recover = self.mse_loss(Recovered, Cover)
            else:
                vgg_on_cov = self.vgg_loss(Cover)
                vgg_on_enc = self.vgg_loss(Marked)
                loss_cover = self.mse_loss(vgg_on_cov, vgg_on_enc)
                vgg_on_recovery = self.vgg_loss(Recovered)
                loss_recover = self.mse_loss(vgg_on_cov, vgg_on_recovery)
            d_on_encoded_for_enc = self.discriminator(Marked)
            g_loss_adv_enc = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)
            d_on_encoded_for_recovery = self.discriminator(Recovered)
            g_loss_adv_recovery = self.bce_with_logits_loss(d_on_encoded_for_recovery, g_target_label_encoded)
            """ Total loss for EncoderDecoder """
            loss_enc_dec =  g_loss_adv_recovery * self.config.hyper_discriminator + loss_recover * self.config.hyper_recovery \
                            + loss_cover * self.config.hyper_cover + g_loss_adv_enc * self.config.hyper_discriminator
                            # + loss_cover * self.config.hyper_cover\
                           # + loss_localization_again * self.config.hyper_localizer\
                            # + g_loss_adv_enc * self.config.hyper_discriminator \
            loss_enc_dec.backward()
            self.optimizer_revert_network.step()
            self.optimizer_preprocessing_network.step()

        losses = {
            'loss_sum': loss_enc_dec.item(),
            'loss_localization': 0, #loss_localization.item(),
            'loss_cover': loss_cover.item(),
            'loss_recover': loss_recover.item(),
            'loss_discriminator_enc': g_loss_adv_enc.item(),
            'loss_discriminator_recovery': g_loss_adv_recovery.item()
        }
        return losses, (Marked, Recovered)

    def validate_on_batch(self, Cover, Another):
        pass
        # batch_size = Cover.shape[0]
        # self.encoder_decoder.eval()
        # self.localizer.eval()
        # with torch.enable_grad():
        #     x_hidden, x_recover, mask, self.jpeg_layer.__class__.__name__ = self.encoder_decoder(Cover, Another)
        #
        #     x_1_crop, cropout_label, _ = self.cropout_layer(x_hidden, Cover)
        #     x_1_gaussian = self.gaussian(x_1_crop)
        #     x_1_attack = self.jpeg_layer(x_1_gaussian)
        #     pred_label = self.localizer(x_1_attack.detach())
        #     loss_localization = self.bce_with_logits_loss(pred_label, cropout_label)
        #
        #     loss_cover = self.mse_loss(x_hidden, Cover)
        #     loss_recover = self.mse_loss(x_recover.mul(mask), Cover.mul(mask)) / self.config.min_required_block_portion
        #     loss_enc_dec = loss_localization * self.hyper[0] + loss_cover * self.hyper[1] + loss_recover * \
        #                    self.hyper[2]
        #
        # losses = {
        #     'loss_sum': loss_enc_dec.item(),
        #     'loss_localization': loss_localization.item(),
        #     'loss_cover': loss_cover.item(),
        #     'loss_recover': loss_recover.item()
        # }
        # return losses, (x_hidden, x_recover.mul(mask) + Cover.mul(1 - mask), pred_label, cropout_label)

    def save_state_dict_PrepRevert(self, path):
        torch.save(self.revert_network.state_dict(), path + '_revert_network.pkl')
        torch.save(self.preprocessing_network.state_dict(), path + '_prep_network.pkl')

    def save_state_dict_EncDec(self, path):
        torch.save(self.encoder_decoder.state_dict(), path + '_encoder_decoder_network.pkl')

    def save_state_dict_Discriminator(self, path):
        torch.save(self.discriminator.state_dict(), path + '_discriminator_network.pkl')

    def load_state_dict_Discriminator(self,path):
        self.discriminator.load_state_dict(torch.load(path + '_discriminator_network.pkl'))

    def load_state_dict_PrepRevert(self,path):
        self.preprocessing_network.load_state_dict(torch.load(path + '_prep_network.pkl'))
        self.revert_network.load_state_dict(torch.load(path + '_revert_network.pkl'))

    def load_state_dict_EncDec(self,path):
        self.encoder_decoder.load_state_dict(torch.load(path + '_encoder_decoder_network.pkl'))

    # def forward(self, Cover, Another, skipLocalizationNetwork, skipRecoveryNetwork, is_test=False):
    #     # 得到Encode后的特征平面
    #     x_1_out = self.encoder(Cover)
    #
    #     # 训练第一个网络：Localize
    #     pred_label, cropout_label = None, None
    #     # 添加Cropout噪声，cover是跟secret无关的图
    #     if not self.skipLocalizationNetwork:
    #         x_1_crop, cropout_label, _ = self.cropout_layer_1(x_1_out,Another)
    #
    #
    #         # 添加一般噪声：Gaussian JPEG 等（optional）
    #         if self.add_other_noise:
    #             #layer_num = np.random.choice(2)
    #             random_noise_layer = np.random.choice(self.other_noise_layers,0)[0]
    #             x_1_attack = random_noise_layer(x_1_crop)
    #         else:
    #             # 固定加JPEG攻击（1），或者原图（0）
    #             random_noise_layer = self.other_noise_layers[1]
    #             x_1_attack = random_noise_layer(x_1_crop)
    #
    #         # Test
    #         # if is_test:
    #         #     imgs = [x_1_attack.data, Cover.data]
    #         #     util.imshow(imgs, '(After Net 1) Fig.1 After EncodeAndAttacked Fig.2 Original', std=self.config.std, mean=self.config.mean)
    #
    #         #如果不添加其他攻击，就是x_1_crop，否则是x_1_crop_attacked
    #         pred_label = self.localize(x_1_attack)
    #
    #     x_2_out, cropout_label_2, mask = None, None, None
    #     # 训练第二个网络：根据部分信息恢复原始图像，这里不乘以之前的pred_label（防止网络太深）
    #     if not self.skipRecoveryNetwork:
    #         if self.add_other_noise:
    #             #layer_num = np.random.choice(2)
    #             random_noise_layer_again = np.random.choice(self.other_noise_layers_again,0)[0]
    #             x_2_attack = random_noise_layer_again(x_1_out)
    #         else:
    #             # 固定加JPEG攻击（1），或者原图（0）
    #             random_noise_layer_again = self.other_noise_layers_again[1]
    #             x_2_attack = random_noise_layer_again(x_1_out)
    #
    #         x_2_crop, cropout_label_2, mask = self.cropout_layer_2(x_2_attack)
    #         # 经过类U-net得到恢复图像
    #         x_2_out = self.recovery(x_2_crop)
    #         # Test
    #         # if is_test:
    #         #     imgs = [x_2_attack.data, x_2_crop.data, x_2_out.data]
    #         #     util.imshow(utils.make_grid(imgs), 'Fig.1 EncoderAttackedByJpeg Fig.2 Then Cropped Fig.3 Recovered', std=self.config.std,
    #         #                 mean=self.config.mean)
    #
    #     return x_1_out, x_2_out, pred_label, cropout_label, mask, self.jpeg_layer.__class__.__name__
