# import torch
# import torch.nn as nn
# from config import GlobalConfig
# from noise_layers.gaussian import Gaussian
# from noise_layers.jpeg_compression import JpegCompression
#
#
# def flip(x, dim):
#     indices = [slice(None)] * x.dim()
#     indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
#                                 dtype=torch.long, device=x.device)
#     return x[tuple(indices)]
#
# class EncoderDecoder(nn.Module):
#     """
#     Combines Encoder->Noiser->Decoder into single pipeline.
#     The input is the cover image and the watermark message. The module inserts the watermark into the image
#     (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
#     to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
#     a three-tuple: (encoded_image, noised_image, decoded_message)
#     """
#     def __init__(self, config=GlobalConfig()):
#         super(EncoderDecoder, self).__init__()
#         self.config = config
#         self.device = self.config.device
#
#         self.hiding_network = Hiding_naive(config=config)
#         # Noise Network
#         self.jpeg_layer = JpegCompression(self.device)
#
#         # self.cropout_layer = Cropout(config).to(self.device)
#         self.gaussian = Gaussian(config).to(self.device)
#         # self.resize_layer = Resize((0.5, 0.7)).to(self.device)
#         self.extract_layer = Extract_naive(config).to(self.device)
#
#     def forward(self, Cover, Another):
#
#         Marked = self.hiding_network(Cover, Another)
#
#
#         Marked_gaussian = self.gaussian(Marked)
#         # x_1_resize = self.resize_layer(x_1_gaussian)
#         Marked_attack = self.jpeg_layer(Marked_gaussian)
#
#         # 训练RecoverNetwork：根据部分信息恢复原始图像，这里不乘以之前的pred_label（防止网络太深）
#         out = self.extract_layer(Marked_attack)
#
#         # Test
#         # if is_test:
#         #     imgs = [x_2_attack.data, x_2_crop.data, x_2_out.data]
#         #     util.imshow(utils.make_grid(imgs), 'Fig.1 EncoderAttackedByJpeg Fig.2 Then Cropped Fig.3 Recovered', std=self.config.std,
#         #                 mean=self.config.mean)
#
#         return Marked, out
