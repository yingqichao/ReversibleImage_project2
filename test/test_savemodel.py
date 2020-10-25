import torch

from config import GlobalConfig
from network.reversible_image_net_hansonRerun256 import ReversibleImageNetwork_hanson

device = torch.device("cuda")
MODELS_PATH = '../output/models/Epoch N2'
net = ReversibleImageNetwork_hanson(username="hanson", config=GlobalConfig())

net.localizer = torch.load(MODELS_PATH + '_localizer.pth')
if torch.cuda.device_count() > 1:
    net.localizer = torch.nn.DataParallel(net.localizer)
net.revert_network = torch.load(MODELS_PATH + '_revert_network.pth')

if torch.cuda.device_count() > 1:
    net.revert_network = torch.nn.DataParallel(net.revert_network)
net.preprocessing_network = torch.load(MODELS_PATH + '_prep_network.pth')

if torch.cuda.device_count() > 1:
    net.preprocessing_network = torch.nn.DataParallel(net.preprocessing_network)
net.discriminator_patchRecovery = torch.load(MODELS_PATH + '_discriminator_patchRecovery.pth')

if torch.cuda.device_count() > 1:
    net.discriminator_patchRecovery = torch.nn.DataParallel(net.discriminator_patchRecovery)
net.discriminator_CoverHidden = torch.load(MODELS_PATH + '_discriminator_CoverHidden.pth')

if torch.cuda.device_count() > 1:
    net.discriminator_CoverHidden = torch.nn.DataParallel(net.discriminator_CoverHidden)
# net.save_state_dict_all(MODELS_PATH + 'Epoch N17')
net.save_model(MODELS_PATH)

net.load_model(MODELS_PATH)
