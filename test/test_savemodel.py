import torch

from config import GlobalConfig
from network.reversible_image_net_hansonRerun256 import ReversibleImageNetwork_hanson

device = torch.device("cuda")
MODELS_PATH = '../output/models/Epoch N2'
net = ReversibleImageNetwork_hanson(username="hanson", config=GlobalConfig())
net.localizer = torch.load(MODELS_PATH + '_localizer.pth')
net.revert_network = torch.load(MODELS_PATH + '_revert_network.pth')
net.preprocessing_network = torch.load(MODELS_PATH + '_prep_network.pth')
net.discriminator_patchRecovery = torch.load(MODELS_PATH + '_discriminator_patchRecovery.pth')
net.discriminator_CoverHidden = torch.load(MODELS_PATH + '_discriminator_CoverHidden.pth')
# net.save_state_dict_all(MODELS_PATH + 'Epoch N17')
net.save_model(MODELS_PATH)



# torch.save({'state_dict': net.revert_network.state_dict()}, MODELS_PATH + '_revert_network.pth.tar')
# model = ReversibleImageNetwork_hanson(username="hanson", config=GlobalConfig())
# checkpoint = torch.load(MODELS_PATH + '_revert_network.pth.tar')
# model.revert_network.load_state_dict(checkpoint['state_dict'])
# print('Success')