import torch

from config import GlobalConfig
from network.reversible_image_net_hansonRerun256 import ReversibleImageNetwork_hanson

device = torch.device("cuda")
MODELS_PATH = '../output/models/Epoch N2'
net = ReversibleImageNetwork_hanson(username="hanson", config=GlobalConfig())
net.load_model_old(MODELS_PATH)
# net.save_state_dict_all(MODELS_PATH + 'Epoch N17')
net.save_state_dict_all(MODELS_PATH)



# torch.save({'state_dict': net.revert_network.state_dict()}, MODELS_PATH + '_revert_network.pth.tar')
# model = ReversibleImageNetwork_hanson(username="hanson", config=GlobalConfig())
# checkpoint = torch.load(MODELS_PATH + '_revert_network.pth.tar')
# model.revert_network.load_state_dict(checkpoint['state_dict'])
# print('Success')