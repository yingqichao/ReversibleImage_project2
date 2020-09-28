from DeepSteg import Net
import torch
from config import GlobalConfig
from network.reversible_image_net_hanson import ReversibleImageNetwork_hanson
device = torch.device("cuda")
MODELS_PATH = './output/models/'
net = ReversibleImageNetwork_hanson(username="hanson", config=GlobalConfig())
net.load_state_dict_all(MODELS_PATH + 'Epoch N5')

torch.save(net.preprocessing_network, './preprocessing.pth')
torch.save(net.hiding_network, './hiding_network.pth')
torch.save(net.revert_network, './revert_network.pth')
torch.save(net.reveal_network, './reveal_network.pth')
torch.save(net.discriminator, './discriminator.pth')

