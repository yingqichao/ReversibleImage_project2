from DeepSteg import Net
import torch
from config import GlobalConfig
from network.reversible_image_net_hanson import ReversibleImageNetwork_hanson
device = torch.device("cuda")
MODELS_PATH = './output/models/'
net = ReversibleImageNetwork_hanson(username="hanson", config=GlobalConfig())
net.load_state_dict_all(MODELS_PATH + 'Epoch N5')

torch.save(net, MODELS_PATH+'ReversibleImageNet.pth')

model = torch.load(MODELS_PATH+'ReversibleImageNet.pth')
