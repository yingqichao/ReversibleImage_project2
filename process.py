from DeepSteg import Net
import torch
from config import GlobalConfig
from network.reversible_image_net import ReversibleImageNetwork_ying
from network.reversible_image_net_hanson import ReversibleImageNetwork_hanson
device = torch.device("cuda")
MODELS_PATH = './output/models/'
net = ReversibleImageNetwork_hanson(username="hanson", config=GlobalConfig())
net.load_model(MODELS_PATH + 'Epoch N17')
net.save_state_dict_all(MODELS_PATH + 'Epoch N17')
# net.save_model(MODELS_PATH + 'Epoch N{}'.format(17))

