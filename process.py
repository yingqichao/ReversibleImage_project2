from DeepSteg import Net
import torch
from config import GlobalConfig
from network.reversible_image_net import ReversibleImageNetwork_ying
device = torch.device("cuda")
MODELS_PATH = './output/models/'
net = ReversibleImageNetwork_ying(username="hanson", config=GlobalConfig())
net.load_model(MODELS_PATH + 'Epoch N12')

net.save_state_dict_all(MODELS_PATH + 'Epoch N{}'.format(12))

