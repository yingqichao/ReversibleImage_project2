from DeepSteg import Net
import torch
device = torch.device("cuda")
MODELS_PATH = './output/models/'
net = Net().to(device)
net.load_state_dict(torch.load(MODELS_PATH + '_pretrain_Epoch N20.pkl'))

torch.save(net.m2.state_dict(), MODELS_PATH + '_pretrain_hiding_Epoch N20.pkl')
torch.save(net.m3.state_dict(), MODELS_PATH + '_pretrain_reveal_Epoch N20.pkl')