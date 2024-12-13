import numpy as np
import dlsia
from dlsia.core.networks import sms1d
from dlsia.core.networks import smsnet
import torch
from torch import nn
import einops
from collections import OrderedDict

class specseg(nn.Module):
    def __init__(self,
                 spectral_projection_head,
                 spatial_net,

                 ):
        super(specseg, self).__init__()
        self.spectral_projection_head = spectral_projection_head
        self.spatial_net = spatial_net

    def forward(self,x):
        x = self.spectral_projection_head(x)
        x = self.spatial_net(x)
        return x


    def save_network_parameters(self, name=None):
        """
        Save the network parameters
        :param name: The filename
        :type name: str
        :return: None
        :rtype: None
        """
        network_dict = OrderedDict()
        network_dict["topo_dict_spectral"] = self.spectral_projection_head.topology_dict()
        network_dict["state_dict_spectral"] = self.spectral_projection_head.state_dict()
        network_dict["topo_dict_sms"] = self.spatial_net.topology_dict()
        network_dict["state_dict_sms"] = self.spatial_net.state_dict()
        if name is None:
            return network_dict
        torch.save(network_dict, name)


class FCNetwork(nn.Module):
    def __init__(self, Cin, Cmiddle, Cout, dropout_rate=0.5):
        super(FCNetwork, self).__init__()

        self.Cin = Cin
        self.Cmiddle = Cmiddle
        self.Cout = Cout
        self.dropout_rate = dropout_rate

        layers = []
        layers.append(nn.Linear(self.Cin, self.Cmiddle[0]))
        layers.append(nn.BatchNorm1d(self.Cmiddle[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))

        for i in range(len(self.Cmiddle)-1):
            layers.append(nn.Linear(self.Cmiddle[i], self.Cmiddle[i+1]))
            layers.append(nn.BatchNorm1d(self.Cmiddle[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))

        layers.append(nn.Linear(self.Cmiddle[-1], self.Cout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        N,C,Y,X = x.shape
        x = einops.rearrange(x, "N C Y X -> (N Y X) C")
        x = self.network(x)
        x = einops.rearrange(x, "(N Y X) C -> N C Y X", N=N, Y=Y, X=X)
        return x

    def topology_dict(self):
        topo = OrderedDict()
        topo["Cin"] = self.Cin
        topo["Cmiddle"] = self.Cmiddle
        topo["Cout"] = self.Cout
        topo["dropout_rate"] = self.dropout_rate
        return topo

    def save_network_parameters(self, name=None):
        network_dict = OrderedDict()
        network_dict["topo_dict"] = self.topology_dict()
        network_dict["state_dict"] = self.state_dict()
        if name is None:
            return network_dict
        torch.save(network_dict, name)


class FCNetwork1D(nn.Module):
    def __init__(self, Cin, Cmiddle, Cout, dropout_rate=0.5, final=None):
        super(FCNetwork1D, self).__init__()

        self.Cin = Cin
        self.Cmiddle = Cmiddle
        self.Cout = Cout
        self.dropout_rate = dropout_rate
        self.final = final

        layers = []
        layers.append(nn.Linear(self.Cin, self.Cmiddle[0]))
        layers.append(nn.BatchNorm1d(self.Cmiddle[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))

        for i in range(len(self.Cmiddle)-1):
            layers.append(nn.Linear(self.Cmiddle[i], self.Cmiddle[i+1]))
            layers.append(nn.BatchNorm1d(self.Cmiddle[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))

        layers.append(nn.Linear(self.Cmiddle[-1], self.Cout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        if self.final is not None:
            x = self.final(x)
        return x

    def topology_dict(self):
        topo = OrderedDict()
        topo["Cin"] = self.Cin
        topo["Cmiddle"] = self.Cmiddle
        topo["Cout"] = self.Cout
        topo["dropout_rate"] = self.dropout_rate
        topo["final"] = self.final
        return topo

    def save_network_parameters(self, name=None):
        network_dict = OrderedDict()
        network_dict["topo_dict"] = self.topology_dict()
        network_dict["state_dict"] = self.state_dict()
        if name is None:
            return network_dict
        torch.save(network_dict, name)


def specseg_from_file(params):
    if isinstance(params, str):
        params = torch.load(params, map_location="cpu")
    spectral_projection_head = FCNetwork(**params["topo_dict_spectral"])
    spectral_projection_head.load_state_dict(params["state_dict_spectral"])
    SMSObj = smsnet.SMSNet(**params["topo_dict_sms"])
    SMSObj.load_state_dict(params["state_dict_sms"])
    this_spec_seg = specseg(spectral_projection_head, SMSObj)
    return this_spec_seg


def build_random_network(Cin, Cmiddle, Clatent, Cout, Y, X, depth, alpha, gamma, P_connect=0.5, dropout=0.25):

    spectral_encoder = FCNetwork(Cin,Cmiddle,Clatent,dropout)
    LPS2 = {"LL_alpha": alpha,
            "LL_gamma":gamma,
            "LL_max_degree": depth,
            "LL_min_degree": 1,
            "IL": P_connect,
            "LO": P_connect,
            "IO": True}

    sizing_settings2 = {'min_power':0,
                      'max_power':0,
                      'stride_base':2}


    spatial_net = smsnet.random_SMS_network(Clatent,
                       Cout,
                       depth,
                       dilation_choices=[1,2,3,4,5],
                       in_shape=(Y, X),
                       out_shape=(Y, X),
                       hidden_out_channels=[Cout],
                       layer_probabilities=LPS2,
                       sizing_settings=sizing_settings2,
                       dilation_mode="Edges",
                       network_type="Classification")

    obj = specseg(spectral_encoder, spatial_net)
    return obj










