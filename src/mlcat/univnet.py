import torch.nn as nn
import torch
import torch.functional as F
import math

class UnivBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UnivBlock, self).__init__()

        self.skip_path = nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False)
        squeeze_factor = 0.5

        h_ch = math.ceil(in_ch*squeeze_factor)

        # TODO use weight normalization instead of batch or instance normalization
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, h_ch, 1, padding=0, bias=False),
            nn.InstanceNorm2d(h_ch),
            nn.ELU(inplace=True),
            nn.Conv2d(h_ch, h_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(h_ch),
            nn.ELU(inplace=True),
            nn.Conv2d(h_ch, out_ch, 1, padding=0, bias=False) #expand
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x += self.skip_path(residual)
        return x

class UnivDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UnivDown, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            #nn.FractionalMaxPool2d(3, output_ratio=(fraction, fraction))
            UnivBlock(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class UnivUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UnivUp, self).__init__()

        self.up = nn.Upsample(2, mode='bilinear')
        self.conv = UnivBlock(in_ch, out_ch)

    def forward(self, x, residual):
        x = self.up(x)
        diffX = x.size()[2] - residual.size()[2]
        diffY = x.size()[3] - residual.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x1,x2], dim=1)
        x = self.conv(x)
        return x

class UnivNet(nn.Module):
    # Auto generate blocks based on in and out specification
    def __init__(self, in_dim, out_dim):
        super(UnivNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        in_size = min(in_dim[1],in_dim[2])
        out_size = min(out_dim[1],out_dim[2])

        # maximum amount of downsample
        num_down = int(math.log2(in_size))
        num_up = int(math.log2(out_size))        
        
        in_ch = in_dim[1]
        self.down_layers = []
        self.up_layers = []

        for i in range(num_down):
            # UnivDown
            out_ch = in_ch # 
            self.down_layers.append(UnivDown(in_ch, out_ch))
            in_ch = out_ch
                        
        

        for i in range(num_up):
            # UnivUp
            out_ch = in_ch
            self.up_layers.append(UnivUp(in_ch, out_ch))
            in_ch = out_ch

        for idx, module in enumerate(self.down_layers):
            self.add_module("{}".format("down_",str(idx)), module)

        for idx, module in enumerate(self.up_layers):
            self.add_module("{}".format("up_",str(idx)), module)

        self.up = nn.Upsample(2, mode='bilinear')
       
    def forward(self,x):
        res_down = []
        
        import pdb; pdb.set_trace()
        
        for layer in self.down_layers:
            res_down.append(x)
            x = layer(x)
        
        for layer in self.up_layers:
            x = layer(x,res_down)

        x = self.up(x)   


if __name__ == "__main__":

    net = UnivNet(in_dim=(1,32,32), out_dim=(1,16,16))
    
    x = torch.rand((1,1,32,32))
    y = net.forward(x)


    
    # # Classifier
    # C = AutoNN(in_dim = (28, 28, 1), out_dim = (10,1), depth=4, width=10)

    # for x, t in enumerate(batches) 
    #     y = C(x)
    #     loss = cross_entropy(y, z)
    #     loss.backward()
    #     optimizer.update()





    # # VAE
    # E = AutoNN(in_dim = (128, 128, 3), out_dim = (100, 1), depth=10, width=10)
    # D = AutoNN(in_dim = (100, 1), out_dim = (128, 128, 3), depth=10, width=10)

    # for x, _ in enumerate(batches) 
    #     z = E(x)
    #     y = D(x)
    #     z = torch.randn(100)

    #     loss = mse_loss(x, y) + kl_loss(y,z_)
    #     loss.backward()
    #     optimizer.update()