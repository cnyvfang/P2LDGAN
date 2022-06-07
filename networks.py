from models.networks_basic import *
import torchvision.models as models

##############################
#         P2LDGAN_G
##############################

class Generator(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.resnext = models.resnext50_32x4d(pretrained=True)

        self.up1 = cross_scale_up(2048, 1024, dropout=0.5)
        self.up2 = cross_scale_up(1024, 512)
        self.up3 = cross_scale_up(512, 256)
        self.up4 = cross_scale_up(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        output1 = self.resnext.conv1(x)
        output1 = self.resnext.bn1(output1)
        output1 = self.resnext.relu(output1)
        #without maxpool
        output2 = self.resnext.layer1(output1)
        output3 = self.resnext.layer2(output2)
        output4 = self.resnext.layer3(output3)
        output5 = self.resnext.layer4(output4)

        u1 = self.up1(4, output5, output4, output3, output2, output1)
        u2 = self.up2(3, u1, output3, output2, output1)
        u3 = self.up3(2, u2, output2, output1)
        u4 = self.up4(u3, output1)
        f = self.final(u4)

        return f


##############################
#        P2LDGAN_D
##############################

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        channels = channels

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        x = self.model(img)
        return x

