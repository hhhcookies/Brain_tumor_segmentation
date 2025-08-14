import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,in_c,out_c,dp_p):
        super().__init__()
        # self.input_c = in_c
        # self.output_c = out_c
        self.conv1 = nn.Conv2d(in_c,out_c,3,padding=1)
        # self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c,out_c,3,padding=1)
        # self.bn2 = nn.BatchNorm2d(out_c)

        self.pool = nn.MaxPool2d(2,2)
        self.activation = nn.ReLU()
        self.dp = nn.Dropout2d(p=dp_p)

    def forward(self,x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.activation(x)
        x = self.dp(x)

        return x,self.pool(x)
        
class Decoder(nn.Module):
    def __init__(self,in_c,out_c,dp_p):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_c,out_c,2,2)
        self.conv1 = nn.Conv2d(2*out_c,out_c,3,padding=1)
        # self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c,out_c,3,padding=1)
        # self.bn2 = nn.BatchNorm2d(out_c)

        self.activation = nn.ReLU()
        self.dp = nn.Dropout2d(p=dp_p)

        

    def forward(self,x,skip):
        x = self.upconv(x)
        x = torch.cat((skip,x),dim=1)
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.activation(x)
        x = self.dp(x)

        return x
    
class Unet(nn.Module):
    def __init__(self,in_channels = 2,
                 features=(32,64,128,256,512),
                 coder_dp =0.2,
                 bottleneck_dp=0.4):
        super().__init__()
        f = features
        self.bottleneck_p = bottleneck_dp

        self.encoder1 = Encoder(in_channels,f[0],coder_dp)  #2 32
        self.encoder2 = Encoder(f[0],f[1],coder_dp)  #32 64
        self.encoder3 = Encoder(f[1],f[2],coder_dp)  #64 128
        self.encoder4 = Encoder(f[2],f[3],coder_dp)  #128 256
        self.encoder5 = Encoder(f[3],f[4],coder_dp)  #256 512
        
        self.decoder1 = Decoder(f[4],f[3],coder_dp)  #512 256
        self.decoder2 = Decoder(f[3],f[2],coder_dp)  #256 128
        self.decoder3 = Decoder(f[2],f[1],coder_dp)  #128 64
        self.decoder4 = Decoder(f[1],f[0],coder_dp)  #64 32
        
        self.conv = nn.Conv2d(32,4,1)

    def forward(self,x):
        skip1,pool1 = self.encoder1(x)
        skip2,pool2 = self.encoder2(pool1)
        skip3,pool3 = self.encoder3(pool2)
        skip4,pool4 = self.encoder4(pool3)
        skip5,_ = self.encoder5(pool4)
        
        skip5 = nn.Dropout2d(p=self.bottleneck_p)(skip5)

        upconv1 = self.decoder1(skip5,skip4)
        upconv2 = self.decoder2(upconv1,skip3)
        upconv3 = self.decoder3(upconv2,skip2)
        upconv4 = self.decoder4(upconv3,skip1)
        
        logit = self.conv(upconv4)

        return logit