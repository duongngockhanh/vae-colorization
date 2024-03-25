import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
  
    def __init__(self):
        super(VAE, self).__init__()
        self.hidden_size = 64

        # Encoder layers - (2, 64, 64)
        self.enc_conv1 = nn.Conv2d(2, 128, 5, stride=2, padding=2)      # (128, 32, 32)
        self.enc_bn1 = nn.BatchNorm2d(128)
        self.enc_conv2 = nn.Conv2d(128, 256, 5, stride=2, padding=2)    # (256, 16, 16)
        self.enc_bn2 = nn.BatchNorm2d(256)
        self.enc_conv3 = nn.Conv2d(256, 512, 5, stride=2, padding=2)    # (512, 8, 8)
        self.enc_bn3 = nn.BatchNorm2d(512)
        self.enc_conv4 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)   # (1024, 4, 4)
        self.enc_bn4 = nn.BatchNorm2d(1024)
        self.enc_fc1 = nn.Linear(4*4*1024, self.hidden_size*2)          # (128,)
        self.enc_dropout1 = nn.Dropout(p=0.7)

        # Conditional encoder layers - (1, 64, 64)
        self.cond_enc_conv1 = nn.Conv2d(1, 128, 5, stride=2, padding=2)     # (128, 32, 32)
        self.cond_enc_bn1 = nn.BatchNorm2d(128)
        self.cond_enc_conv2 = nn.Conv2d(128, 256, 5, stride=2, padding=2)   # (256, 16, 16)
        self.cond_enc_bn2 = nn.BatchNorm2d(256)
        self.cond_enc_conv3 = nn.Conv2d(256, 512, 5, stride=2, padding=2)   # (512, 8, 8)
        self.cond_enc_bn3 = nn.BatchNorm2d(512)
        self.cond_enc_conv4 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)  # (1024, 4, 4)
        self.cond_enc_bn4 = nn.BatchNorm2d(1024)

        # Decoder layers - (64, 1, 1)
        self.dec_upsamp1 = nn.Upsample(scale_factor=4, mode='bilinear')                 
        self.dec_conv1 = nn.Conv2d(1024+self.hidden_size, 512, 3, stride=1, padding=1)  
        self.dec_bn1 = nn.BatchNorm2d(512)
        self.dec_upsamp2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec_conv2 = nn.Conv2d(512*2, 256, 5, stride=1, padding=2)
        self.dec_bn2 = nn.BatchNorm2d(256)
        self.dec_upsamp3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec_conv3 = nn.Conv2d(256*2, 128, 5, stride=1, padding=2)
        self.dec_bn3 = nn.BatchNorm2d(128)
        self.dec_upsamp4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec_conv4 = nn.Conv2d(128*2, 64, 5, stride=1, padding=2)
        self.dec_bn4 = nn.BatchNorm2d(64)
        self.dec_upsamp5 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec_conv5 = nn.Conv2d(64, 2, 5, stride=1, padding=2)

    def encoder(self, x):                   # (2, 64, 64)
        x = F.relu(self.enc_conv1(x))
        x = self.enc_bn1(x)                 # (128, 32, 32)
        x = F.relu(self.enc_conv2(x))
        x = self.enc_bn2(x)                 # (256, 16, 16)
        x = F.relu(self.enc_conv3(x))
        x = self.enc_bn3(x)                 # (512, 8, 8)
        x = F.relu(self.enc_conv4(x))
        x = self.enc_bn4(x)                 # (1024, 4, 4)
        x = x.view(-1, 4*4*1024)
        x = self.enc_dropout1(x)
        x = self.enc_fc1(x)                 # (128,)
        mu = x[..., :self.hidden_size]      # (64,)
        logvar = x[..., self.hidden_size:]  # (64,)
        return mu, logvar

    def cond_encoder(self, x):                      # (1, 64, 64)
        x = F.relu(self.cond_enc_conv1(x))
        sc_feat32 = self.cond_enc_bn1(x)            # (128, 32, 32)
        x = F.relu(self.cond_enc_conv2(sc_feat32))
        sc_feat16 = self.cond_enc_bn2(x)            # (256, 16, 16)
        x = F.relu(self.cond_enc_conv3(sc_feat16))
        sc_feat8 = self.cond_enc_bn3(x)             # (512, 8, 8)
        x = F.relu(self.cond_enc_conv4(sc_feat8))
        sc_feat4 = self.cond_enc_bn4(x)             # (1024, 4, 4)
        return sc_feat32, sc_feat16, sc_feat8, sc_feat4

    def decoder(self, z, sc_feat32, sc_feat16, sc_feat8, sc_feat4):
        x = z.view(-1, self.hidden_size, 1, 1)      # (64, 1, 1)
        x = self.dec_upsamp1(x)                     # (64, 4, 4)
        x = torch.cat([x, sc_feat4], 1)             # (64+1024, 4, 4)
        x = F.relu(self.dec_conv1(x))               # (512, 4, 4)
        x = self.dec_bn1(x)                         # (512, 4, 4)
        x = self.dec_upsamp2(x)                     # (512, 8, 8)
        x = torch.cat([x, sc_feat8], 1)             # (512+512, 8, 8)
        x = F.relu(self.dec_conv2(x))               # (256, 8, 8)
        x = self.dec_bn2(x)                         # (256, 8, 8)
        x = self.dec_upsamp3(x)                     # (256, 16, 16)
        x = torch.cat([x, sc_feat16], 1)            # (256+256, 16, 16)
        x = F.relu(self.dec_conv3(x))               # (128, 16, 16)
        x = self.dec_bn3(x)                         # (128, 16, 16)
        x = self.dec_upsamp4(x)                     # (128, 32, 32)
        x = torch.cat([x, sc_feat32], 1)            # (128+128, 32, 32)
        x = F.relu(self.dec_conv4(x))               # (64, 32, 32)
        x = self.dec_bn4(x)                         # (64, 32, 32)
        x = self.dec_upsamp5(x)                     # (64, 64, 64)
        x = torch.tanh(self.dec_conv5(x))           # (2, 64, 64)
        return x
      
    def forward(self, color, greylevel, z_in=None):
        sc_feat32, sc_feat16, sc_feat8, sc_feat4 = self.cond_encoder(greylevel)
        mu, logvar = self.encoder(color)
        if self.training:
            stddev = torch.sqrt(torch.exp(logvar))
            eps = torch.randn_like(stddev)
            z = mu + eps * stddev
        else:
            z = z_in
        color_out = self.decoder(z, sc_feat32, sc_feat16, sc_feat8, sc_feat4)
        return mu, logvar, color_out


if __name__ == "__main__":
    model = VAE()
    from torchsummary import summary
    summary(model, [(2, 64, 64), (1, 64, 64), (64,)], device="cpu")
