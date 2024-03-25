import torch
import torch.nn as nn
import torch.nn.functional as F

class MDN(nn.Module):
  
    def __init__(self):
        super(MDN, self).__init__()

        self.feats_nch = 512
        self.hidden_size = 64
        self.nmix = 8
        self.nout = (self.hidden_size + 1) * self.nmix

        # Define MDN Layers - (512, 64, 64)
        self.model = nn.Sequential(
            nn.Conv2d(self.feats_nch, 384, 5, stride=1, padding=2), # (384, 32, 32)
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 320, 5, stride=1, padding=2),            # (320, 32, 32)
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.Conv2d(320, 288, 5, stride=1, padding=2),            # (288, 32, 32)
            nn.BatchNorm2d(288),
            nn.ReLU(),
            nn.Conv2d(288, 256, 5, stride=2, padding=2),            # (256, 16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 5, stride=1, padding=2),            # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 96, 5, stride=2, padding=2),             # (96, 8, 8)
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 64, 5, stride=2, padding=2),              # (64, 4, 4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.7)
        )

        self.fc = nn.Linear(4 * 4 * 64, self.nout)

    def forward(self, feats):
        x = self.model(feats)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(x)
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = MDN()
    from torchsummary import summary
    summary(model, (512, 32, 32))   # (520,)
