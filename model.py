from torch import nn
from torchvision import models
from torchinfo import summary


class ResNet(nn.Module):
    """ Modified ResNet-34 for audio embedding"""
    def __init__(self, embed_dim=256, pretrained=False):
        super().__init__()
        base_model = models.resnet34(pretrained=pretrained)
        # The first conv layer of resnet50 is changed to accept 1 channel input
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        layers = list(base_model.children())[:-1]

        self.backbone = nn.Sequential(*layers)
        self.embedding = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 512),
            nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(512, embed_dim),
        )
        

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = x.view(batch_size, -1)
        x = self.embedding(x)
        return x


if __name__ == "__main__":
    model = ResNet(embed_dim=256)
    summary(model, input_size=(4, 1, 128, 501))
    print(model)
