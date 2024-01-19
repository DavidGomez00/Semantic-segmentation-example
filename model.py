import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    ''' This module performs a convolution followed by batch 
    normalization and relu twice.
    '''
    def __init__(self, in_channels, out_channels): 
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), # Bias = False because we use batch norm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), # Bias = False because we use batch norm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
             self.ups.append(
                 nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
             )
             self.ups.append(DoubleConv(feature*2, feature))

        # Bottleneck layer
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Final conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        # Reverse skip connections for easier use later
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    

class UNET_pl(L.LightningModule):
    def __init__(self, in_channels=3, out_channels=1, learning_rate=1e-4):
        super(UNET_pl, self).__init__()
        
        # Define model, lr and loss function
        self.model = UNET(in_channels=in_channels, out_channels=out_channels)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate

        self.train_outputs = []
        self.val_outputs = []

        # Metrics
        self.f1_score = torchmetrics.F1Score(task='binary')
        self.dice_score = torchmetrics.Dice()

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log_dict({'train_loss': loss},
                      on_step=False, on_epoch=True, prog_bar=True)
        self.train_outputs.append({'loss': loss, 'scores': scores, 'y':y})
        return {'loss': loss, 'scores': scores, 'y':y}
    
    def on_train_epoch_end(self):
        scores = torch.cat([x["scores"] for x in self.train_outputs])
        y = torch.cat([x["y"] for x in self.train_outputs])
        self.train_outputs = []

        self.log_dict({
            "train_dice_score": self.dice_score(scores, y.to(torch.int)),
            "train_f1": self.f1_score(scores, y)
        },
        on_step=False,
        on_epoch=True,
        prog_bar=True
        )

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log_dict({'val_loss': loss},
                      on_step=False, on_epoch=True, prog_bar=True)
        self.val_outputs.append({'loss': loss, 'scores': scores, 'y':y})
        return {'loss': loss, 'scores': scores, 'y':y}
    
    def on_validation_epoch_end(self):
        scores = torch.cat([x["scores"] for x in self.val_outputs])
        y = torch.cat([x["y"] for x in self.val_outputs])
        self.val_outputs = []

        self.log_dict({
            "val_dice_score": self.dice_score(scores, y.to(torch.int)),
            "val_f1": self.f1_score(scores, y)
        },
        on_step=False,
        on_epoch=True,
        prog_bar=True
        )
    
    def test_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.model.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        scores = torch.squeeze(self.model.forward(x), dim=1)
        preds = torch.argmax(scores, dim=1)
        return preds
        
    def configure_optimizers(self):
        # Filter the parameters based on `requires_grad`
        return optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)