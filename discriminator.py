from torch import nn

class Discriminator(nn.Module):
    def __init__(self,channels_img,hidden_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, hidden_dim, kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            self.get_disc_block(hidden_dim, hidden_dim*2),
            self.get_disc_block(hidden_dim*2, hidden_dim*4),
            self.get_disc_block(hidden_dim*4, hidden_dim*8),
            self.get_disc_block(hidden_dim*8, 1, 4,2,0,final_layer=True)
            )
    
    def get_disc_block(self,input_channels,output_channels,kernel_size=4,stride=2,padding=1,final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size,stride,padding,bias=False),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2))
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size,stride,padding),
                nn.Sigmoid())
        
    def forward(self,img):
        return self.disc(img)