from torch import nn

class Generator(nn.Module):
    def __init__(self,z_dim,channels_img,hidden_dim):
        super().__init__()
        self.gen = nn.Sequential(
            self.get_gen_block(z_dim, hidden_dim*16,4,1,0),
            self.get_gen_block(hidden_dim*16, hidden_dim*8),
            self.get_gen_block(hidden_dim*8, hidden_dim*4),
            self.get_gen_block(hidden_dim*4, hidden_dim*2),
            self.get_gen_block(hidden_dim*2, channels_img, final_layer=True),            
            )
    
    def get_gen_block(self,input_channels,output_channels,kernel_size=4,stride=2,padding=1,final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size,stride,padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
                )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels,output_channels,kernel_size,stride,padding),
                nn.Tanh())
    
    def forward(self,noise):
        return self.gen(noise)