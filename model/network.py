import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def conv3_3(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias = False)
def conv1_1(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,padding=0,bias = False)


class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=False),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate),affine=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels,affine=True)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).contiguous() 

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out
class ResBlock(nn.Module):
    def __init__(self,in_planes,out_planes,stride=1):
        super(ResBlock,self).__init__()
        self.conv1=conv3_3(in_planes,out_planes,stride)
        self.bn1 = nn.BatchNorm2d(out_planes,affine=True)
        
        self.conv2 = conv3_3(out_planes,out_planes,stride)
        self.bn2=nn.BatchNorm2d(out_planes,affine=True)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False),
            nn.BatchNorm2d(out_planes,affine=True),
            nn.ReLU()
        )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out+=self.shortcut(x)
        out = F.relu(out)

        return out
        
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BasicBlock, self).__init__()

        self.conv1=conv3_3(in_planes,out_planes)
        self.bn1 = nn.BatchNorm2d(out_planes,affine=True)
        #
        #self.bn2=nn.BatchNorm2d(out_planes)

        self.gam = GAM_Attention(out_planes,out_planes)
        

        self.atrous_block3 = nn.Conv2d(out_planes, out_planes, 3, 1, padding=3, dilation=3)
        self.abn3 = nn.BatchNorm2d(out_planes,affine=True)
       

        self.atrous_block6 = nn.Conv2d(out_planes, out_planes, 3, 1, padding=6, dilation=6)
        self.abn6 = nn.BatchNorm2d(out_planes,affine=True)
        
        self.conv2 = conv3_3(out_planes,out_planes)
        self.final_bn = nn.BatchNorm2d(out_planes,affine=True)
        

    def forward(self, x):
        out = self.bn1(F.relu(self.conv1(x)))
        #out = self.bn2(self.conv2(out))

        out1 = self.gam(out)
        out2 = self.abn3(F.relu(self.atrous_block3(out)))
        out3 = self.abn6(F.relu(self.atrous_block6(out)))

        Out = out1+out2+out3

        Out = self.final_bn(F.relu(self.conv2(Out)))
        return Out

class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=1, stride=1, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(self.in_chans, self.embed_dim, kernel_size=self.patch_size,
                              stride=self.stride, padding=self.padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super(LayerNormChannel, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x

class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, num_head = 4, hidden_size=512, drop_rate = 0.):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_head
        self.drop_rate = drop_rate
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        #position embeded
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_size))

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.out = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_dropout = nn.Dropout(self.drop_rate)
        self.proj_dropout = nn.Dropout(self.drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        #new_x_shape = x.size()[:-1] + (1,self.num_attention_heads, self.attention_head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous() 

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(hidden_states.size(0), hidden_states.size(1),-1)
        hidden_states = hidden_states.permute(0, 2, 1).contiguous() 
        hidden_states = hidden_states+self.absolute_pos_embed
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
    
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        #weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        attention_output = attention_output.permute(0, 2, 1).contiguous() 
        attention_output = attention_output.view(attention_output.size(0),attention_output.size(1),int(attention_output.size(2)**0.5),-1)
        return attention_output

class Transformer(nn.Module):
    def __init__(self,in_channel,hidden_size):
        super(Transformer, self).__init__()
        self.in_channel = in_channel
        self.hidden_size = hidden_size
        self.embeded = PatchEmbed(in_chans=self.in_channel,embed_dim=self.hidden_size)
        self.Layer_Nor1 = LayerNormChannel(num_channels=self.hidden_size)
        self.msa = Attention(hidden_size=self.hidden_size)
        self.Layer_Nor2 = LayerNormChannel(num_channels=self.hidden_size)
        self.mlp = Mlp(in_features=self.hidden_size)

    def forward(self,x):
        out = self.embeded(x)
        out1 = self.Layer_Nor1(out)
        att = self.msa(out1)
        # new_att_shape = att.size()[:-1] + (x.shape[1],1)
        # att = att.view(*new_att_shape)
        # att=att.permute(0, 2, 1, 3)
        out2 = out+att

        out3 = self.Layer_Nor2(out2)
        mpl = self.mlp(out3)
        final_out = (out2+mpl)*x
        del out,out1,att,out2,out3,mpl,x
        return final_out

class AUnet(nn.Module):
    def __init__(self,in_channel,num_classes):
        super(AUnet, self).__init__()
       

        self.encoding1 = BasicBlock(in_channel, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoding2 = BasicBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoding3 = BasicBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoding4 = BasicBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        
        self.encoding5 = BasicBlock(512, 512)

        

        self.UP1 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.decoding1 = ResBlock(512, 256)
        self.UP2 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.decoding2 = ResBlock(256, 128)
        self.UP3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.decoding3 = ResBlock(128, 64)
        self.UP4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.decoding4 = ResBlock(64, num_classes)

        #self.final_conv = conv3_3(64,16)

    def forward(self,x):
    
        en1 = self.encoding1(x)
        p1 = self.pool1(en1)

        en2 = self.encoding2(p1)
        p2 = self.pool2(en2)

        en3 = self.encoding3(p2 )
        p3 = self.pool3(en3)

        en4 = self.encoding4(p3 )
        p4 = self.pool4(en4)

        en5 = self.encoding5(p4)
       
    
        u1 = self.UP1(en5)
        de1 = self.decoding1(u1 + en4)
        u2 = self.UP2(de1)
        de2 = self.decoding2(u2 + en3)
        u3 = self.UP3(de2)
        de3 = self.decoding3(u3 + en2)
        u4 = self.UP4(de3)
        out = self.decoding4(u4 + en1)
        out = F.log_softmax(out,dim=1)
        #out = self.final_conv(de4)
        return out

class ATUnet(nn.Module):
    def __init__(self,in_channel,num_classes):
        super(ATUnet, self).__init__()
      

        self.encoding1 = BasicBlock(in_channel, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoding2 = BasicBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoding3 = BasicBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoding4 = BasicBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.encoding5 = BasicBlock(512, 512)

        self.T1 =Transformer(in_channel=512,hidden_size=512)

        self.T2 =Transformer(in_channel=1024,hidden_size=1024)
        self.T_pool2 = nn.MaxPool2d(4)
        self.T3 = Transformer(in_channel=1280, hidden_size=1280)
        self.T_pool3 = nn.MaxPool2d(8)
        self.T4 = Transformer(in_channel=1408, hidden_size=1408)

        self.UP1 = nn.ConvTranspose2d(1408, 512, 2, stride=2)
        self.decoding1 = ResBlock(512, 256)
        self.UP2 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.decoding2 = ResBlock(256, 128)
        self.UP3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.decoding3 = ResBlock(128, 64)
        self.UP4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.decoding4 = ResBlock(64, num_classes)

        #self.final_conv = conv3_3(64,16)

    def forward(self,x):
        en1 = self.encoding1(x)
        p1 = self.pool1(en1)

        en2 = self.encoding2(p1)
        p2 = self.pool2(en2)

        en3 = self.encoding3(p2 )
        p3 = self.pool3(en3)

        en4 = self.encoding4(p3 )
        p4 = self.pool4(en4)

        en5 = self.encoding5(p4)
       
        t1 = self.T1(en5)
        t2 = self.T2(torch.cat([t1,p4],dim=1))
        t3 = self.T3(torch.cat([t2,self.T_pool2(en3)],dim=1))
        t4 = self.T4(torch.cat([t3, self.T_pool3(en2)], dim=1))

        u1 = self.UP1(t4)
        de1 = self.decoding1(u1 + en4)
        u2 = self.UP2(de1)
        de2 = self.decoding2(u2 + en3)
        u3 = self.UP3(de2)
        de3 = self.decoding3(u3 + en2)
        u4 = self.UP4(de3)
        out = self.decoding4(u4 + en1)
        out = F.log_softmax(out,dim=1)
        
        return out

class ATULnet(nn.Module):
    def __init__(self,in_channel,num_classes):
        super(ATULnet, self).__init__()
       

        self.encoding1 = BasicBlock(in_channel, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoding2 = BasicBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoding3 = BasicBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoding4 = BasicBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.encoding5 = BasicBlock(512, 512)

        self.T1 =Transformer(in_channel=512,hidden_size=512)

        self.T2 =Transformer(in_channel=1024,hidden_size=1024)
        self.T_pool2 = nn.MaxPool2d(4)
        self.T3 = Transformer(in_channel=1280, hidden_size=1280)
        self.T_pool3 = nn.MaxPool2d(8)
        self.T4 = Transformer(in_channel=1408, hidden_size=1408)

        self.UP1 = nn.ConvTranspose2d(1408, 512, 2, stride=2)
        self.decoding1 = ResBlock(512, 256)
        self.UP2 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.decoding2 = ResBlock(256, 128)
        self.UP3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.decoding3 = ResBlock(128, 64)
        self.UP4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.decoding4 = ResBlock(64, num_classes)

        #self.final_conv = conv3_3(64,16)
        self.L_conv1 = conv3_3(num_classes+1,32)
        self.L_bn1 = nn.BatchNorm2d(32,affine=True)
        self.L_pool1 = nn.MaxPool2d(4)
        self.L_conv2 = conv3_3(32,64)
        self.L_bn2 = nn.BatchNorm2d(64,affine=True)
        self.L_pool2 = nn.MaxPool2d(4)
        self.L_fc1 =nn.Linear(16384,512)
        self.L_fc2 = nn.Linear(512,1)
        
    def forward(self,x,y):
        en1 = self.encoding1(x)
        p1 = self.pool1(en1)

        en2 = self.encoding2(p1)
        p2 = self.pool2(en2)

        en3 = self.encoding3(p2 )
        p3 = self.pool3(en3)

        en4 = self.encoding4(p3 )
        p4 = self.pool4(en4)

        en5 = self.encoding5(p4)
       
        t1 = self.T1(en5)
        t2 = self.T2(torch.cat([t1,p4],dim=1))
        t3 = self.T3(torch.cat([t2,self.T_pool2(en3)],dim=1))
        t4 = self.T4(torch.cat([t3, self.T_pool3(en2)], dim=1))

        u1 = self.UP1(t4)
        de1 = self.decoding1(u1 + en4)
        u2 = self.UP2(de1)
        de2 = self.decoding2(u2 + en3)
        u3 = self.UP3(de2)
        de3 = self.decoding3(u3 + en2)
        u4 = self.UP4(de3)
        out = self.decoding4(u4 + en1)
        out = F.log_softmax(out,dim=1)
        #out = self.final_conv(de4)
        if len(y)==0:
            return out
        else:
            y = torch.reshape(y,(y.shape[0],1,y.shape[1],y.shape[2]))
            L_input = torch.cat((out,y),1)
            L_out = F.relu(self.L_bn1(self.L_conv1(L_input)))
            L_out = self.L_pool1(L_out)
            L_out = F.relu(self.L_bn2(self.L_conv2(L_out)))
            L_out = self.L_pool2(L_out)
            L_out = L_out.view(L_out.size(0),-1)
            L_out = self.L_fc1(L_out)
            L_out = self.L_fc2(L_out)
            L_out = L_out.squeeze(-1)
            return out, L_out
            
