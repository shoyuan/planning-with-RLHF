import torch
from torch import nn
from tqdm import tqdm
from tqdm import trange
import common
from common.resnet import resnet34
from common.normalize import Normalize
from torch.utils.data import DataLoader

from .preference_dataset import PreferenceDataset, create_dataloader



class PathEncoder(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(2, embedding_dim),  # 假设每个路径点有2个坐标
            nn.ReLU(True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(True)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, path_points):
        # path_points: shape [batch_size, 10, 2]
        embedded = self.embedding(path_points)  # [batch_size, 10, embedding_dim]
        embedded = embedded.permute(1, 0, 2)    # [10, batch_size, embedding_dim] 适合Transformer
        encoded = self.transformer_encoder(embedded)  # [10, batch_size, embedding_dim]
        encoded = encoded.permute(1, 2, 0)       # [batch_size, embedding_dim, 10]
        pooled = self.pooling(encoded).squeeze(-1)  # [batch_size, embedding_dim]
        return pooled

class RewardModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        

        # 图像特征提取器 (与LBC相同)
        self.backbone = resnet34(pretrained=pretrained, num_channels=3)
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        
        # 速度编码器 (与LBC相同)
        self.spd_encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
        )

        self.path_encoder = PathEncoder(embedding_dim=128)
        
        # 特征融合和偏好预测
        self.preference_head = nn.Sequential(
            # 输入: 512 (ResNet) + 128 (速度) = 640 通道
            nn.Conv2d(768, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 空间池化
            nn.AdaptiveAvgPool2d(1),
            
            # MLP预测偏好分数
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),  # 输出单个偏好分数 (1-5)
        )
        
    def forward(self, images, speeds, path_points):
        # 图像特征
        images = self.normalize(images)
        image_features = self.backbone(images)  # [batch_size, 512, H, W]
        #print(image_features.shape)
        
        # 速度特征
        speed_features = self.spd_encoder(speeds)  # [batch_size, 128]
        speed_features = speed_features.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 128, 1, 1]
        #print(speed_features.shape)
        speed_features = speed_features.expand(-1, -1, image_features.size(2), image_features.size(3))
        
        # 路径点特征
        path_features = self.path_encoder(path_points)  # [batch_size, 128]
        path_features = path_features.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 128, 1, 1]
        path_features = path_features.expand(-1, -1, image_features.size(2), image_features.size(3))
        
        # 特征拼接
        concatenated = torch.cat([image_features, speed_features, path_features], dim=1)  # [batch_size, 768, H, W]
        
        # 偏好预测
        preference = self.preference_head(concatenated)  # [batch_size, 1]
        
        return preference
    
    def compute_loss(self, pred_preference, target_preference):
        """
        计算MSE损失
        """
        return nn.MSELoss()(pred_preference, target_preference)

def train_reward_model(model, train_loader, val_loader, 
                      num_epochs=10, lr=1e-4, device='cuda'):
    """
    训练奖励模型
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', total=len(train_loader))
        for batch in pbar:
            rgb = batch['rgb'].to(device)
            spd = batch['speed'].to(device)
            path_points = batch['path_points'].to(device)
            preference = batch['preference'].float().to(device)
            
            pred = model(rgb, spd, path_points)
            loss = model.compute_loss(pred, preference)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(device)
                spd = batch['speed'].to(device)
                path_points = batch['path_points'].to(device)
                preference = batch['preference'].float().to(device)
                
                pred = model(rgb, spd, path_points)
                loss = model.compute_loss(pred, preference)
                val_loss += loss.item()
        torch.save(model.state_dict(), f'./expirements/models/reward_model/reward_model_{epoch}.pth')

    
        # pbar = tqdm(total=num_epochs, desc=f'Epoch{epoch}', unit='epoch')
        # pbar.update(epoch + 1)
        # pbar.set_postfix({'Train Loss': f'{train_loss/len(train_loader):.4f}', 'Val Loss': f'{val_loss/len(val_loader):.4f}'})
        # print(f'Epoch {epoch+1}/{num_epochs}')
        # print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        # print(f'Val Loss: {val_loss/len(val_loader):.4f}') 

        # 保存模型
       

if __name__ == '__main__':

    dataset = './expirements/data/collected_data/train_test/'#prefer_data1245_yevyc
    train_loader, val_loader = create_dataloader(dataset)

    model = RewardModel()

    train_reward_model(model, train_loader, val_loader)