import torch
from torch import nn
from common.resnet import resnet34
from common.normalize import Normalize

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
        
        # 特征融合和偏好预测
        self.preference_head = nn.Sequential(
            # 输入: 512 (ResNet) + 128 (速度) = 640 通道
            nn.Conv2d(640, 256, 1),
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
        
    def forward(self, rgb, spd):
        """
        Args:
            rgb: RGB图像 [B, 3, H, W]
            spd: 速度 [B]
        Returns:
            preference: 预测的偏好分数 [B]
        """
        # 图像特征提取
        rgb = self.normalize(rgb/255.0)
        visual_features = self.backbone(rgb)  # [B, 512, H/32, W/32]
        
        # 速度编码
        spd_features = self.spd_encoder(spd[:,None])  # [B, 128]
        spd_features = spd_features[...,None,None].expand(-1,-1,
                                                         visual_features.shape[2],
                                                         visual_features.shape[3])
        
        # 特征融合
        features = torch.cat([visual_features, spd_features], dim=1)
        
        # 预测偏好分数
        preference = self.preference_head(features)
        return preference.squeeze(1)
    
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
        for batch in train_loader:
            rgb = batch['rgb'].to(device)
            spd = batch['speed'].to(device)
            preference = batch['preference'].float().to(device)
            
            pred = model(rgb, spd)
            loss = model.compute_loss(pred, preference)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(device)
                spd = batch['speed'].to(device)
                preference = batch['preference'].float().to(device)
                
                pred = model(rgb, spd)
                loss = model.compute_loss(pred, preference)
                val_loss += loss.item()
                
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}') 