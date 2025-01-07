import os
import lmdb
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class PreferenceDataset(Dataset):
    def __init__(self, data_root):
        """
        Args:
            data_root: 包含所有数据目录的根目录路径
        """
        self.samples = []
        data_root = Path(data_root)
        
        # 遍历所有数据目录
        for episode_dir in sorted(data_root.iterdir()):
            if not episode_dir.is_dir():
                continue
                
            # 遍历每个episode下的数据文件
            for data_dir in sorted(episode_dir.iterdir()):
                if not data_dir.is_dir():
                    continue
                    
                # 检查是否有preference标注
                env = lmdb.open(str(data_dir), readonly=True, lock=False)
                with env.begin() as txn:
                    length = int(txn.get('len'.encode()).decode())
                    
                    # 检查每一帧是否有preference标注
                    for i in range(length):
                        pref = txn.get(f'preference_{i:05d}'.encode())
                        if pref is not None:
                            self.samples.append((str(data_dir), i))
                env.close()
        
        print(f"Found {len(self.samples)} labeled samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data_dir, frame_idx = self.samples[idx]
        
        # 打开LMDB文件
        env = lmdb.open(data_dir, readonly=True, lock=False)
        
        with env.begin() as txn:
            # 读取数据
            rgb = np.frombuffer(txn.get(f'rgb_{frame_idx:05d}'.encode()), np.uint8)
            rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
            rgb = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW
            
            speed = np.frombuffer(txn.get(f'speed_{frame_idx:05d}'.encode()), np.float32)
            preference = float(txn.get(f'preference_{frame_idx:05d}'.encode()).decode())
            
        env.close()
        
        return {
            'rgb': torch.FloatTensor(rgb),
            'speed': torch.FloatTensor(speed),
            'preference': torch.FloatTensor([preference])
        }

def create_dataloader(data_root, batch_size=32, val_split=0.1, num_workers=4):
    """
    创建训练和验证数据加载器
    
    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        val_split: 验证集比例
        num_workers: 数据加载线程数
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    # 创建数据集
    dataset = PreferenceDataset(data_root)
    
    # 划分训练集和验证集
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 