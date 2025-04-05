import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim  # Import optimization algorithms
from torch.distributions import Categorical  # Import Categorical for probabilistic action sampling
import numpy as np  # Import NumPy for numerical computations
import random
import copy
import collections
import tqdm
import lbc.lbc as lbc
from lbc.models import PointModel, RGBPointModel, Converter, spatial_softmax
from lbc.models.spatial_softmax import SpatialSoftmax


from torch.utils.data import Dataset, DataLoader




# 定义Actor网络
class Actor(RGBPointModel):
    def __init__(self, pretrained_model):
        super(Actor, self).__init__(backbone='resnet34',
            pretrained=True,
            height=224, width=480,
            output_channel=60,
            need_prob=True)#输出概率图
        
        
        self.load_state_dict(pretrained_model.state_dict())
        # 冻结backbone和spd_encoder的参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.spd_encoder.parameters():
            param.requires_grad = False



# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        # 复制backbone和spd_encoder的结构及参数
        self.backbone = copy.deepcopy(pretrained_model.backbone)
        self.spd_encoder = copy.deepcopy(pretrained_model.spd_encoder)
        self.kh = pretrained_model.kh
        self.kw = pretrained_model.kw
        self.normalize = copy.deepcopy(pretrained_model.normalize) if hasattr(pretrained_model, 'normalize') else None
        
        # 冻结参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.spd_encoder.parameters():
            param.requires_grad = False
        
        # 计算合并后的特征维度: backbone输出512通道，spd_encoder输出128通道，合并后768通道
        combined_channels = 512 + 128
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(combined_channels * self.kh * self.kw, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, rgb, spd):
        # 预处理与Actor一致
        if self.normalize is not None:
            rgb = self.normalize(rgb / 255.0)
        else:
            rgb = rgb
        
        inputs = self.backbone(rgb)
        #print(inputs.shape)
        spd_embds = self.spd_encoder(spd.unsqueeze(1))  # (batch, 128)
        #print(spd_embds.shape)
        spd_embds = spd_embds.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.kh, self.kw)
        #print(spd_embds.shape)
        combined = torch.cat([inputs, spd_embds], dim=1)
        #print(combined.shape)
        value = self.value_head(combined)
        return value





class ReplayBuffer:
    def __init__(self):
        self.states_rgb = []
        self.states_spd = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.next_states = []

    def clear(self):
        self.states_rgb = []
        self.states_spd = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.next_states = []

    def add(self, rgb, spd, action, logprob, reward, done, next_state):
        self.states_rgb.append(rgb)
        self.states_spd.append(spd)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(done)
        self.next_states.append(next_state)

    def size(self):
        return len(self.rewards)

# ========== PPO类实现 ==========
class PPO:
    def __init__(self, actor, critic, device, lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 epsilon=0.2, c1=1.0, c2=0.01, batch_size=64, epochs=10):
        # 设备配置
        self.device = device
        
        # 网络定义
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        
        # 超参数配置
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.c1 = c1#价值损失系数
        self.c2 = c2#熵正则化损失系数
        self.batch_size = batch_size
        self.epochs = epochs

        # 优化器配置（仅优化可训练参数）
        actor_params = filter(lambda p: p.requires_grad, self.actor.parameters())
        critic_params = filter(lambda p: p.requires_grad, self.critic.parameters())
        self.optimizer = optim.Adam(list(actor_params)+list(critic_params), lr=lr)

    def compute_advantages(self, rewards, values, dones):
        """改进后的GAE计算, 正确处理终止状态"""
        advantages = []
        gae = 0
        next_value = 0
        next_not_done = 1.0
        
        # 逆序计算
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * next_not_done - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_not_done * gae
            advantages.insert(0, gae)
            
            # 更新下一个状态的值和终止标志
            next_value = values[t]
            next_not_done = 1.0 - dones[t]

        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def update(self, memory):
        """使用回放缓冲区的batch进行更新"""
        # 将数据转换为张量
        states_rgb = torch.cat(memory.states_rgb, dim=0).detach()#提取张量并堆叠成一个新张量。不进行反向传播
        states_spd = torch.cat(memory.states_spd, dim=0).detach()
        actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()
        rewards = torch.cat(memory.rewards, dim=0)
        dones = torch.tensor(memory.is_terminals, device=self.device, dtype=torch.float32)

        # 计算价值估计
        with torch.no_grad():
            values = self.critic(states_rgb, states_spd).squeeze()
        
        # 计算优势估计
        advantages = self.compute_advantages(rewards, values, dones)
        returns = advantages + values

        # 创建数据集
        # print(states_rgb.size())
        # print(states_spd.size())
        # print(actions.size())
        # print(old_logprobs.size())
        # print(returns.size())
        # print(advantages.size())
        dataset = torch.utils.data.TensorDataset(
            states_rgb, states_spd, actions, old_logprobs, returns, advantages
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 多轮参数更新
        for _ in range(self.epochs):
            for batch in tqdm.tqdm(dataloader):
                rgb_batch, spd_batch, a_batch, old_lp_batch, ret_batch, adv_batch = batch
                #print("old log shape: ", old_lp_batch.shape)
                # 评估当前策略
                log_probs, values = self.evaluate(rgb_batch, spd_batch, a_batch)
                log_probs = log_probs.reshape(old_lp_batch.shape)
                #print("new log shape: ", log_probs.shape)
                # 计算重要性采样比例
                ratios = torch.exp(log_probs - old_lp_batch)

                # 策略损失
                adv_batch = adv_batch.unsqueeze(1)
                #print("adv size : ", adv_batch.shape)
                surr1 = ratios * adv_batch
                surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = F.mse_loss(values, ret_batch)

                # 熵正则化
                entropy_loss = -log_probs.mean()

                # 总损失
                loss = policy_loss + self.c1*value_loss + self.c2*entropy_loss

                # 参数更新
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # 添加梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

        # 清空缓冲区
        
        memory.clear()

    def evaluate(self, states_rgb, states_spd, actions):
        """评估当前策略"""
        action_probs, action_x, action_y = self.actor(states_rgb, states_spd)

        batch_size, output_channel, height_width = action_probs.shape
        
        flattened_probs = action_probs.view(batch_size * output_channel, -1)  

        indices = torch.multinomial(flattened_probs, num_samples=1)  
        action_x = self.gather(action_x, 1, indices, batch_size, output_channel)
        action_y = self.gather(action_y, 1, indices, batch_size, output_channel)
        action = torch.stack((action_x, action_y), dim=1).squeeze()      
        # 计算对数概率
        log_probs = torch.log(torch.gather(flattened_probs, 1, indices))
        values = self.critic(states_rgb, states_spd).squeeze()
        return log_probs, values

    def select_action(self, state_rgb, state_spd, memory):
        """改进的动作选择，自动处理设备转换"""
        rgb_tensor = torch.tensor(state_rgb, dtype=torch.float32, device=self.device)
        spd_tensor = torch.tensor(state_spd, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            action_probs, action_x, action_y = self.actor(rgb_tensor, spd_tensor)
        
            # 采样逻辑修改
        batch_size, output_channel, height_width = action_probs.shape
        #batch_list = torch.unbind(action_probs, dim=0)
        
        flattened_probs = action_probs.view(batch_size * output_channel, -1)  
        # 多项式采样（每行采样一个索引）
        indices = torch.multinomial(flattened_probs, num_samples=1)  
        # 恢复为 (B, C, 1)
        #indices = indices.view(batch_size, output_channel, 1)
    
        

        action_x = self.gather(action_x, 1, indices, batch_size, output_channel)
        action_y = self.gather(action_y, 1, indices, batch_size, output_channel)
        action = torch.stack((action_x, action_y), dim=1).squeeze()
        # print("indices shape :",indices.shape)
        # print('action shape : ',action.shape)
        # print("action_prob shape :",action_probs.shape)
        
        # 计算对数概率
        log_prob = torch.log(torch.gather(flattened_probs, 1, indices)).squeeze()
        
        # 存储到记忆缓冲区
        memory.states_rgb.append(rgb_tensor)
        memory.states_spd.append(spd_tensor)
        memory.actions.append(action)
        memory.logprobs.append(log_prob)

        return action, log_prob

    def gather(self, action, dim, indices, batch_size, output_channel):
        action = torch.unsqueeze(action, 0).repeat(batch_size*output_channel,1)
        #print(action.shape)
        return torch.gather(action, dim, indices)

    def run():
        pretrained_model = RGBPointModel(
                'resnet34',
                pretrained=True,
                height=224, width=480,#h=224,w=480
                output_channel=60,#10*6
                pred_seg=False,
                need_prob=True
            ).to(torch.device('cuda'))
        pretrained_model.load_state_dict(torch.load('pretrained.pth'))
        actor = Actor(pretrained_model)
        critic = Critic(pretrained_model)

# 注意初始化PPO时需要传入device参数：
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ppo = PPO(actor, critic, device, ...)