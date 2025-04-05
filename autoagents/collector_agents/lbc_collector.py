import os
import math
import yaml
import lmdb
import numpy as np
import torch
import wandb
import carla
from collections import deque
import cv2
import string
import random


from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from utils import visualize_obs, _numpy
from lbc.models import RGBPointModel, Converter
from autoagents.waypointer import Waypointer
from utils.ls_fit import ls_circle, project_point_to_circle, signed_angle

def get_entry_point():
    return 'LBCCollector'

class LBCCollector(AutonomousAgent):
    """LBC数据收集器"""
    
    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        self.num_frames = 0
        self.data_index = 0
        # 首先设置基本属性
        self.num_cmds = 6
        self.dt = 1./20
        self.N = 10
        
        # 读取配置
        with open(path_to_conf_file, 'r') as f:
            config = yaml.safe_load(f)
        
        for key, value in config.items():
            setattr(self, key, value)
            
        # 设置设备
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.device = torch.device('cuda')
        
        # 初始化LBC模型
        self.rgb_model = RGBPointModel(
            'resnet34',
            pretrained=True,
            height=240-self.crop_top-self.crop_bottom,
            width=480,
            output_channel=self.num_plan*self.num_cmds
        ).to(self.device)
        self.rgb_model.load_state_dict(torch.load(self.rgb_model_dir))
        self.rgb_model.eval()
        
        self.converter = Converter(offset=6.0, scale=[1.5, 1.5]).to(self.device)
        
        # 初始化数据存储
        self.lbls = []
        self.vizs = []
        self.rgbs = []
        self.pred_locs = []
        self.world_locs = []
        self.map_locs = []
        self.controls = []
        self.speeds = []
        self.cmds = []
        self.positions = []
        self.rotations = []
        
        self.waypointer = None
        
        if self.log_wandb:
            wandb.init(project='lbc_collector', config=config)
        
        # 添加控制相关的属性
        self.alpha_errors = deque()
        self.accel_errors = deque()
        
        self.steer_points = {0: 4, 1: 2, 2: 2, 3: 3, 4: 3, 5: 3}
        self.steer_pids = {
            0 : {"Kp": 2.0, "Ki": 0.1, "Kd":0}, # Left
            1 : {"Kp": 1.5, "Ki": 0.1, "Kd":0}, # Right
            2 : {"Kp": 0.5, "Ki": 0.0, "Kd":0}, # Straight
            3 : {"Kp": 1.5, "Ki": 0.1, "Kd":0}, # Follow
            4 : {"Kp": 1.5, "Ki": 0.1, "Kd":0}, # Change Left
            5 : {"Kp": 1.5, "Ki": 0.1, "Kd":0}, # Change Right
        }
        self.accel_pids = {"Kp": 2.0, "Ki": 0.2, "Kd":0}

    def sensors(self):
        """配置传感器"""
        return [
            {'type': 'sensor.map', 'id': 'MAP'},
            {'type': 'sensor.collision', 'id': 'COLLISION'},
            {'type': 'sensor.speedometer', 'id': 'EGO'},
            {'type': 'sensor.other.gnss', 'x': 0., 'y': 0.0, 'z': self.camera_z, 'id': 'GPS'},
            {'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'RGB_0'},
            {'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'RGB_1'},
            {'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'RGB_2'},
        ]

    def run_step(self, input_data, timestamp):
        """运行一步"""
        # 获取传感器数据
        _, rgb_0 = input_data.get('RGB_0')
        _, rgb_1 = input_data.get('RGB_1')
        _, rgb_2 = input_data.get('RGB_2')
        rgb = np.concatenate([rgb_0[...,:3], rgb_1[...,:3], rgb_2[...,:3]], axis=1)
        
        # 裁剪图像
        rgb = rgb[self.crop_top:-self.crop_bottom,:,:3]
        rgb = rgb[...,::-1].copy()  # BGR -> RGB
        
        _, lbl = input_data.get('MAP')
        _, ego = input_data.get('EGO')
        _, gps = input_data.get('GPS')
        _, col = input_data.get('COLLISION')
        
        # 获取导航命令
        if self.waypointer is None:
            self.waypointer = Waypointer(self._global_plan, gps)
        _, _, cmd = self.waypointer.tick(gps)
        
        # 获取车辆状态
        spd = ego.get('spd')
        yaw = ego.get('rot')[-1]

        cmd_value = cmd.value-1
        cmd_value = 3 if cmd_value < 0 else cmd_value

        if cmd_value in [4,5]:
            if self.lane_changed is not None and cmd_value != self.lane_changed:
                self.lane_change_counter = 0

            self.lane_change_counter += 1
            self.lane_changed = cmd_value if self.lane_change_counter > {4:200,5:200}.get(cmd_value) else None
        else:
            self.lane_change_counter = 0
            self.lane_changed = None
            
        if cmd_value == self.lane_changed:
            cmd_value = 3

        # 模型预测
        _rgb = torch.tensor(rgb[None]).float().permute(0,3,1,2).to(self.device)
        _spd = torch.tensor([spd]).float().to(self.device)
        
        with torch.no_grad():
            pred_locs = self.rgb_model(_rgb, _spd, pred_seg=False).view(self.num_cmds, self.num_plan, 2)
            pred_locs = (pred_locs + 1) * self.rgb_model.img_size/2

            pred_loc = self.converter.cam_to_world(pred_locs[cmd_value])
            map_loc = self.converter.world_to_map(pred_loc)
            pred_loc = torch.flip(pred_loc, [-1])

            
        
        # 获取控制信号
        
        control = self.get_control(_numpy(pred_loc), cmd_value, spd)
        
        # 保存数据
        if self.num_frames % (self.num_repeat+1) == 0:
            self.rgbs.append(rgb)
            self.pred_locs.append(_numpy(pred_locs[cmd_value]))
            self.world_locs.append(_numpy(pred_loc))
            self.map_locs.append(_numpy(map_loc))
            self.controls.append(np.array([control.steer, control.throttle, control.brake]))
            self.speeds.append(np.array([spd]))
            self.cmds.append(np.array([cmd_value]))
            self.positions.append(ego.get('loc'))
            self.rotations.append(ego.get('rot'))
            self.lbls.append(lbl)
            
            
            self.vizs.append(visualize_obs(
                rgb, yaw/180*math.pi,
                (control.steer, control.throttle, control.brake),
                spd, cmd=cmd.value
            ))
        
        if len(self.vizs) > self.num_per_flush:
            self.flush_data()
            
        if col:
            self.flush_data()
            raise Exception('Collector has collided!')
            
        self.num_frames += 1
        return control

    def get_control(self, locs, cmd, spd):
        """计算控制信号"""
        # 从lbc_agent.py复制的控制逻辑
        locs = np.concatenate([[[0, 0]], locs], 0)
        c, r = ls_circle(locs)

        n = self.steer_points.get(cmd, 1)
        closest = project_point_to_circle(locs[n], c, r)

        v = [0.0, 1.0, 0.0]
        w = [closest[0], closest[1], 0.0]
        alpha = -signed_angle(v, w)

        # Compute steering
        self.alpha_errors.append(alpha)
        if len(self.alpha_errors) > self.N:
            self.alpha_errors.pop()

        if len(self.alpha_errors) >= 2:
            integral = sum(self.alpha_errors) * self.dt
            derivative = (self.alpha_errors[-1] - self.alpha_errors[-2]) / self.dt
        else:
            integral = 0.0
            derivative = 0.0

        steer = 0.0
        steer += self.steer_pids[cmd]['Kp'] * alpha
        steer += self.steer_pids[cmd]['Ki'] * integral
        steer += self.steer_pids[cmd]['Kd'] * derivative

        # Compute throttle and brake
        tgt_spd = np.linalg.norm(locs[:-1] - locs[1:], axis=1).mean()
        accel = tgt_spd - spd

        # Compute acceleration
        self.accel_errors.append(accel)
        if len(self.accel_errors) > self.N:
            self.accel_errors.pop()

        if len(self.accel_errors) >= 2:
            integral = sum(self.accel_errors) * self.dt
            derivative = (self.accel_errors[-1] - self.accel_errors[-2]) / self.dt
        else:
            integral = 0.0
            derivative = 0.0

        throt = 0.0
        throt += self.accel_pids['Kp'] * accel
        throt += self.accel_pids['Ki'] * integral
        throt += self.accel_pids['Kd'] * derivative

        if throt > 0:
            brake = 0.0
        else:
            brake = -throt
            throt = max(0, throt)

        if tgt_spd < 0.5:
            steer = 0.0
            throt = 0.0
            brake = 1.0

        return carla.VehicleControl(steer=steer, throttle=throt, brake=brake)

    def destroy(self):
        if len(self.vizs) > 0:
            self.flush_data()

    def flush_data(self):
        """保存数据"""
        if self.log_wandb:
            wandb.log({
                'vid': wandb.Video(np.stack(self.vizs).transpose((0,3,1,2)), fps=20, format='mp4')
            })
        
        data_path = os.path.join(self.main_data_dir, f'prefer_data{self.num_frames}_{_random_string()}' )
        print(f'Saving to {data_path}')
        
        lmdb_env = lmdb.open(data_path, map_size=int(1e10))
        
        length = len(self.rgbs)
        with lmdb_env.begin(write=True) as txn:
            txn.put('len'.encode(), str(length).encode())
            
            for i in range(length):
                txn.put(f'rgb_{i:05d}'.encode(),
                        cv2.imencode('.jpg', self.rgbs[i])[1].tobytes())
                        
                txn.put(f'pred_loc_{i:05d}'.encode(),
                        self.pred_locs[i].astype(np.float32))
                        
                txn.put(f'world_loc_{i:05d}'.encode(),
                        self.world_locs[i].astype(np.float32))
                
                txn.put(f'map_loc_{i:05d}'.encode(),
                        self.map_locs[i].astype(np.float32))
                        
                txn.put(f'control_{i:05d}'.encode(),
                        self.controls[i].astype(np.float32))
                        
                txn.put(f'speed_{i:05d}'.encode(),
                        self.speeds[i].astype(np.float32))
                        
                txn.put(f'cmd_{i:05d}'.encode(),
                        self.cmds[i].astype(np.float32))
                        
                # 修复: 将列表转换为numpy数组
                txn.put(f'position_{i:05d}'.encode(),
                        np.array(self.positions[i]).astype(np.float32))
                        
                txn.put(f'rotation_{i:05d}'.encode(),
                        np.array(self.rotations[i]).astype(np.float32))
                
                txn.put(f'lbl_{i:05d}'.encode(),
                        np.array(self.lbls[i]).tobytes())
        self.data_index += 1
        # 清空缓存
        self.vizs.clear()
        self.rgbs.clear()
        self.pred_locs.clear()
        self.world_locs.clear()
        self.controls.clear()
        self.speeds.clear()
        self.cmds.clear()
        self.positions.clear()
        self.rotations.clear()
        
        lmdb_env.close()

def _random_string(length=5):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))