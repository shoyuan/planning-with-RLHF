import os
import numpy as np
import lmdb
import random
import string
import torch
import carla
import cv2

from waypointer import Waypointer
from utils import visualize_obs
from pathlib import Path

from autoagents.lbc_agent import LBCAgent
from utils import _numpy

class LBCCollector(LBCAgent):
    """
    用于收集LBC模型数据的代理
    继承自LBCAgent, 添加数据采集功能
    """
    def setup(self, path_to_conf_file):
        # 首先执行原始LBC代理的设置
        super().setup(path_to_conf_file)
        
        # 初始化数据收集相关的变量
        self.data_save_path = Path('expirements/data/lbc_data')
        self.data_save_path.mkdir(parents=True, exist_ok=True)
        
        # 数据存储列表
        self.rgbs = []          # RGB图像
        self.pred_locs = []     # 模型预测的路径点
        self.world_locs = []    # 转换到世界坐标系的路径点
        self.controls = []      # 控制指令(转向、油门、刹车)
        self.speeds = []        # 速度
        self.cmds = []          # 导航指令
        self.positions = []     # 车辆位置
        self.rotations = []     # 车辆旋转
        
        self.save_freq = 100    # 每收集100帧保存一次
        
    def destroy(self):
        """清理资源时保存数据"""
        if len(self.rgbs) > 0:
            self.flush_data()
        
        # 调用父类的destroy方法
        super().destroy()

    def flush_data(self):
        """将收集的数据保存到磁盘"""
        # 生成随机文件名
        data_path = self.data_save_path / f'lbc_data_{self._random_string()}'
        print(f'Saving data to {data_path}')
        
        # 创建LMDB环境
        lmdb_env = lmdb.open(str(data_path), map_size=int(1e10))
        
        length = len(self.rgbs)
        with lmdb_env.begin(write=True) as txn:
            # 保存数据长度
            txn.put('len'.encode(), str(length).encode())
            
            # 保存每一帧的数据
            for i in range(length):
                # 保存RGB图像
                success, encoded_image = cv2.imencode('.jpg', self.rgbs[i])

                if success:
                    txn.put(
                        f'rgb_{i:05d}'.encode(),
                        encoded_image.tobytes()
                    )
                else:
                    print('Failed to encode image')
                    # 处理编码失败的情况
                
                # 保存模型预测的路径点
                txn.put(
                    f'pred_loc_{i:05d}'.encode(),
                    np.ascontiguousarray(self.pred_locs[i]).astype(np.float32)
                )
                
                # 保存世界坐标系下的路径点
                txn.put(
                    f'world_loc_{i:05d}'.encode(),
                    np.ascontiguousarray(self.world_locs[i]).astype(np.float32)
                )
                
                # 保存控制指令
                txn.put(
                    f'control_{i:05d}'.encode(),
                    np.ascontiguousarray(self.controls[i]).astype(np.float32)
                )
                
                # 保存速度
                txn.put(
                    f'speed_{i:05d}'.encode(),
                    np.ascontiguousarray(self.speeds[i]).astype(np.float32)
                )
                
                # 保存导航指令
                txn.put(
                    f'cmd_{i:05d}'.encode(),
                    np.ascontiguousarray(self.cmds[i]).astype(np.float32)
                )
                
                # 保存位置
                txn.put(
                    f'position_{i:05d}'.encode(),
                    np.ascontiguousarray(self.positions[i]).astype(np.float32)
                )
                
                # 保存旋转
                txn.put(
                    f'rotation_{i:05d}'.encode(),
                    np.ascontiguousarray(self.rotations[i]).astype(np.float32)
                )

        # 清空数据缓存
        self.rgbs.clear()
        self.pred_locs.clear()
        self.world_locs.clear()
        self.controls.clear()
        self.speeds.clear()
        self.cmds.clear()
        self.positions.clear()
        self.rotations.clear()
        
        lmdb_env.close()

    def run_step(self, input_data, timestamp):
        """重写run_step方法以添加数据收集功能"""
        # 获取RGB图像
        _, rgb_0 = input_data.get(f'RGB_0')
        _, rgb_1 = input_data.get(f'RGB_1')
        _, rgb_2 = input_data.get(f'RGB_2')
        rgb = np.concatenate([rgb_0[...,:3], rgb_1[...,:3], rgb_2[...,:3]], axis=1)
        
        # 获取车辆状态
        _, ego = input_data.get('EGO')
        _, gps = input_data.get('GPS')
        
        # 裁剪图像
        _rgb = rgb[self.crop_top:-self.crop_bottom,:,:3]
        _rgb = _rgb[...,::-1].copy()
        
        # 初始化waypointer
        if self.waypointer is None:
            self.waypointer = Waypointer(self._global_plan, gps)
        _, _, cmd = self.waypointer.tick(gps)
        
        spd = ego.get('spd')
        cmd_value = cmd.value-1
        cmd_value = 3 if cmd_value < 0 else cmd_value
        
        # 处理车道变换
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
            
        # 模型推理
        _rgb = torch.tensor(_rgb[None]).float().permute(0,3,1,2).to(self.device)
        _spd = torch.tensor([spd]).float().to(self.device)
        
        with torch.no_grad():
            pred_locs = self.rgb_model(_rgb, _spd, pred_seg=False).view(self.num_cmds,self.num_plan,2)
            pred_locs = (pred_locs + 1) * self.rgb_model.img_size/2
            
            pred_loc = self.converter.cam_to_world(pred_locs[cmd_value])
            pred_loc = torch.flip(pred_loc, [-1])
        
        # 获取控制指令
        steer, throt, brake = self.get_control(_numpy(pred_loc), cmd_value, float(spd))
        
        # 收集数据
        self.rgbs.append(rgb.copy())
        self.pred_locs.append(_numpy(pred_locs[cmd_value]))
        self.world_locs.append(_numpy(pred_loc))
        self.controls.append(np.array([steer, throt, brake]))
        self.speeds.append(np.array([spd]))
        self.cmds.append(np.array([cmd_value]))
        self.positions.append(ego.get('loc'))
        self.rotations.append(ego.get('rot'))
        
        # 可视化
        self.vizs.append(visualize_obs(rgb, 0, (steer, throt, brake), spd, cmd=cmd_value+1))
        
        # 定期保存数据
        if len(self.rgbs) >= self.save_freq:
            self.flush_data()
        
        self.num_frames += 1
        
        return carla.VehicleControl(steer=steer, throttle=throt, brake=brake)

    @staticmethod
    def _random_string(length=10):
        """生成随机字符串作为文件名"""
        return ''.join(random.choice(string.ascii_lowercase) for i in range(length))