import os
import math
import yaml
import lmdb
import numpy as np
import torch
import wandb
import carla
import random
import string
from collections import deque
from torch.distributions.categorical import Categorical

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from utils import visualize_obs, _numpy
from utils.ls_fit import ls_circle, project_point_to_circle, signed_angle

from lbc.models import RGBPointModel, Converter
from rlhf.models import PPO
from rlhf.models import reward_model
from autoagents.waypointer import Waypointer

def get_entry_point():
    return 'PPOAgent'

class PPOAgent(AutonomousAgent):
    """
    RLHF PPO agent
    """
    
    def setup(self, path_to_conf_file):
        
        self.track = Track.SENSORS
        self.num_frames = 0
        self.num_cmds = 6
        self.dt = 1./20
        self.N = 10
        
        self.alpha_errors = deque()
        self.accel_errors = deque()
        
        with open(path_to_conf_file, 'r') as f:
            config = yaml.safe_load(f)
            
        for key, value in config.items():
            setattr(self, key, value)
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'    
        self.device = torch.device('cuda')
        
        self.rgb_model = RGBPointModel(
            'resnet34',
            pretrained=True,
            height=240-self.crop_top-self.crop_bottom, width=480,#h=224,w=480
            output_channel=self.num_plan*self.num_cmds,#10*6
            need_prob = True
        ).to(self.device)
        self.rgb_model.load_state_dict(torch.load(self.rgb_model_dir))
        #self.rgb_model.eval()
        
        self.actor = PPO.Actor(self.rgb_model)
        self.critic = PPO.Critic(self.rgb_model)
        self.memory = PPO.ReplayBuffer()

        self.ppo_model = PPO.PPO(self.actor, self.critic, self.device)
        self.reward_model = reward_model.RewardModel(self.device)
        self.reward_model.load_state_dict(torch.load(self.reward_model_dir))
        self.reward_model.to(self.device)
        self.reward_model.eval()

        self.converter = Converter(offset=6.0, scale=[1.5, 1.5]).to(self.device)
        
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
        
        self.vizs = []
        
        self.waypointer = None
        
        self.lane_change_counter = 0
        self.stop_counter = 0
        self.lane_changed = None

        if self.log_wandb:
            wandb.init(project='carla_evaluate')

    def destroy(self):
        if len(self.vizs) == 0:
            return

        self.flush_data()
        self.prev_steer = 0
        self.lane_change_counter = 0
        self.stop_counter = 0
        self.lane_changed = None
        
        self.alpha_errors.clear()
        self.accel_errors.clear()

        del self.waypointer
        del self.rgb_model
        del self.ppo_model
        del self.reward_model
        
    def flush_data(self):

        if self.log_wandb:
            wandb.log({
                'vid': wandb.Video(np.stack(self.vizs).transpose((0,3,1,2)), fps=20, format='mp4')
            })
        self.memory.is_terminals.append(1.0)
        self.ppo_model.update(self.memory)
        torch.save(self.ppo_model, self.ppo_model_dir)
        self.vizs.clear()
        
    def sensors(self):
        sensors = [
            {'type': 'sensor.collision', 'id': 'COLLISION'},
            {'type': 'sensor.speedometer', 'id': 'EGO'},
            {'type': 'sensor.other.gnss', 'x': 0., 'y': 0.0, 'z': self.camera_z, 'id': 'GPS'},
            {'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'RGB_0'},
            {'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'RGB_1'},
            {'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw':  55.0,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'RGB_2'},
        ]
        
        return sensors
        
    def run_step(self, input_data, timestamp):
        
        _, rgb_0 = input_data.get(f'RGB_0')
        _, rgb_1 = input_data.get(f'RGB_1')
        _, rgb_2 = input_data.get(f'RGB_2')
        rgb = np.concatenate([rgb_0[...,:3], rgb_1[...,:3], rgb_2[...,:3]], axis=1)
        
        # Crop images
        _rgb = rgb[self.crop_top:-self.crop_bottom,:,:3]

        _rgb = _rgb[...,::-1].copy()
        
        _, ego = input_data.get('EGO')
        _, gps = input_data.get('GPS')
        _, col = input_data.get('COLLISION')
        
        if self.waypointer is None:
            self.waypointer = Waypointer(self._global_plan, gps)

        _, _, cmd = self.waypointer.tick(gps)
        
        spd = ego.get('spd')
        
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
            
        _rgb = torch.tensor(_rgb[None]).float().permute(0,3,1,2).to(self.device)
        _spd = torch.tensor([spd]).float().to(self.device)
        
        # with torch.no_grad():
        pred_locs, pred_locs_prob = self.ppo_model.select_action(_rgb, _spd, self.memory)
        pred_locs = pred_locs.view(self.num_cmds, self.num_plan, 2)
        pred_locs = (pred_locs + 1) * self.rgb_model.img_size/2
            
        pred_loc = self.converter.cam_to_world(pred_locs[cmd_value])
        pred_loc = torch.flip(pred_loc, [-1])
        #print("pred_loc shape :",pred_loc.shape)

        with torch.no_grad():
            _spd.unsqueeze_(0)
            reward = self.reward_model(_rgb, _spd, pred_loc.unsqueeze(0))

        self.memory.rewards.append(reward)
        
        steer, throt, brake = self.get_control(_numpy(pred_loc.squeeze(0)), cmd_value, float(spd))
    
        self.vizs.append(visualize_obs(rgb, 0, (steer, throt, brake), spd, cmd=cmd_value+1))

        if col:
            self.flush_data()
            raise Exception('Collector has collided!')

        if len(self.vizs) > 1000:
            self.flush_data()
        else:
            self.memory.is_terminals.append(0.0)
        
        self.num_frames += 1

        return carla.VehicleControl(steer=steer, throttle=throt, brake=brake)
        
    def get_control(self, locs, cmd, spd):

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

        return steer, throt, brake
