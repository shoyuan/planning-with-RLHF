from runners import ScenarioRunner
import ray
import argparse
import os
import numpy as np
import cv2
from pathlib import Path
import xml.etree.ElementTree as ET
from lbc.models import RGBPointModel, Converter

def load_routes_info(routes_file):
    """加载路线信息"""
    routes = []
    tree = ET.parse(routes_file)
    root = tree.getroot()
    
    for route in root.findall('route'):
        route_id = route.get('id')
        town = route.get('town')
        routes.append({
            'id': route_id,
            'town': town
        })
    
    return routes

def main(args):
    # 定义可用的城镇地图
    towns = {i: f'Town{i+1:02d}' for i in range(7)}
    towns.update({7: 'Town10HD'})

    # 加载路线信息
    routes = load_routes_info('assets/routes_50.xml')
    
    # 使用无场景配置
    scenario = 'assets/all_towns_traffic_scenarios.json'
    route = 'assets/routes_50.xml'

    # 使用新的collector agent
    args.agent = 'autoagents/collector_agents/lbc_collector'
    args.agent_config = 'expirements/config/config_prefer.yaml'

    jobs = []
    for i in range(args.num_runners):
        scenario_class = args.scenario
        town = towns.get(i, 'Town03')
        port = (i+1) * args.port
        tm_port = port + 2
        
        # 修改为与 data_phase1.py 一致的检查点格式
        checkpoint = f'expirements/sim_results/{i:02d}_{args.checkpoint}'
        
        runner = ScenarioRunner.remote(
            args, scenario_class, scenario, route,
            checkpoint=checkpoint,
            town=town,
            port=port, tm_port=tm_port
        )
        jobs.append(runner.run.remote())
    
    ray.wait(jobs, num_returns=args.num_runners)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA LBC数据采集工具')
    
    parser.add_argument('--num-runners', type=int, default=1,
                        help='并行运行的采集器数量')
    parser.add_argument('--scenario', choices=['train_scenario', 'nocrash_train_scenario'], 
                        default='nocrash_train_scenario')
    
    parser.add_argument('--host', default='localhost',
                        help='CARLA 服务器IP')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA 服务器端口')
    parser.add_argument('--timeout', default="600.0",
                        help='CARLA 客户端超时时间(秒)')
    parser.add_argument('--debug', type=bool, default=True,
                        help='是否启用调试模式')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='每条路线重复次数')
    parser.add_argument('--track', type=str, default='SENSORS',
                        help='参与类型: SENSORS, MAP')
    parser.add_argument('--resume', type=bool, default=True,
                        help='是否从检查点恢复')
    parser.add_argument('--checkpoint', type=str,
                        default='prefer_simulation_results.json',
                        help='检查点文件名，用于保存和恢复模拟状态')
    parser.add_argument('--model-epoch', type=int, required=True,
                       help='LBC model checkpoint epoch number')
    parser.add_argument('--trafficManagerSeed', type=int, default=0,
                        help='交通管理器的随机种子')

    args = parser.parse_args()

    # 初始化Ray
    ray.init(logging_level=40, local_mode=True, log_to_driver=True)
    
    main(args)