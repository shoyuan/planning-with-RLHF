from runners import ScenarioRunner
import ray
import argparse
import os
import numpy as np
import cv2
from pathlib import Path

def main(args):
    # 定义可用的城镇地图
    towns = {i: f'Town{i+1:02d}' for i in range(7)}
    towns.update({7: 'Town10HD'})

    # 使用无场景配置
    scenario = 'assets/no_scenarios.json'
    route = 'assets/routes_all.xml'

    # 使用LBC agent
    args.agent = 'autoagents/lbc_agent.py'
    args.agent_config = 'lbc_config.yaml'  # LBC的配置文件

    # 创建输出目录
    output_dir = Path('collected_data')
    output_dir.mkdir(exist_ok=True)
    
    jobs = []
    for i in range(args.num_runners):
        scenario_class = 'train_scenario'
        town = towns.get(i, 'Town03')
        port = (i+1) * args.port
        tm_port = port + 2
        
        # 为每个运行实例创建独立的检查点和输出目录
        checkpoint = output_dir / f'runner_{i:02d}' / 'checkpoint.json'
        checkpoint.parent.mkdir(exist_ok=True)
        
        # 创建并运行scenario runner
        runner = ScenarioRunner.remote(
            args,
            scenario_class=scenario_class,
            scenario=scenario,
            route=route,
            checkpoint=str(checkpoint),
            town=town,
            port=port,
            tm_port=tm_port,
            debug=args.debug
        )
        jobs.append(runner.run.remote())

    try:
        ray.wait(jobs, num_returns=args.num_runners)
    except Exception as e:
        print("采集过程发生错误:", str(e))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA LBC数据采集工具')
    
    parser.add_argument('--num-runners', type=int, default=1,
                        help='并行运行的采集器数量')
    parser.add_argument('--host', default='localhost',
                        help='CARLA 服务器IP')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA 服务器端口')
    parser.add_argument('--timeout', default="600.0",
                        help='CARLA 客户端超时时间(秒)')
    parser.add_argument('--debug', type=bool, default=False,
                        help='是否启用调试模式')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='每条路线重复次数')
    parser.add_argument('--track', type=str, default='SENSORS',
                        help='参与类型: SENSORS, MAP')
    parser.add_argument('--resume', type=bool, default=True,
                        help='是否从检查点恢复')

    args = parser.parse_args()

    # 初始化Ray
    ray.init(logging_level=10, local_mode=True, log_to_driver=False)
    
    main(args)