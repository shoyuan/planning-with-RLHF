import os
import cv2
import lmdb
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def draw_path_points(img, points, color=(0, 255, 0), thickness=2):
    """在图像上绘制路径点"""
    h, w = img.shape[:2]
    for i in range(len(points)-1):
        pt1 = (int(points[i][0]), int(points[i][1]))
        pt2 = (int(points[i+1][0]), int(points[i+1][1]))
        cv2.line(img, pt1, pt2, color, thickness)
        cv2.circle(img, pt1, 2, color, -1)
    # 绘制最后一个点
    if len(points) > 0:
        cv2.circle(img, (int(points[-1][0]), int(points[-1][1])), 2, color, -1)

def visualize_frame(rgb, pred_loc, world_loc, control, speed, cmd, position, rotation):
    """可视化单帧数据"""
    # 复制图像以避免修改原始数据
    vis_img = rgb.copy()
    
    # 在图像上绘制预测的路径点
    draw_path_points(vis_img, pred_loc, color=(0, 255, 0))  # 绿色表示相机坐标系下的预测点
    
    # 添加文本信息
    text_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 20
    
    # 构建要显示的文本信息
    info_texts = [
        f"Speed: {speed[0]:.2f} m/s",
        f"Steer: {control[0]:.2f}, Throttle: {control[1]:.2f}, Brake: {control[2]:.2f}",
        f"Command: {['left','right','straight','follow','change left','change right'][int(cmd[0])]}",
        f"Position: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})",
        f"Rotation: ({rotation[0]:.1f}, {rotation[1]:.1f}, {rotation[2]:.1f})"
    ]
    
    # 绘制半透明背景
    overlay = vis_img.copy()
    bg_height = (len(info_texts) + 1) * line_height
    cv2.rectangle(overlay, (0, 0), (400, bg_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
    
    # 添加文本
    for i, text in enumerate(info_texts):
        position = (10, (i + 1) * line_height)
        cv2.putText(vis_img, text, position, font, font_scale, text_color, thickness)
    
    return vis_img

def visualize_lmdb(lmdb_path, output_dir=None, start_frame=0, end_frame=None, save_video=False):
    """可视化LMDB数据"""
    # 打开LMDB环境
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    with env.begin() as txn:
        # 获取数据长度
        length = int(txn.get('len'.encode()).decode())
        print(f"Total frames: {length}")
        
        # 设置结束帧
        if end_frame is None:
            end_frame = length
        end_frame = min(end_frame, length)
        
        # 设置视频写入器
        if save_video:
            output_path = os.path.join(output_dir, 'visualization.mp4') if output_dir else 'visualization.mp4'
            first_frame = cv2.imdecode(np.frombuffer(txn.get(f'rgb_{0:05d}'.encode()), np.uint8), cv2.IMREAD_COLOR)
            h, w = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
        
        # 遍历每一帧
        for i in range(start_frame, end_frame):
            # 读取数据
            rgb = cv2.imdecode(np.frombuffer(txn.get(f'rgb_{i:05d}'.encode()), np.uint8), cv2.IMREAD_COLOR)
            pred_loc = np.frombuffer(txn.get(f'pred_loc_{i:05d}'.encode()), np.float32).reshape(-1, 2)
            world_loc = np.frombuffer(txn.get(f'world_loc_{i:05d}'.encode()), np.float32).reshape(-1, 2)
            control = np.frombuffer(txn.get(f'control_{i:05d}'.encode()), np.float32)
            speed = np.frombuffer(txn.get(f'speed_{i:05d}'.encode()), np.float32)
            cmd = np.frombuffer(txn.get(f'cmd_{i:05d}'.encode()), np.float32)
            position = np.frombuffer(txn.get(f'position_{i:05d}'.encode()), np.float32)
            rotation = np.frombuffer(txn.get(f'rotation_{i:05d}'.encode()), np.float32)
            
            # 可视化当前帧
            vis_img = visualize_frame(rgb, pred_loc, world_loc, control, speed, cmd, position, rotation)
            
            if save_video:
                video_writer.write(vis_img)
            else:
                # 显示图像
                cv2.imshow('Visualization', vis_img)
                key = cv2.waitKey(50)  # 50ms延迟
                if key == 27:  # ESC键退出
                    break
        
        if save_video:
            video_writer.release()
    
    env.close()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Visualize LBC collected data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to LMDB data directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for saved visualizations')
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame for visualization')
    parser.add_argument('--end_frame', type=int, default=None, help='End frame for visualization')
    parser.add_argument('--save_video', action='store_true', help='Save visualization as video')
    
    args = parser.parse_args()
    
    # 如果指定了输出目录，确保它存在
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取数据文件列表
    if os.path.isdir(args.data_path):
        data_files = [f for f in Path(args.data_path).glob('lbc_data_*')]
        print(f"Found {len(data_files)} data files")
        
        for data_file in data_files:
            print(f"\nVisualizing {data_file}")
            visualize_lmdb(str(data_file), args.output_dir, args.start_frame, args.end_frame, args.save_video)
    else:
        visualize_lmdb(args.data_path, args.output_dir, args.start_frame, args.end_frame, args.save_video)

if __name__ == '__main__':
    main()