import os
import cv2
import lmdb
import numpy as np
import argparse
from pathlib import Path


def draw_path_points(img, points, color=(0, 255, 0), thickness=2):
    """Draw path points on image"""
    h, w = img.shape[:2]
    
    # Convert points from BEV (Bird's Eye View) to image coordinates
    for i in range(len(points)-1):
        # Scale and center the points
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        
        # Convert to pixel coordinates (similar to logger.py)
        pt1 = (int(x1 + w//2), int(h - y1))  # Center x, invert y
        pt2 = (int(x2 + w//2), int(h - y2))  # Center x, invert y
        
        cv2.line(img, pt1, pt2, color, thickness)
        cv2.circle(img, pt1, 3, color, -1)
    
    # Draw the last point
    if len(points) > 0:
        x, y = points[-1]
        last_point = (int(x + w//2), int(h - y))
        cv2.circle(img, last_point, 3, color, -1)

def visualize_frame(rgb, pred_loc, control, speed, cmd, position, rotation, preference=None):
    """可视化单帧数据"""
    # 解码图像
    vis_img = cv2.imdecode(np.frombuffer(rgb, np.uint8), cv2.IMREAD_COLOR)
    if vis_img is None:
        print("错误：图像解码失败")
        return None
    
    # 绘制预测轨迹
    draw_path_points(vis_img, pred_loc, color=(0, 255, 0))
    
    # 添加文本信息
    text_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 20
    
    cmd_names = ['LEFT','RIGHT','STRAIGHT','FOLLOW','CHANGE_L','CHANGE_R']
    info_texts = [
        f"Speed: {speed[0]:.2f} m/s",
        f"Control: steer={control[0]:.2f}, throttle={control[1]:.2f}, brake={control[2]:.2f}",
        f"Command: {cmd_names[int(cmd[0])]}",
        f"Position: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})",
        f"Rotation: ({rotation[0]:.1f}, {rotation[1]:.1f}, {rotation[2]:.1f})"
    ]
    
    if preference is not None:
        info_texts.append(f"Preference: {preference}")
    
    # 绘制半透明背景
    overlay = vis_img.copy()
    bg_height = (len(info_texts) + 1) * line_height
    cv2.rectangle(overlay, (0, 0), (400, bg_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
    
    # 绘制文本
    for i, text in enumerate(info_texts):
        position = (10, (i + 1) * line_height)
        cv2.putText(vis_img, text, position, font, font_scale, text_color, thickness)
    
    return vis_img

def visualize_data(data_path, output_dir=None, start_frame=0, end_frame=None, save_video=False):
    """可视化LMDB数据"""
    # 检查数据路径
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"错误：{data_path} 不存在")
        return
        
    # 直接读取LMDB文件
    print(f"正在可视化 {data_path}")
    visualize_lmdb_file(str(data_path), output_dir, start_frame, end_frame, save_video)

def visualize_lmdb_file(lmdb_path, output_dir=None, start_frame=0, end_frame=None, save_video=False):
    """可视化单个LMDB文件"""
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    with env.begin() as txn:
        length = int(txn.get('len'.encode()).decode())
        print(f"总帧数: {length}")
        
        if end_frame is None:
            end_frame = length
        end_frame = min(end_frame, length)
        
        if save_video:
            output_path = os.path.join(output_dir, 'visualization.mp4') if output_dir else 'visualization.mp4'
            first_frame = cv2.imdecode(np.frombuffer(txn.get(f'rgb_{0:05d}'.encode()), np.uint8), cv2.IMREAD_COLOR)
            h, w = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
        
        for i in range(start_frame, end_frame):
            # 读取数据
            encoded_image = txn.get(f'rgb_{i:05d}'.encode())
            if encoded_image:
                rgb = cv2.imdecode(
                    np.frombuffer(encoded_image, np.uint8),
                    cv2.IMREAD_COLOR
                )
            else:
                print(f"警告：未找到 rgb_{i:05d} 数据")
                continue
                
            # 打印数据大小以便调试
            print(f"Frame {i}: RGB数据大小 = {len(encoded_image)} bytes")
            
            try:
                # 读取其他数据
                pred_loc = np.frombuffer(txn.get(f'pred_loc_{i:05d}'.encode()), np.float32).reshape(-1, 2)
                control = np.frombuffer(txn.get(f'control_{i:05d}'.encode()), np.float32)
                speed = np.frombuffer(txn.get(f'speed_{i:05d}'.encode()), np.float32)
                cmd = np.frombuffer(txn.get(f'cmd_{i:05d}'.encode()), np.float32)
                position = np.frombuffer(txn.get(f'position_{i:05d}'.encode()), np.float32)
                rotation = np.frombuffer(txn.get(f'rotation_{i:05d}'.encode()), np.float32)
            except Exception as e:
                print(f"错误：读取帧 {i} 数据失败: {e}")
                continue
            
            # 可视化当前帧
            vis_img = visualize_frame(encoded_image, pred_loc, control, speed, cmd, position, rotation)
            if vis_img is None:
                print(f"警告：帧 {i} 可视化失败")
                continue
                
            if save_video:
                video_writer.write(vis_img)
            else:
                cv2.imshow('可视化', vis_img)
                key = cv2.waitKey(50)  # 50ms延迟
                if key == 27:  # ESC键退出
                    break
        
        if save_video:
            video_writer.release()
    
    env.close()
    cv2.destroyAllWindows()

class PreferenceLabeler:
    def __init__(self, data_path):
        """Initialize labeler"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        # Open LMDB environment
        self.env = lmdb.open(data_path, readonly=False, lock=False)
        
        with self.env.begin() as txn:
            # Get database length
            len_data = txn.get('len'.encode())
            if len_data is None:
                raise ValueError(f"Database is empty or invalid: {data_path}")
                
            self.length = int(len_data.decode())
            
            # Get last labeled position
            last_labeled = txn.get('last_labeled'.encode())
            self.current_idx = int(last_labeled.decode()) if last_labeled else 0
            
            print(f"Database info:")
            print(f"- Total frames: {self.length}")
            print(f"- Current position: {self.current_idx}")
    
    def load_frame(self, idx):
        """加载指定帧的数据"""
        with self.env.begin() as txn:
            rgb = txn.get(f'rgb_{idx:05d}'.encode())
            pred_loc = np.frombuffer(txn.get(f'pred_loc_{idx:05d}'.encode()), np.float32).reshape(-1, 2)
            world_loc = np.frombuffer(txn.get(f'world_loc_{idx:05d}'.encode()), np.float32).reshape(-1, 2)
            control = np.frombuffer(txn.get(f'control_{idx:05d}'.encode()), np.float32)
            speed = np.frombuffer(txn.get(f'speed_{idx:05d}'.encode()), np.float32)
            cmd = np.frombuffer(txn.get(f'cmd_{idx:05d}'.encode()), np.float32)
            position = np.frombuffer(txn.get(f'position_{idx:05d}'.encode()), np.float32)
            rotation = np.frombuffer(txn.get(f'rotation_{idx:05d}'.encode()), np.float32)
            
            # 尝试加载偏好值
            pref = txn.get(f'preference_{idx:05d}'.encode())
            preference = float(pref.decode()) if pref else None
            
            # 只返回visualize_frame需要的数据
            return {
                'rgb': rgb,
                'pred_loc': pred_loc,
                'control': control,
                'speed': speed,
                'cmd': cmd,
                'position': position,
                'rotation': rotation,
                'preference': preference
            }
    
    def save_preference(self, idx, preference):
        """保存偏好值"""
        with self.env.begin(write=True) as txn:
            txn.put(f'preference_{idx:05d}'.encode(), str(preference).encode())
            txn.put('last_labeled'.encode(), str(idx + 1).encode())
    
    def run(self):
        """Run labeling program"""
        window_name = 'Trajectory Preference Labeling (1-5: Score, N: Next, P: Previous, ESC: Exit)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        idx = self.current_idx
        while idx < self.length:
            frame_data = self.load_frame(idx)
            vis_img = visualize_frame(**frame_data)
            
            # Show operation hints
            cv2.putText(vis_img, 
                       "1-5: Score  N: Next  P: Previous  ESC: Exit", 
                       (10, vis_img.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, vis_img)
            
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                    preference = int(chr(key))
                    self.save_preference(idx, preference)
                    idx += 1
                    break
                elif key == ord('n'):  # Next frame
                    idx += 1
                    break
                elif key == ord('p') and idx > 0:  # Previous frame
                    idx -= 1
                    break
                elif key == 27:  # ESC
                    cv2.destroyAllWindows()
                    return
    
    def close(self):
        """Close database"""
        self.env.close()

def main():
    parser = argparse.ArgumentParser(description='LBC Trajectory Preference Labeling Tool')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data')
    parser.add_argument('--mode', choices=['label', 'visualize'], default='label', 
                      help='Run mode: label or visualize')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for visualization')
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame')
    parser.add_argument('--end_frame', type=int, default=None, help='End frame')
    parser.add_argument('--save_video', action='store_true', help='Save as video')
    
    args = parser.parse_args()
    
    if args.mode == 'label':
        labeler = PreferenceLabeler(args.data_path)
        try:
            labeler.run()
        finally:
            labeler.close()
    else:
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        visualize_data(args.data_path, args.output_dir, args.start_frame, args.end_frame, args.save_video)

if __name__ == '__main__':
    main() 