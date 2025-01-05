import os
import cv2
import lmdb
import numpy as np
import argparse
from pathlib import Path

def draw_path_points(img, points, color=(0, 255, 0), thickness=2):
    """在图像上绘制路径点"""
    for i in range(len(points)-1):
        pt1 = (int(points[i][0]), int(points[i][1]))
        pt2 = (int(points[i+1][0]), int(points[i+1][1]))
        cv2.line(img, pt1, pt2, color, thickness)
        cv2.circle(img, pt1, 2, color, -1)
    if len(points) > 0:
        cv2.circle(img, (int(points[-1][0]), int(points[-1][1])), 2, color, -1)

def visualize_frame(rgb, pred_loc, world_loc, control, speed, cmd, position, rotation, index=None):
    """可视化单帧数据"""
    vis_img = rgb.copy()
    draw_path_points(vis_img, pred_loc, color=(0, 255, 0))
    
    # 添加文本信息
    text_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 20
    
    info_texts = [
        f"Frame: {index}" if index is not None else "",
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
    
    for i, text in enumerate(info_texts):
        position = (10, (i + 1) * line_height)
        cv2.putText(vis_img, text, position, font, font_scale, text_color, thickness)
    
    return vis_img

class DatasetComparator:
    def __init__(self, lmdb_path1, lmdb_path2, preference_path):
        self.env1 = lmdb.open(lmdb_path1, readonly=True, lock=False)
        self.env2 = lmdb.open(lmdb_path2, readonly=True, lock=False)
        self.pref_env = lmdb.open(preference_path, map_size=1099511627776)
        
        with self.env1.begin() as txn1, self.env2.begin() as txn2:
            self.length = min(
                int(txn1.get('len'.encode()).decode()),
                int(txn2.get('len'.encode()).decode())
            )
    
    def load_frame(self, txn, idx):
        """从LMDB加载单帧数据"""
        return {
            'rgb': cv2.imdecode(np.frombuffer(txn.get(f'rgb_{idx:05d}'.encode()), np.uint8), cv2.IMREAD_COLOR),
            'pred_loc': np.frombuffer(txn.get(f'pred_loc_{idx:05d}'.encode()), np.float32).reshape(-1, 2),
            'world_loc': np.frombuffer(txn.get(f'world_loc_{idx:05d}'.encode()), np.float32).reshape(-1, 2),
            'control': np.frombuffer(txn.get(f'control_{idx:05d}'.encode()), np.float32),
            'speed': np.frombuffer(txn.get(f'speed_{idx:05d}'.encode()), np.float32),
            'cmd': np.frombuffer(txn.get(f'cmd_{idx:05d}'.encode()), np.float32),
            'position': np.frombuffer(txn.get(f'position_{idx:05d}'.encode()), np.float32),
            'rotation': np.frombuffer(txn.get(f'rotation_{idx:05d}'.encode()), np.float32)
        }
    
    def get_last_compared_frame(self):
        """获取上次比较到的帧索引"""
        with self.pref_env.begin() as txn:
            last_frame = txn.get('last_compared'.encode())
            return int(last_frame.decode()) if last_frame else 0
    
    def save_preference(self, frame_idx, preference):
        """保存偏好结果"""
        with self.pref_env.begin(write=True) as txn:
            txn.put(f'prefer_{frame_idx:05d}'.encode(), preference.tobytes())
            txn.put('last_compared'.encode(), str(frame_idx + 1).encode())
    
    def compare_frames(self):
        """交互式比较两个数据集的帧"""
        start_frame = self.get_last_compared_frame()
        print(f"从第 {start_frame} 帧开始比较")
        
        window_name = 'Dataset Comparison (1: Left better, 2: Right better, ESC: Exit)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        for idx in range(start_frame, self.length):
            with self.env1.begin() as txn1, self.env2.begin() as txn2:
                frame1 = self.load_frame(txn1, idx)
                frame2 = self.load_frame(txn2, idx)
                
                vis1 = visualize_frame(**frame1, index=idx)
                vis2 = visualize_frame(**frame2, index=idx)
                
                # 水平拼接两个图像
                combined = np.hstack((vis1, vis2))
                
                # 添加操作提示
                cv2.putText(combined, 
                           "1: Left better  2: Right better  ESC: Exit", 
                           (10, combined.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(window_name, combined)
                
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('1'):
                        self.save_preference(idx, np.array([1, 0], dtype=np.int8))
                        break
                    elif key == ord('2'):
                        self.save_preference(idx, np.array([0, 1], dtype=np.int8))
                        break
                    elif key == 27:  # ESC
                        cv2.destroyAllWindows()
                        return
    
    def close(self):
        """关闭所有LMDB环境"""
        self.env1.close()
        self.env2.close()
        self.pref_env.close()

def main():
    parser = argparse.ArgumentParser(description='比较两个LBC数据集并记录偏好')
    parser.add_argument('--dataset1', type=str, required=True, help='第一个数据集的LMDB路径')
    parser.add_argument('--dataset2', type=str, required=True, help='第二个数据集的LMDB路径')
    parser.add_argument('--output', type=str, required=True, help='偏好数据保存路径')
    
    args = parser.parse_args()
    
    comparator = DatasetComparator(args.dataset1, args.dataset2, args.output)
    try:
        comparator.compare_frames()
    finally:
        comparator.close()

if __name__ == '__main__':
    main() 