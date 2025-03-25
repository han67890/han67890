import pickle
from My_radar import EKF
from My_radar import Gtrack_visualize
import numpy as np

if __name__ == '__main__':
    # 初始化 EKF 跟踪器
    tracker = EKF.EKF()

    # 加载雷达数据
    file_name = "E:\\data_radar\\1TX4RX_20241127.pkl"  # 替换为实际路径
    with open(file_name, 'rb') as f:
        adc_matrix = np.load(f)  # 加载 .pkl 文件中的数据

    # adc_matrix 示例维度: [Frames, Chirps, Samples, Antennas]

    # 初始化可视化工具
    Gtrack_visualize.create()

    # 遍历每一帧数据
    for frame_idx, frame_data in enumerate(adc_matrix):
        try:
            # 假设点云数据在 frame_data 中解析
            pc = frame_data['pointCloud2D']
            ranges = pc['range']        # 距离信息
            azimuths = pc['azimuth']    # 方位角信息
            dopplers = pc['doppler']    # 多普勒频移
            snrs = pc['snr']            # 信噪比

            # 初始化可视化帧
            frame = Gtrack_visualize.get_empty_frame()

            # 更新点云到跟踪器
            tracker.update_point_cloud(ranges, azimuths, dopplers, snrs)

            # 获取跟踪结果
            targetDescr, tNum = tracker.step()

            # 更新可视化帧
            frame = Gtrack_visualize.update_frame(targetDescr, int(tNum[0]), frame)
            frame = Gtrack_visualize.draw_points(tracker.point_cloud, len(ranges), frame)

            # 显示可视化结果
            if not Gtrack_visualize.show(frame, wait=10):  # 模拟实时帧率
                break

        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            continue

    # 结束可视化
    Gtrack_visualize.destroy()