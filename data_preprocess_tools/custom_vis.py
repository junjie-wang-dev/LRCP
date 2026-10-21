from matplotlib import pyplot as plt
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from io import BytesIO
import cv2
from matplotlib import cm
import numpy as np

from opencood.tools.inference_utils import get_cav_box
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev

def vis_offset_maps(offset_maps, points_masks, lidars, scale=8):
    BEV_images = [convert_lidar_to_BEV_image(lidar).canvas for lidar in lidars]

    for offset_map, mask, img in zip(offset_maps, points_masks, BEV_images):
        for i in range(offset_map.shape[0]):  # i是纵坐标y
            for j in range(offset_map.shape[1]): # j是横坐标x
                if not mask[i][j]:
                    x = (j + offset_map[i][j][0]) * scale
                    y = (i + offset_map[i][j][1]) * scale
                    cv2.circle(img, (int(x), int(y)), 1, (0,0,255), -1)
    return BEV_images

def summ_traj2ds_on_rgbs(trajs, rgbs, valids=None, frame_ids=None, only_return=False, show_dots=True, cmap='coolwarm', linewidth=1, bev_shape=[100, 252], scale=8):
    # trajs: hash_points: {"原始点Hash+帧": 点}
    # rgbs is [ H, W, C], a list
    # 原先 trajs is S, N, 2
    # rgbs is S, C, H, W

    S = len(rgbs)
  
    for i in range(bev_shape[0]):
        for j in range(bev_shape[1]):
            if str(j) + '_' + str(i) + '_0' in trajs:
                traj_one_point = []
                traj_one_point.append( (trajs[str(j) + '_' + str(i) + '_0'] * scale).astype(int) )
                for k in range(6): # 最多11帧
                    if str(j) + '_' + str(i) + '_' + str(k+1) in trajs:
                        traj_one_point.append( (trajs[str(j) + '_' + str(i) + '_' +str(k+1)] * scale).astype(int) )
                    else:
                        break
                traj_one_point = np.vstack(traj_one_point)
                for t in range(len(traj_one_point)):
                    # traj[:t+1]是一个点了
                    rgbs[t] = draw_traj_on_image_py(rgbs[t], traj_one_point[:t+1], S=S, show_dots=show_dots, cmap='spring', linewidth=linewidth)

    rgbs_p = [Image.fromarray(rgb) for rgb in rgbs]
    return rgbs_p

def draw_traj_on_image_py(rgb, traj, S=5, linewidth=1, show_dots=False, cmap='coolwarm', maxdist=None):
    # all inputs are numpy tensors
    # rgb is 3 x H x W
    # traj is S x 2; 绝对的坐标
    
    H, W, C = rgb.shape
    assert(C==3)

    rgb = rgb.astype(np.uint8).copy()

    S1, D = traj.shape
    assert(D==2)

    color_map = cm.get_cmap(cmap)
    S1, D = traj.shape

    for s in range(S1-1):
        if maxdist is not None:
            val = (np.sqrt(np.sum((traj[s]-traj[0])**2))/maxdist).clip(0,1)
            color = np.array(color_map(val)[:3]) * 255 # rgb
        else:
            color = np.array(color_map((s)/max(1,float(S-2)))[:3]) * 255 # rgb

        cv2.line(rgb,
                    (int(traj[s,0]), int(traj[s,1])),
                    (int(traj[s+1,0]), int(traj[s+1,1])),
                    color,
                    linewidth,
                    cv2.LINE_AA)
        if show_dots:
            cv2.circle(rgb, (traj[s,0], traj[s,1]), linewidth, color, -1)

    if maxdist is not None:
        val = (np.sqrt(np.sum((traj[-1]-traj[0])**2))/maxdist).clip(0,1)
        color = np.array(color_map(val)[:3]) * 255 # rgb
    else:
        # draw the endpoint of traj, using the next color (which may be the last color)
        color = np.array(color_map((S1-1)/max(1,float(S-2)))[:3]) * 255 # rgb
        
    # color = np.array(color_map(1.0)[:3]) * 255
    cv2.circle(rgb, (traj[-1,0], traj[-1,1]), linewidth*2, color, -1)
    return rgb


def convert_lidar_to_BEV_image(pcd_np, pc_range=[-100.8, -40, -3.5, 100.8, 40, 1.5], bev_shape=[100, 252], scale=8):
    # 将lidar图可视化到一个二维
    canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=(bev_shape[0]*scale, bev_shape[1]*scale),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=True)  # 一律全部按照右手坐标来
    canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np) # Get Canvas Coords
    canvas.draw_canvas_points(canvas_xy[valid_mask]) # Only draw valid points
    return canvas


def visualize(infer_result, pcd, pc_range, save_path, method='3d', left_hand=False):
        """
        Visualize the prediction, ground truth with point cloud together.
        They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

        Parameters
        ----------
        infer_result:
            pred_box_tensor : torch.Tensor
                (N, 8, 3) prediction.

            gt_tensor : torch.Tensor
                (N, 8, 3) groundtruth bbx
            
            uncertainty_tensor : optional, torch.Tensor
                (N, ?)

            lidar_agent_record: optional, torch.Tensor
                (N_agnet, )


        pcd : torch.Tensor
            PointCloud, (N, 4).

        pc_range : list
            [xmin, ymin, zmin, xmax, ymax, zmax]

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        method: str, 'bev' or '3d'

        """
        plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40])
        pc_range = [int(i) for i in pc_range]
        pcd_np = pcd.cpu().numpy()

        pred_box_tensor = infer_result.get("pred_box_tensor", None)
        gt_box_tensor = infer_result.get("gt_box_tensor", None)

        if pred_box_tensor is not None:
            pred_box_np = pred_box_tensor.cpu().numpy()
            pred_name = ['pred'] * pred_box_np.shape[0]

            score = infer_result.get("score_tensor", None)
            if score is not None:
                score_np = score.cpu().numpy()
                pred_name = [f'score:{score_np[i]:.3f}' for i in range(score_np.shape[0])]

            uncertainty = infer_result.get("uncertainty_tensor", None)
            if uncertainty is not None:
                uncertainty_np = uncertainty.cpu().numpy()
                uncertainty_np = np.exp(uncertainty_np)
                d_a_square = 1.6**2 + 3.9**2
                
                if uncertainty_np.shape[1] == 3:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) 
                    # yaw angle is in radian, it's the same in g2o SE2's setting.

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:.3f} a_u:{uncertainty_np[i,2]:.3f}' \
                                    for i in range(uncertainty_np.shape[0])]

                elif uncertainty_np.shape[1] == 2:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f}' \
                                    for i in range(uncertainty_np.shape[0])]

                elif uncertainty_np.shape[1] == 7:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f} a_u:{uncertainty_np[i,6]:3f}' \
                                    for i in range(uncertainty_np.shape[0])]                    

        if gt_box_tensor is not None:
            gt_box_np = gt_box_tensor.cpu().numpy()
            gt_name = [''] * gt_box_np.shape[0]

        if method == 'bev':
            canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=left_hand) 

            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np) # Get Canvas Coords
            canvas.draw_canvas_points(canvas_xy[valid_mask]) # Only draw valid points
            if gt_box_tensor is not None:
                canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
            if pred_box_tensor is not None:
                canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)

            # heterogeneous
            lidar_agent_record = infer_result.get("lidar_agent_record", None)
            cav_box_np = infer_result.get("cav_box_np", None)
            if lidar_agent_record is not None:
                cav_box_np = copy.deepcopy(cav_box_np)
                for i, islidar in enumerate(lidar_agent_record):
                    text = ['lidar'] if islidar else ['camera']
                    color = (0,191,255) if islidar else (255,185,15)
                    canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=text)

        elif method == '3d':
            canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
            canvas.draw_canvas_points(canvas_xy[valid_mask])
            if gt_box_tensor is not None:
                canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
            if pred_box_tensor is not None:
                canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)

            # heterogeneous
            lidar_agent_record = infer_result.get("lidar_agent_record", None)
            cav_box_np = infer_result.get("cav_box_np", None)
            if lidar_agent_record is not None:
                cav_box_np = copy.deepcopy(cav_box_np)
                for i, islidar in enumerate(lidar_agent_record):
                    text = ['lidar'] if islidar else ['camera']
                    color = (0,191,255) if islidar else (255,185,15)
                    canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=text)

        else:
            raise(f"Not Completed for f{method} visualization.")

        plt.axis("off")

        plt.imshow(canvas.canvas)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, transparent=False, dpi=500)
        plt.clf()
        plt.close()

        # return Image.open(BytesIO(canvas.canvas.tostring_rgb()))
        return Image.fromarray(canvas.canvas)
