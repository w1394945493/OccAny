import argparse
import os
os.environ["TMPDIR"] = "/vepfs-mlp2/c20250502/haoce/wangyushen/tmp" #
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from PIL import Image
import pickle


REPO_ROOT = Path(__file__).resolve().parent
VENDORED_IMPORT_PATHS = [
    REPO_ROOT / "third_party",
    REPO_ROOT / "third_party" / "dust3r",
    REPO_ROOT / "third_party" / "croco" / "models" / "curope",
    REPO_ROOT / "third_party" / "Grounded-SAM-2",
    REPO_ROOT / "third_party" / "Grounded-SAM-2" / "grounding_dino",
    REPO_ROOT / "third_party" / "sam3",
    REPO_ROOT / "third_party" / "Depth-Anything-3" / "src",
]
for vendored_path in reversed(VENDORED_IMPORT_PATHS):
    vendored_path_str = str(vendored_path)
    if vendored_path.exists() and vendored_path_str not in sys.path:
        sys.path.insert(0, vendored_path_str)


from occany.utils.inference_helper import (
    build_demo_reconstruction_views,
    extract_demo_rgb_images,
    populate_demo_sam2_box_dicts,
    build_intrinsics_from_focal,

)
from occany.utils.resolution import get_output_resolution

from occany.model.model_must3r import Must3r, Dust3rEncoder, RaymapEncoderDiT, Must3rDecoder  # noqa: F401
from occany.model.must3r_blocks.head import ActivationType
from occany.must3r_inference import inference_occany_gen

from occany.semantic_inference import infer_semantic_from_boxes_and_sam2_feat_list
from occany.utils.helpers import (
    transform_points_torch, 
    create_voxel_prediction,
    apply_majority_pooling,
)

from occany.utils.image_util import convert_images_to_uint8_hwc
def get_output_resolution_from_image(image_path: str, model_family: str) -> Tuple[int, int]:
    with Image.open(image_path) as image:
        return get_output_resolution(image.size, model_family=model_family) # image.size:[1226 370] model_family: 'must3r'

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./demo_data/output',
        help="Directory where inference outputs and visualizations are written",
    )
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the RGB demo directory containing frame folders",
        default='./demo_data/input',
    )
    parser.add_argument('--frame_interval', type=int, default=5, help='Frame interval for video processing')
    parser.add_argument(
        '--model',
        type=str,
        default='occany_da3',
        choices=['occany_must3r', 'occany_da3'],
        help='Model to use',
    )
    parser.add_argument('--gen', action='store_true', default=False, help='Predict raymap')
    
    parser.add_argument(
        '--semantic', "-sem",
        type=str,
        choices=['pretrained@SAM2_small',
                 'distill@SAM2_small',
                 'distill@SAM2_base',
                 'distill@SAM2_large',
                 'pretrained@SAM2_base',
                 'pretrained@SAM2_large',
                 'pretrained@SAM3',
                 'distill@SAM3'],
        default=None,
        help='Semantic processing option. Choices: pretrained@SAM_small, distill@SAM_tiny, distill@SAM_small.'
    )
    parser.add_argument('--compute_segmentation_masks', action='store_true', default=False,
                        help='Compute segmentation masks')
    parser.add_argument('--sam3_conf_th', type=float, default=0.15,
                        help='Confidence threshold for SAM3 semantic inference')
    parser.add_argument('--sam3_resolution', type=int, default=1008,
                        help='Resolution for SAM3 model')
    parser.add_argument('--view_batch_size', type=int, default=4,
                        help='Number of views per SAM3 inference chunk (lower uses less GPU memory)')
    parser.add_argument('--key_to_get_pts3d', type=str, default='pts3d',
                        help='Key to get pts3d from the output')
    parser.add_argument('--views_per_interval', '-vpi', type=int, default=2,
                        help='Number of views per interval for inference')
    parser.add_argument('--gen_rotate_novel_poses_angle', '-rot', type=int, default=0,
                        help='Angle to rotate novel poses')
    parser.add_argument('--gen_forward_novel_poses_dist', '-fwd', type=int, default=1,
                        help='Distance to move forward for novel poses (in meters)')
    parser.add_argument('--num_seed_rotations', '-nseed', type=int, default=0,
                        help='Number of seed rotations to generate (e.g., 5 for [-10, -5, 0, 5, 10]). If 0, uses standard mode.')
    parser.add_argument('--seed_rotation_angle', '-seed_rot', type=int, default=None,
                        help='Angle in degrees between seed rotations. If None, defaults to 15.0 degrees.')
    parser.add_argument('--seed_translation_distance', '-seed_trans', type=int, default=None,
                        help='Distance in meters to translate seed poses laterally. Positive rotations translate right, negative translate left.')
    parser.add_argument('--batch_gen_view', '-bs_gen', type=int, default=4,
                        help='Number of generated views per batch')
    parser.add_argument('--no_semantic_from_rotated_views', "-nsr", action='store_true', default=False,
                        help='Disable using semantics from rotated views (only use semantics from straight/forward views)')
    parser.add_argument('--box_conf_thres', type=float, default=0.05,
                        help='Confidence threshold for bounding box filtering')
    parser.add_argument('--merge_masks', action='store_true', default=False,
                        help='Merge masks by label and binned confidence (0.01 bins)')
    parser.add_argument('--only_semantic_from_recon_view', "-osr", action='store_true', default=False,
                        help='Use only semantic information from the reconstruction view (exclude all generated views)')
    parser.add_argument(
        '--gen_semantic_from_distill_sam3',
        action='store_true',
        default=False,
        help='For pretrained@SAM3, infer generated-view semantics from distilled SAM3 features when available',
    )
    parser.add_argument('--apply_majority_pooling', action='store_true', default=False,
                        help='Apply majority pooling to voxel predictions (3x3x3 neighborhood)')
    
    parser.add_argument('--pose_from_depth_ray', action='store_true', default=False,
                        help='Use ray pose estimation (set to True for trained models that use ray pose)')
    parser.add_argument('--point_from_depth_and_pose', action='store_true', default=False,
                        help='Compute pointmap from depth, intrinsics and c2w')
    parser.add_argument('--recon_conf_thres', type=float, default=2.0,
                        help='Reconstruction confidence threshold.')
    parser.add_argument('--gen_conf_thres', type=float, default=6.0,
                        help='Generation confidence threshold.')
    

    return parser





CLASS_NAMES = [
            "other",
            "barrier",
            "bicycle",
            "bus",
            "car",
            "construction_vehicle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "trailer",
            "truck",
            "driveable_surface",
            "other_flat",
            "sidewalk",
            "terrain",
            "manmade",
            "vegetation",
            "free",
        ]
empty_class = 17
other_class = 0
n_classes = 18

weights_path = Path('/vepfs-mlp2/c20250502/haoce/wangyushen/OccAny/checkpoints/occany.pth')

if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()

    def maybe_apply_pooling(voxel_pred_np: np.ndarray) -> np.ndarray:
        if not args.apply_majority_pooling:
            return voxel_pred_np

        return apply_majority_pooling(
            voxel_pred_np,
            n_classes=n_classes,
            other_class=other_class,
            empty_class=empty_class,
            is_geometry_only=False,
        )
    model_family = "must3r"   
    semantic_family = "SAM2"
    sam_model_for_inference = "SAM2"

    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    checkpoint_args = checkpoint['args']
    # ================================================#
    # 3D重建
    # encoder
    encoder = Dust3rEncoder() # Encoder: Dust3r 编码器
    # decoder
    decoder = Must3rDecoder(img_size=(512, 512), 
                            enc_embed_dim=1024, 
                            embed_dim=768,                
                            pointmaps_activation=ActivationType.LINEAR, 
                            pred_sam_features=True,                
                            feedback_type='single_mlp', 
                            memory_mode='kv', 
                            ray_map_encoder_depth=6, 
                            use_multitask_token=True)
    decoder.pointmaps_activation = ActivationType.LINEAR

    # ================================================#
    # NVR
    # encoder
    raymap_encoder = RaymapEncoderDiT(
        use_time_cond=False,    
        use_raymap_only_conditioning=False, # False
        projection_features='pts3d_local,pts3d,rgb,conf,sam',
    )   
    # decodr
    gen_decoder = Must3rDecoder(img_size=(512, 512), 
                            enc_embed_dim=1024, 
                            embed_dim=768,                
                            pointmaps_activation=ActivationType.LINEAR, 
                            pred_sam_features=True,                
                            feedback_type='single_mlp', 
                            memory_mode='kv', 
                            ray_map_encoder_depth=6, 
                            use_multitask_token=True)
    
    gen_decoder.pointmaps_activation = ActivationType.LINEAR
    gen_decoder.eval()

    encoder.load_state_dict(checkpoint['encoder'], strict=False)
    decoder.load_state_dict(checkpoint['decoder'], strict=False)
    raymap_encoder.load_state_dict(checkpoint['raymap_encoder'], strict=False)
    gen_decoder.load_state_dict(checkpoint['gen_decoder'], strict=False)

    encoder.eval()
    decoder.eval()
    raymap_encoder.eval()
    gen_decoder.eval()

    encoder.to(args.device)
    decoder.to(args.device)
    raymap_encoder.to(args.device)
    gen_decoder.to(args.device)
    

    recon_conf_thres = args.recon_conf_thres        # 2.0
    gen_conf_thres = args.gen_conf_thres            # 2.0
    # ================================================#
    voxel_size = 0.4
    occ_size = [200, 200, 24]
    voxel_origin = torch.tensor([-40.0, -40.0, -3.6], device=args.device, dtype=torch.float32)
    
    item_count = 0
    T_cam_to_voxel = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=args.device,
    )

    
    frame_dirs = [
        Path('/vepfs-mlp2/c20250502/haoce/wangyushen/OccAny/demo_data/input/kitti_08_1390'), 
        Path('/vepfs-mlp2/c20250502/haoce/wangyushen/OccAny/demo_data/input/nuscenes_scene-0039')
    ]
    
    for frame_dir in tqdm(frame_dirs, desc=f"Processing RGB demo frames"):  # frame_dirs: kitti, nuscenes
        demo_image_paths, demo_frame_id = extract_demo_rgb_images(str(frame_dir))
        frame_output_resolution = get_output_resolution_from_image(
            demo_image_paths[0],
            model_family=model_family,
        )   
        recon_views = build_demo_reconstruction_views(
            image_paths=demo_image_paths,
            output_resolution=frame_output_resolution,  # 160x512
            model_family=model_family,                  # must3r
            semantic_family=semantic_family,            # SAM2
            frame_interval=args.frame_interval,         # 5
            sam3_resolution=args.sam3_resolution,       # 1008
            device=args.device,
        )
        data = {
            "frame_id": [demo_frame_id],
        }
        B = recon_views[0]["img"].shape[0]
        _, C, H, W = recon_views[0]["img"].shape    # (1 3 160 512)
        # ======================================================#
        # 1. GroundingDino预测框prompt：
        # GroundingDino：使用GroundingDino对真实输入视角进行开放词汇检测，为每个图生成SAM2语义分割和类别信息，作为后续SAM2语义分割的提示
        box_summary = populate_demo_sam2_box_dicts(
            recon_views=recon_views, # 返回给 recon_views[0]['box_dict']
            class_names=CLASS_NAMES,
            device=args.device,
        )
        
        with torch.inference_mode():
            x_ray = None
            sam_feats = None
            sam_feats_raymap = None
            recon_2_gen_mapping = None
            generated_output_c2w = None
            # =================================================#
            # 2. 场景重建+新视角渲染：场景重建：预测pointmap，sam2特征和相机位姿；新视角渲染：根据新视角位姿，预测pointmap和sam2特征(不再预测相机位姿)
            # 2.1 场景重建：输入多视角RGB图像，输入多视角RGB图像，通过must3R encoder-decoder预测pointmap，相机位姿，点云置信度，并额外提取SAM2特征
            # 2.2 新视角生成：根据预测相机轨迹生成虚拟相机位姿，并通过轻量化encoder和must3R式decoder预测新视角pointmap，点云置信度和SAM2特征
            img_out, raymap_out, x_ray, sam_feats, sam_feats_raymap, recon_2_gen_mapping = inference_occany_gen(
                recon_views,
                None,
                raymap_encoder,     # RaymapEncoderDiT
                encoder,            # Dust3rEncoder
                decoder,            # Must3rDecoder
                gen_decoder,        # Must3rDecoder
                decoder.pointmaps_activation,
                args.device,
                gen_rotate_novel_poses_angle=args.gen_rotate_novel_poses_angle, # 30
                gen_novel_poses=args.gen,                                       # True
                pred_raymap=args.gen,                                           # True
                views_per_interval=args.views_per_interval,                     # 2
                gen_forward_novel_poses_dist=args.gen_forward_novel_poses_dist, # 5
                num_seed_rotations=args.num_seed_rotations,                     # 0
                seed_rotation_angle=args.seed_rotation_angle,
                seed_translation_distance=args.seed_translation_distance,       # 2
                use_local_points_with_pose_as_pts3d=False,
                use_raymap_only_conditioning=False,                             # False
                raymap_batch_size=args.batch_gen_view,                          # 2
                key_to_get_pts3d=args.key_to_get_pts3d,             
                dtype=torch.float32,
                sam_model=sam_model_for_inference,                              # 'SAM2'
            )   # 重建 pts(1 5 160 512 3) 渲染 pts(1 30 160 512 3) x_ray(1 2 320 1024) 重建 sam(1 5 256 32 32) 渲染 sam(1 30 256 32 32)
            
            sam_feats_img_and_raymap = None
            sam3_recon_distill_feats = sam_feats[:3] if sam_feats is not None else None
            sam3_gen_distill_feats = sam_feats_raymap[:3] if sam_feats_raymap is not None else None
            
            if sam_feats is not None and sam_feats_raymap is not None:
                sam_feats_img_and_raymap = [
                    torch.cat([sam_feats[level_idx], sam_feats_raymap[level_idx]], dim=1)
                    for level_idx in range(min(len(sam_feats), len(sam_feats_raymap)))
                ]   # 将重建和新视图渲染阶段预测的sam特征拼接
        
        res = img_out
        imgs = [v['img'] for v in recon_views]
        imgs = torch.stack(imgs, dim=1) # (1 5 3 160 512)        

        recon_semantic_2ds = None
        gen_semantic_2ds = None
        sam2_feats_batch = []
        
        # =========================================================#
        # 3. 2D语义标注生成：给原始重建视角 + 生成新视角
        # 输入GroundingDINO检测框，重建视角/新视角渲染的SAM feature预测每个像素对应语义类别ID
        # 'distill@SAM2_large'
        if args.semantic is not None:
            feat_src = 'distill'
            n_recon_views = len(recon_views) # 5
            n_gen_views = 0 if raymap_out is None else raymap_out['pts3d'].shape[1] # 30
            n_recon_and_gen_views = n_recon_views + n_gen_views # 5+30=35

            semantic_fill_value = empty_class
            other_class =  other_class

            semantic_2ds = torch.full(
                (B, n_recon_and_gen_views, H, W),
                semantic_fill_value,
                dtype=torch.uint8,
            )   # (1 35 160 512) 创建semantic存储，每个像素存类别ID
            if semantic_family == "SAM2":
                sam2_model_type = 'SAM2_large'
                sam2_imgs_recon = None

                class_names = CLASS_NAMES
                class2idx = {name: idx for idx, name in enumerate(class_names)}
                ignore_ids = {empty_class, other_class, 255}
                # 遍历batch
                for batch_i in range(B):
                    # 取SAM feature
                    sam2_feats = {
                        "image_embed": sam_feats_img_and_raymap[0][batch_i],    # (35 256 32 32)
                        "high_res_feats": [
                            sam_feats_img_and_raymap[2][batch_i],               # (35 32 128 128)
                            sam_feats_img_and_raymap[1][batch_i],               # (35 64 64 64)
                        ],
                    }
                    sam2_feats_batch.append(sam2_feats)
                    # 对真实视角做检测框语义
                    for recon_view_i in range(n_recon_views):
                        box_dict = recon_views[recon_view_i]['box_dict'][batch_i] # 取检测框：
                        boxes = box_dict['boxes']               # (35 4)
                        confidences = box_dict['confidences']   # (35)
                        labels = box_dict['labels']

                        valid_indices = [idx for idx, label in enumerate(labels) if label in class2idx] # 过滤无效类别
                        if len(valid_indices) == 0:
                            continue

                        boxes_np = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
                        conf_np = (
                            confidences.detach().cpu().numpy()
                            if torch.is_tensor(confidences)
                            else np.asarray(confidences)
                        )
                        if boxes_np.size == 0:
                            continue
                        boxes_np = boxes_np.reshape(-1, 4)[valid_indices]
                        conf_np = conf_np.reshape(-1)[valid_indices]
                        label_ids = [class2idx[labels[idx]] for idx in valid_indices]
                        # 获取对应的新视角
                        corresponding_gen_view_ids = [
                            view_idx + n_recon_views
                            for view_idx in recon_2_gen_mapping[recon_view_i]
                        ]
                        for gen_view_i in range(
                            0,
                            max(1, len(corresponding_gen_view_ids)),
                            args.batch_gen_view, # 2
                        ):
                            recon_and_gen_ids = [recon_view_i] + corresponding_gen_view_ids[
                                gen_view_i:gen_view_i + args.batch_gen_view 
                            ]   # 每次处理三个视角的sam语义图

                            sam2_feat_list = [] # 准备sam2输入feature
                            for view_id in recon_and_gen_ids:
                                sam2_feat_list.append(
                                    {
                                        "high_res_feats": [
                                            sam2_feats['high_res_feats'][0][view_id:view_id + 1],
                                            sam2_feats['high_res_feats'][1][view_id:view_id + 1],
                                        ],
                                        "image_embed": sam2_feats['image_embed'][view_id:view_id + 1],
                                    }
                                )
                            # bbox使用原始视角的bbox
                            sem2d = infer_semantic_from_boxes_and_sam2_feat_list(
                                sam2_model_type,
                                H,
                                W,
                                label_ids,      # (35)
                                ignore_ids,
                                boxes_np,       # (35 4) 提示框来自于原始视角，新视角的精确位置依赖于SAM2 feature propagation 论文3.3 利用sam2的视频跟踪能力，将语义mask传播至整个场景
                                conf_np,        
                                other_class=other_class,
                                empty_class=empty_class,
                                use_sam_video=True,
                                sam2_feats_list=sam2_feat_list,
                                poses=None,
                                focals=None,
                                depth_maps=None,
                                device=args.device,
                                box_conf_thres=args.box_conf_thres,
                                merge_masks=args.merge_masks,
                            )   # (3 160 512) 每次处理3个视角

                            for local_idx, view_i in enumerate(recon_and_gen_ids):
                                semantic_2ds[batch_i, view_i] = torch.from_numpy(sem2d[local_idx])

                            del sam2_feat_list, sem2d
                            torch.cuda.empty_cache()
                
                recon_semantic_2ds = semantic_2ds[:, :n_recon_views]
                gen_semantic_2ds = semantic_2ds[:, n_recon_views:] if n_gen_views > 0 else None
        
        # =====================================================#
        # 4. pointmap结果整理              
        outputs = {}
        pts3d_render = res[args.key_to_get_pts3d]   # (1 5 160 512 3) 3D点图全局坐标
        pts3d_local_render = res['pts3d_local']     # (1 5 160 512 3)
        conf_render = res['conf']                   # (1 5 160 512)
        outputs["render"] = {                       # 重建结果
            "pts3d": pts3d_render,                  # (1 5 160 512 3)
            "pts3d_local": pts3d_local_render,      # (1 5 160 512 3)
            "conf": conf_render,                    # (1 5 160 512)
            "colors": imgs,                         # (1 5 3 160 512)
            "focal": res['focal'],                  # (1 5)
            "c2w": res['c2w'],                      # (1 5 4 4)
            "estimated_camera_poses": res['c2w_pose'] if 'c2w_pose' in res else res['c2w'], # (1 5 4 4)
            "semantic_2ds": recon_semantic_2ds,     # (1 5 160 512)
            "is_recon": torch.ones(B, pts3d_render.shape[1], dtype=torch.bool, device=pts3d_render.device), # (1 5) 全为True
            # "c2w_pose": res['c2w_pose']
        }
        # 新视图渲染结果
        pts3d_gen = raymap_out[args.key_to_get_pts3d]   # (1 30 160 512 3)
        pts3d_local_gen = raymap_out['pts3d_local']     # (1 30 160 512 3)
        conf_gen = raymap_out['conf']                   # (1 30 160 512)
        render_gen_c2w = generated_output_c2w
        if render_gen_c2w is None:
            render_gen_c2w = raymap_out.get('c2w_input')    # (1 30 4 4) 渲染阶段不再预测相机位姿，直接使用虚拟相机位姿
        if render_gen_c2w is None:
            render_gen_c2w = raymap_out['c2w']

        outputs["render_gen"] = {
            "pts3d": pts3d_gen,                 # (1 30 160 512 3)
            "pts3d_local": pts3d_local_gen,     # (1 30 160 512 3)
            "conf": conf_gen,                   # (1 30 160 512)
            "colors": torch.zeros(B, pts3d_gen.shape[1], 3, H, W, device=pts3d_gen.device),
            "focal": raymap_out['focal'],       # (1 30)
            "c2w": render_gen_c2w,              # (1 30 4 4)
            "semantic_2ds": gen_semantic_2ds,   # (1 30 160 512)
            "is_recon": torch.zeros(B, pts3d_gen.shape[1], dtype=torch.bool, device=pts3d_gen.device),
            # "c2w_pose": gen_out['c2w_pose']
        }
        # 将重建 和 新视角渲染 两部分结果整合
        outputs['render_recon_gen'] = {
            "pts3d": torch.cat([outputs['render']['pts3d'], outputs['render_gen']['pts3d']], dim=1),
            "pts3d_local": torch.cat([outputs['render']['pts3d_local'], outputs['render_gen']['pts3d_local']], dim=1),
            "conf": torch.cat([outputs['render']['conf'], outputs['render_gen']['conf']], dim=1),
            "colors": torch.cat([outputs['render']['colors'], outputs['render_gen']['colors']], dim=1),
            "focal": torch.cat([outputs['render']['focal'], outputs['render_gen']['focal']], dim=1),
            "c2w": torch.cat([outputs['render']['c2w'], outputs['render_gen']['c2w']], dim=1),
            "semantic_2ds": (
                torch.cat([outputs['render']['semantic_2ds'], outputs['render_gen']['semantic_2ds']], dim=1)
                if outputs['render']['semantic_2ds'] is not None and outputs['render_gen']['semantic_2ds'] is not None
                else None
            ),
            "is_recon": torch.cat([outputs['render']['is_recon'], outputs['render_gen']['is_recon']], dim=1),
            # "c2w_pose": torch.cat([outputs['render']['c2w_pose'], outputs['render_gen']['c2w_pose']], dim=1)
        }
        
        # =====================================================#
        # 逐场景保存3D点云，并把点云转换成occupancy voxel预测
        for j in tqdm(range(B), leave=False):

            frame_id = data['frame_id'][j]
            voxel_pred_save_dir = os.path.join(args.output_dir, f"{frame_id}_{args.model}")
            os.makedirs(voxel_pred_save_dir, exist_ok=True)
            # =====================================================#
            # 保存点云级别npy
            for name, output in outputs.items():
                has_semantic_output = output.get("semantic_2ds") is not None
                colors_hwc = output['colors'][j].permute(0, 2, 3, 1).cpu().numpy()  # (5 160 512 3)
                save_dict = {
                    "pts3d": output['pts3d'][j].cpu().numpy(),                      # (5 160 512 3)
                    "pts3d_local": output['pts3d_local'][j].cpu().numpy(),          # (5 160 512 3)
                    "colors": colors_hwc,
                    "conf": output['conf'][j].cpu().numpy(),
                    "focal": output['focal'][j].cpu().numpy(),
                    "c2w": output['c2w'][j].cpu().numpy(),
                }
                if has_semantic_output:
                    save_dict["semantic_2ds"] = output["semantic_2ds"][j].cpu().numpy()

                save_path = os.path.join(voxel_pred_save_dir, f"pts3d_{name}.npy")
                np.save(save_path, save_dict)
            
            # =====================================================#
            # 准备voxel参数
            grid_size = tuple(occ_size)
            voxel_predictions_dict = {
                "estimated_input_camera_poses": outputs['render']['estimated_camera_poses'][j].cpu().numpy(),   # (5 4 4) 相机位姿
                "estimated_input_intrinsics": build_intrinsics_from_focal(
                    outputs['render']['focal'][j],
                    H,
                    W,
                ).cpu().numpy(), # 只保存真实输入相机位姿
                "estimated_input_images": convert_images_to_uint8_hwc(outputs['render']['colors'][j]),
                "voxel_size": voxel_size,                       # 0.4
                "voxel_origin": voxel_origin.cpu().numpy(),     # (3) [-40 -40 -3.6]
            }
            # 处理真实重建点云
            recon_output = outputs['render']
            # Process render (reconstruction) output
            render_conf_mask = recon_output['conf'][j] > recon_conf_thres # confidence过滤
            render_pts3d_th = recon_output['pts3d'][j][render_conf_mask]
            render_conf_th = recon_output['conf'][j][render_conf_mask]

            render_semantic_2ds_th = recon_output.get('semantic_2ds', [None])[j]
            render_semantic_2ds_th = render_semantic_2ds_th.to(render_conf_mask.device)
            render_semantic_2ds_th = render_semantic_2ds_th[render_conf_mask]                               # mask:(5 160 512) -> sum():296542
            render_has_semantic = True

            # Create and save render voxel prediction
            # 点云坐标转换：将点云坐标转换到体素坐标系下
            render_pts3d_in_velo = transform_points_torch(T=T_cam_to_voxel.float(), points=render_pts3d_th) # 点云坐标转换 (296542 3)
            # ===============================================#
            # 5.点云 -> 体素 预测：点云体素化流程将3D点坐标根据体素大小和原点映射到体素网格，通过三线性插值将点的占用信息和语义置信度分配到邻近体素，并聚合得到每个体素的占用状态与语义类别，最终生成3D语义体素预测。
            # 3D占用预测结果通过聚合所有点图并使用三线性插值进行体素化处理得到
            # 
            render_voxel_pred = create_voxel_prediction(
                render_pts3d_in_velo, render_has_semantic, render_semantic_2ds_th, render_conf_th,  # (296542 3) True (296542) (296542)
                grid_size, voxel_origin, voxel_size,
                n_classes, other_class, empty_class
            )   # (200 200 24)
            render_voxel_pred_np = render_voxel_pred.cpu().numpy().astype(np.uint8)
            voxel_predictions_dict[f"render_th{recon_conf_thres}"] = render_voxel_pred_np
            # ================================================#
            # 新视角渲染 -> 体素化过程，用于补全
            # If render_gen exists, also create and save combined output
            if 'render_gen' in outputs:
                gen_output = outputs['render_gen']

                # Filter generation output
                gen_conf_mask = gen_output['conf'][j] > gen_conf_thres
                gen_pts3d_th = gen_output['pts3d'][j][gen_conf_mask]
                gen_conf_th = gen_output['conf'][j][gen_conf_mask]
                if args.semantic is not None:
                    gen_semantic_2ds_th = gen_output.get('semantic_2ds', [None])[j]
                    if gen_semantic_2ds_th is not None:
                        gen_semantic_2ds_th = gen_semantic_2ds_th.to(gen_conf_mask.device)
                        gen_semantic_2ds_th = gen_semantic_2ds_th[gen_conf_mask]
                        gen_has_semantic = True
                    else:
                        gen_has_semantic = False
                        gen_semantic_2ds_th = None
                else:
                    gen_has_semantic = False
                    gen_semantic_2ds_th = None
                
                if gen_pts3d_th.numel() == 0:
                    gen_voxel_pred = torch.full(
                        grid_size,
                        empty_class,
                        dtype=torch.long,
                        device=render_voxel_pred.device,
                    )
                else:
                    gen_pts3d_in_velo = transform_points_torch(
                        T=T_cam_to_voxel.float(),
                        points=gen_pts3d_th,
                    )
                    gen_voxel_pred = create_voxel_prediction(
                        gen_pts3d_in_velo,
                        gen_has_semantic,
                        gen_semantic_2ds_th,
                        gen_conf_th,
                        grid_size,
                        voxel_origin,
                        voxel_size,
                        n_classes,
                        other_class,
                        empty_class,
                    )
                
                voxel_pred = gen_voxel_pred.clone()
                non_empty_mask = render_voxel_pred != empty_class
                voxel_pred[non_empty_mask] = render_voxel_pred[non_empty_mask] # 以新视角生成的voxel作为基础，使用真实重建视角voxel覆盖对应位置，得到融合后的occupancy

                voxel_pred_np = voxel_pred.cpu().numpy().astype(np.uint8)
                voxel_pred_np = maybe_apply_pooling(voxel_pred_np)
                print("Number of occupied voxels:", np.sum(voxel_pred_np != empty_class))
                key = f"render_recon_gen_recon{recon_conf_thres}_gen{gen_conf_thres}"
                voxel_predictions_dict[key] = voxel_pred_np
                print(f"Added combined voxel prediction: {key}")
            
            save_path = os.path.join(voxel_pred_save_dir, "voxel_predictions.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(voxel_predictions_dict, f)
            print(f"Saved voxel predictions dictionary: {save_path}")

            item_count += 1  # Increment item counter after processing each item in batch
            
            # Clean up tensors from current iteration
            del outputs, data, imgs
            if 'sam2_imgs' in locals() and sam2_imgs is not None:
                del sam2_imgs
            if 'sam3_imgs' in locals() and sam3_imgs is not None:
                del sam3_imgs
            if 'recon_semantic_2ds' in locals() and recon_semantic_2ds is not None:
                del recon_semantic_2ds
            if 'gen_semantic_2ds' in locals() and gen_semantic_2ds is not None:
                del gen_semantic_2ds
            if 'semantic_2ds' in locals():
                del semantic_2ds
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    print("=" * 50)
    print(f"Total items processed: {item_count}")