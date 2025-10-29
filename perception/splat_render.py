import numpy as np
import torch
import cv2
import json

from pathlib import Path
from transforms3d.euler import euler2mat
from typing import Dict

# from torch.serialization import add_safe_globals

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap
from nerfstudio.viewer.utils import CameraState, get_camera
from nerfstudio.cameras.cameras import CameraType

from scipy.spatial.transform import Rotation as R
from transforms3d.euler import mat2euler

try:
    from .modules import BaseModule
except ImportError:
    from modules import BaseModule


class SplatRenderer(BaseModule):
    def __init__(self, config_path: str, json_path: str, aspect_ratio: float = 16/9):

        config, pipeline, _, _ = eval_setup(Path(config_path), eval_num_rays_per_chunk=None, test_mode="test")
        self.config = config
        self.pipeline = pipeline
        self.model = pipeline.model
        self.model.eval()
        self.device = self.model.device
        self.aspect_ratio = aspect_ratio

        # Set background
        self.background_color = torch.tensor([0.1490, 0.1647, 0.2157], device=self.device)
        self.model.set_background(self.background_color)

        # Load camera settings
        with open(json_path, 'r') as f:
            camera_data = json.load(f)
        cam_matrix = np.array(camera_data["camera"]["c2w_matrix"])
        self.init_position = cam_matrix[:, 3]
        self.init_orientation = cam_matrix[:, :3]

        fov = camera_data["camera"].get("fov_radians", 1.3089969389957472)
        resolution = camera_data["camera"].get("render_resolution", 1080)
        self.fov = fov
        self.image_height = resolution
        self.image_width = int(resolution * aspect_ratio)

        self.colormap_options_rgb = ColormapOptions(colormap='default', normalize=True)
        self.colormap_options_depth = ColormapOptions(colormap='gray', normalize=True)

    
    def getNEDPose(self, splatRT):
        Rsplat = splatRT[:3, :3]
        tsplat = splatRT[:3, 3]

        RsplatInv = Rsplat.T
        euf = self.init_orientation.T @ (tsplat - self.init_position)
        x = -euf[2]
        y = euf[0]
        z = -euf[1]
        print('NED Position ', x, y, z)

        # now get the orientation
        R_ned = self.init_orientation.T @ Rsplat
        pitch, yaw, roll = mat2euler(R_ned)
        print('NED Orientation (rpy) ', np.degrees(-roll), np.degrees(pitch), np.degrees(-yaw))


        return [x,y,z, np.degrees(-roll), np.degrees(pitch), np.degrees(-yaw)]


    def render(self, position: np.ndarray, orientation_rpy: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Render RGB and depth images from Gaussian Splat.
        
        Args:
            position: (3,) array [x, y, z] in meters (NED frame)
            orientation_rpy: (3,) array [roll, pitch, yaw] in radians (NED frame)
            **kwargs: Additional parameters (unused but included for interface consistency)
        
        Returns:
            Dictionary with 'rgb' (H,W,3 uint8), 'depth' (H,W,3 uint8 colormap), 
            and 'depth_raw' (H,W float32 raw depth values)
        """
        # Position transform from NED → GSplat (NWU)
        pos_update = self.init_orientation @ np.array([
            [position[1]],  # East
            [-position[2]], # Up
            [-position[0]]  # Forward
        ])
        pos_cam = self.init_position + pos_update.flatten()

        # Orientation (NED to GSplat)
        R_cam = self.init_orientation @ euler2mat(
            orientation_rpy[1], -orientation_rpy[2], -orientation_rpy[0]
        )
        
        c2w = torch.tensor(np.column_stack([R_cam, pos_cam]), dtype=torch.float32, device=self.device)
        
        camera_state = CameraState(
            fov=self.fov,
            aspect=self.aspect_ratio,
            c2w=c2w,
            camera_type=CameraType.PERSPECTIVE
        )

        camera = get_camera(camera_state, self.image_height, self.image_width).to(self.device)
        outputs = self.model.get_outputs_for_camera(camera)

        rgb = apply_colormap(outputs["rgb"], self.colormap_options_rgb)
        rgb = (rgb * 255).type(torch.uint8).cpu().numpy()
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = apply_colormap(outputs["depth"], self.colormap_options_depth)
        depth = (depth * 255).type(torch.uint8).cpu().numpy()

        return {
            'rgb': rgb,
            'depth': depth,
            'depth_raw': outputs["depth"].cpu().numpy()
        }