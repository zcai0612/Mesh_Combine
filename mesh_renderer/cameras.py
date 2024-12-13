import torch
import numpy as np
import random
from utils.graphics_utils import getProjectionMatrix, getProjectionMatrix2, getWorld2View2, focal2fov, fov2focal
from utils.common_utils import safe_normalize

def cyclic_iterator(lst):
    """Return an iterator that cycles through the elements of lst indefinitely."""
    while True:
        for item in lst:
            yield item


def circle_poses(radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0])):
    theta = torch.tensor([theta / 180 * np.pi])
    phi = torch.tensor([phi / 180 * np.pi])

    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.cos(theta),
        radius * torch.sin(theta) * torch.cos(phi),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 1, 0]).unsqueeze(0).repeat(len(centers), 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses.numpy()

class Camera:
    def __init__(self, R, T, FoVx, FoVy, image_width, image_height, delta_azimuth,
                trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda") -> None:
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.delta_azimuth = delta_azimuth
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        self.zfar = 100.0
        self.znear = 0.01
        
        self.image_width = image_width
        self.image_height = image_height
        
        self.trans = trans
        self.scale = scale
        
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.projection_matrix = getProjectionMatrix2(znear=self.znear, zfar=self.zfar, fovX=self.FoVx)
        # self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class CameraData:
    def __init__(self, opt) -> None:
        self.opt = opt

        self.fovy_range = opt.fovy_range
        self.radius_range = opt.radius_range
        self.theta_range = opt.theta_range
        self.phi_range = opt.phi_range

        self.default_fov = opt.default_fov
        self.default_polar = opt.default_polar
        self.default_radius = opt.default_radius
        self.default_azimuth = opt.default_azimuth

        self.image_h = opt.image_h
        self.image_w = opt.image_w
        
        self.center = opt.center
        
    def generate_rand_cameras(self, size=50):
        cam_infos = []
        
        image_h = self.image_h
        image_w = self.image_w
        
        for idx in range(size):
            theta = random.random() * (self.theta_range[1] - self.theta_range[0]) + self.theta_range[0]
            phi = random.random() * (self.phi_range[1] - self.phi_range[0]) + self.phi_range[0]
            delta_azimuth = phi - self.default_azimuth
            if delta_azimuth > 180: 
                delta_azimuth -= 360 
            fov = random.random() * (self.fovy_range[1] - self.fovy_range[0]) + self.fovy_range[0]
            radius = random.random() * (self.radius_range[1] - self.radius_range[0]) + self.radius_range[0]
            poses = circle_poses(
                radius=radius, theta=theta, phi=phi
            )
            matrix = np.linalg.inv(poses[0])
            R = -np.transpose(matrix[:3,:3])
            # R[:,0] = -R[:,0]
            T = -matrix[:3, 3]
            # matrix = poses[0]
            # R = matrix[:3,:3]
            # T = matrix[:3, 3]
            fovy = focal2fov(fov2focal(fov, image_h), image_w)
            FovY = fov
            FovX = fovy
            
            cam_infos.append(
                Camera(
                    R=R, T=T, FoVx=FovX, FoVy=FovY, image_width=image_w, image_height=image_h,
                    delta_azimuth=delta_azimuth
                )
            )
            
        return cyclic_iterator(cam_infos)
    
    def generate_circle_cameras(self, h=512, w=512, size=50):
        cam_infos = []
        
        image_h = h
        image_w = w

        radius = self.default_radius
        
        theta = self.default_polar
        fov = self.default_fov
        
        for idx in range(size):
            phi = (idx/size) * 360
            delta_azimuth = phi - self.default_azimuth
            if delta_azimuth > 180: 
                delta_azimuth -= 360 

            poses = circle_poses(
                radius=radius, theta=theta, phi=phi
            )
            matrix = np.linalg.inv(poses[0])
            R = -np.transpose(matrix[:3,:3])
            # R[:,0] = -R[:,0]
            T = -matrix[:3, 3]
            # matrix = poses[0]
            # R = matrix[:3,:3]
            # T = matrix[:3, 3]
            fovy = focal2fov(fov2focal(fov, image_h), image_w)
            FovY = fov
            FovX = fovy
            
            cam_infos.append(
                Camera(
                    R=R, T=T, FoVx=FovX, FoVy=FovY, image_width=image_w, image_height=image_h,
                    delta_azimuth=delta_azimuth
                )
            )
            
        return cam_infos

