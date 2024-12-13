import random
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from mesh_renderer.typing import *
from mesh_renderer.rasterize import NVDiffRasterizerContext
from utils.common_utils import safe_normalize, dot
from .cameras import Camera

def compute_normal(vertices, faces):
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.as_tensor(vertices).float()
    if not isinstance(faces, torch.Tensor):
        faces = torch.as_tensor(faces).long()

    i0, i1, i2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()

    v0, v1, v2 = vertices[i0, :], vertices[i1, :], vertices[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    vn = torch.zeros_like(vertices)
    vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    vn = torch.where(dot(vn, vn) > 1e-20, vn, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device))
    vn = safe_normalize(vn)

    face_normals = safe_normalize(face_normals)
    return vn

class NVRenderer(torch.nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.ctx = NVDiffRasterizerContext('cuda', device)
        
    def render_from_camera(
        self, verts, faces, cam: Camera, render_rgb: bool=False,
        vt=None, ft=None, albedo=None,
    ):  
        full_proj_transform = cam.full_proj_transform.clone()
        # full_proj_transform[:,0] = -full_proj_transform[:,0]
        full_proj = full_proj_transform.T[None, ...]
        
        return self.render_mesh(
            verts, faces, mvp_mtx=full_proj, height=cam.image_height, width=cam.image_width,
            render_rgb=render_rgb, vt=vt, ft=ft, albedo=albedo
        )
        
    def render_mesh(
        self,
        v, f,
        mvp_mtx: Float[Tensor, "B 4 4"],
        height: int,
        width: int,
        render_rgb: bool = False,
        vt=None, ft=None, albedo=None,
    ) -> Dict[str, Any]:
        f = f.int()
        mvp_mtx = mvp_mtx.to(v.device)
        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            v, mvp_mtx
        )

        rast, rast_db = self.ctx.rasterize(v_pos_clip, f, (height, width))
        mask = rast[..., 3:] > 0
        visible_faces = rast[..., 3]
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, f)

        out = {"opacity": mask_aa}
        out.update({"visible_faces": visible_faces}) 

        # vn = -compute_normal(v_pos_clip[0, :, :3], f)
        vn = compute_normal(v, f)

        gb_normal, _ = self.ctx.interpolate_one(vn.float(), rast, f)

        gb_normal = (gb_normal + 1.) / 2.

        gb_normal_aa = self.ctx.antialias(
            gb_normal, rast, v_pos_clip, f
        )
        out.update({"comp_normal": gb_normal_aa})  # in [0, 1]

        # gb_depth = rast[..., 2:3]
        # gb_depth = 1./(gb_depth + 1e-7)
        gb_depth, _ = self.ctx.interpolate_one(v_pos_clip[0,:, :3].contiguous(), rast, f)
        gb_depth = 1./(gb_depth[..., 2:3] + 1e-7)
        # print(gb_depth.shape, gb_depth[mask[..., 0]].shape)
        max_depth = torch.max(gb_depth[mask[..., 0]])
        min_depth = torch.min(gb_depth[mask[..., 0]])
        gb_depth_aa = torch.lerp(
                torch.zeros_like(gb_depth), (gb_depth - min_depth) / (max_depth - min_depth + 1e-7), mask.float()
            )
        gb_depth_aa = self.ctx.antialias(
            gb_depth_aa, rast, v_pos_clip, f
        )
        out.update({"comp_depth":gb_depth_aa})  # in [0, 1]

        if render_rgb:
            albedo = self.ctx.get_2d_texture(
                vt[None, ...], ft, albedo.unsqueeze(0),
                rast, rast_db=rast_db, diff_attrs='all', filter_mode='linear-mipmap-linear'
            )
            color = albedo
            color = self.ctx.antialias(color, rast, v_pos_clip, f)

            out.update({"rgb": color})
        else:
            color = gb_depth_aa.repeat(1,1,1,3)
            out.update({"rgb": color})
        return out
        
