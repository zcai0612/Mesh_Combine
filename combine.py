import os
import torch
import imageio
import time
from PIL import Image
from torchvision.utils import save_image
from core.remesh import calc_vertex_normals
from core.opt import MeshOptimizer
from utils.poisson import poisson_reconstruction
from utils.mesh_utils import load_mesh_from_obj, normalize_vert, save_pointcloud_obj

from mesh_renderer.renderer import NVRenderer, compute_normal
from mesh_renderer.cameras import CameraData
from tqdm import tqdm
from pytorch3d.io import save_obj

from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes

class Combine:
    def __init__(self, device, opt):
        self.opt = opt
        self.device = device
        self.renderer = NVRenderer(device)
        camera = CameraData(opt.camera)
        self.train_cams = camera.generate_rand_cameras(size=opt.camera.num_views_train)
        self.test_cams = camera.generate_circle_cameras(size=opt.camera.num_views_test)
        self.filter_cams = camera.generate_circle_cameras(
            h=opt.camera.filter_render_h, 
            w=opt.camera.filter_render_w, 
            size=opt.camera.num_views_filter
        )

        self.train_opt = opt.train  


    def load_mesh(self, mesh_path):
        v, f, _, _ = load_mesh_from_obj(mesh_path, device=self.device, auto_uv=False)
        v = normalize_vert(v)
        return v, f

    def filter_inivisible_verts(self, v, f, subdivision=0):
        visible_faces = []
        if subdivision>0:
            # 执行mesh subdivision
            # 创建Meshes对象
            mesh = Meshes(verts=[v], faces=[f])
            subdivider = SubdivideMeshes()
            # 进行一次subdivision
            for _ in range(subdivision):
                mesh = subdivider(mesh)
                
            # 获取最终的顶点和面片
            v = mesh.verts_packed()
            f = mesh.faces_packed()
        vn = compute_normal(v, f)
        for viewpoint_cam in self.filter_cams:
            render_pkg = self.renderer.render_from_camera(v, f, viewpoint_cam)
            visible_face = render_pkg['visible_faces']
            # 将visible_face转换为1维tensor并找出非零值
            visible_face = visible_face.detach().cpu()
            non_zero_values = visible_face.flatten()[visible_face.flatten().nonzero()]
            non_zero_values = [int(x) for x in non_zero_values]
            # 如果non_zero_values是单个数值，确保它被转换为列表
            if isinstance(non_zero_values, (int, float)):
                non_zero_values = [non_zero_values]
            visible_faces.extend(non_zero_values)
        
        # 使用set去除重复元素，然后转回list
        visible_faces = list(set(visible_faces))
        visible_faces.sort()
        visible_faces = [x - 1 for x in visible_faces]

        # 获取可见面片中包含的所有顶点索引
        visible_vertices_idx = torch.unique(f[visible_faces].flatten())
        
        # 创建新的顶点数组
        new_vertices = v[visible_vertices_idx]

        return new_vertices, vn[visible_vertices_idx]
        

    def run(self, mesh_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        image_save_dir = os.path.join(output_dir, 'images')
        video_save_dir = os.path.join(output_dir, 'videos')
        mesh_save_dir = os.path.join(output_dir, 'meshes')
        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(video_save_dir, exist_ok=True)
        os.makedirs(mesh_save_dir, exist_ok=True)

        v, f = self.load_mesh(mesh_path)
        self.video_test(v, f, os.path.join(video_save_dir, 'init.mp4'))
        save_obj(os.path.join(mesh_save_dir, 'init.obj'), v, f)
        pcd, pn = self.filter_inivisible_verts(v, f, subdivision=self.train_opt.subdivision)
        pcd = pcd.detach().cpu().numpy()
        pn = pn.detach().cpu().numpy()
        save_pointcloud_obj(pcd, os.path.join(mesh_save_dir, 'filter_pcd.obj'), normals=pn)


        print("Start Poisson Reconstruction ...")
        start_time = time.time()
        new_v_np, new_f_np = poisson_reconstruction(pcd, pn)
        end_time = time.time()
        reconstruction_time = end_time - start_time
        print(f"Poisson Reconstruction Done! Time used: {reconstruction_time:.2f} s")

        new_v = torch.from_numpy(new_v_np).to(torch.float32).to(self.device)
        new_f = torch.from_numpy(new_f_np).to(torch.int32).to(self.device)
        new_v = normalize_vert(new_v)
        
        self.video_test(new_v, new_f, os.path.join(video_save_dir, 'poisson.mp4'))
        save_obj(os.path.join(mesh_save_dir, 'poisson.obj'), new_v, new_f)

        opt = MeshOptimizer(new_v, new_f.long(), lr=self.train_opt.lr, edge_len_lims=self.train_opt.edge_len_lims)

        source_v = opt.vertices
        source_f = opt.faces

        target_v = v
        target_f = f

        C_batch_size = self.train_opt.C_batch_size
        total_iters = self.train_opt.total_iters
        image_save_interval = self.train_opt.image_save_interval
        do_recon = self.train_opt.do_recon
        if do_recon:
            pbar = tqdm(range(total_iters+1))
            for iter in pbar:
                normals = []
                masks = []
                normals_target = []
                masks_target = []

                for i in range(C_batch_size):
                    viewpoint_cam = next(self.train_cams)
                    render_pkg = self.renderer.render_from_camera(source_v, source_f, viewpoint_cam)
                    normal = render_pkg['comp_normal'].permute(0, 3, 1, 2)
                    alpha = render_pkg['opacity'].permute(0, 3, 1, 2)
                    normal = normal * alpha
                    normals.append(normal)
                    masks.append(alpha)

                    render_pkg_target = self.renderer.render_from_camera(target_v, target_f, viewpoint_cam)
                    normal_target = render_pkg_target['comp_normal'].permute(0, 3, 1, 2)
                    alpha_target = render_pkg_target['opacity'].permute(0, 3, 1, 2)
                    normal_target = normal_target * alpha_target
                    normals_target.append(normal_target)
                    masks_target.append(alpha_target)

                normals = torch.cat(normals, dim=0)
                masks = torch.cat(masks, dim=0)
                normals_target = torch.cat(normals_target, dim=0)
                masks_target = torch.cat(masks_target, dim=0)

                loss_normal = (normals - normals_target).abs().mean()
                loss_mask = (masks - masks_target).abs().mean()

                loss = loss_normal + loss_mask
                loss.backward()
                opt.step()
                source_v, source_f = opt.remesh()

                with torch.no_grad():
                    # 更新tqdm进度条显示loss信息
                    pbar.set_postfix({
                        'loss': f'{loss.item():.6f}',
                        'normal_loss': f'{loss_normal.item():.6f}',
                        'mask_loss': f'{loss_mask.item():.6f}'
                    })

                    if iter % image_save_interval == 0:
                        save_image(torch.cat([
                            normals[0].unsqueeze(0), normals_target[0].unsqueeze(0)
                        ], dim=0), os.path.join(image_save_dir, f'{iter}.png'))

        self.video_test(source_v, source_f, os.path.join(video_save_dir, 'final.mp4'))
        save_obj(os.path.join(mesh_save_dir, 'final.obj'), source_v, source_f)

    @torch.no_grad()
    def video_test(self, v, f, output_path):
        normal_frames = []
        for viewpoint_cam in self.test_cams:
            render_pkg = self.renderer.render_from_camera(
                v, f, viewpoint_cam,
            )
            normal = render_pkg['comp_normal'].permute(0, 3, 1, 2)
            alpha = render_pkg['opacity'].permute(0, 3, 1, 2)
            normal = normal * alpha
            
            normal = normal.detach().cpu().squeeze(0).permute(1,2,0).numpy()
            normal = (normal * 255).round().astype('uint8')
            normal_frames.append(normal)
        imageio.mimwrite(output_path, normal_frames, fps=30, quality=8)

if __name__ == '__main__':
    import argparse
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/combine.yaml', help="path to the yaml configs file")
    parser.add_argument("--mesh",  help="path to the mesh file, can be .obj path or a directory full of .obj files")
    parser.add_argument("--output_dir", default='output', help="path to the output directory")
    parser.add_argument("--device", default='cuda', help="path to the output directory")
    args = parser.parse_args()

    opt = OmegaConf.load(args.config)
    combine = Combine(args.device, opt)

    if os.path.isfile(args.mesh):
        # 如果是文件，直接处理
        combine.run(args.mesh, args.output_dir)
    elif os.path.isdir(args.mesh):
        # 如果是目录，遍历处理所有.obj文件
        for filename in os.listdir(args.mesh):
            if filename.endswith('.obj'):
                mesh_path = os.path.join(args.mesh, filename)
                # 为每个文件创建对应的输出目录
                output_subdir = os.path.join(args.output_dir, os.path.splitext(filename)[0])
                combine.run(mesh_path, output_subdir)
    else:
        raise FileNotFoundError(f"Input mesh: {args.mesh} is not a .obj file or a directory")
