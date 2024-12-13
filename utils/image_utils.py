import time
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from typing import List
from diffusers.utils.loading_utils import load_image
from PIL import Image
from rembg import remove
from rembg.session_factory import new_session


def preprocess_image(img_pil, ratio=1.85/2.0, resolution=512):
    img = np.array(img_pil) # H,W,C=3 
    # remove background
    img_rembg = remove(img, post_process_mask=True, session=new_session("u2net")) 
    # resize & center human
    ret, mask = cv2.threshold(img_rembg[..., -1], 0, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    side_len = int(max_size / ratio) 
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len // 2
    padded_image[
        center - h // 2 : center - h // 2 + h,
        center - w // 2 : center - w // 2 + w,
    ] = img_rembg[y : y + h, x : x + w]
    # resize image
    rgba = Image.fromarray(padded_image).resize((resolution, resolution), Image.LANCZOS)
    # white bg
    rgba_arr = np.array(rgba) / 255.0
    rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
    rgb_pil = Image.fromarray((rgb * 255).astype(np.uint8))
    # mask
    image = (rgba_arr * 255).astype(np.uint8)
    color_mask = image[..., -1]
    image = (rgb * 255).astype(np.uint8)
    invalid_color_mask = color_mask < 255*0.5
    threshold =  np.ones_like(image[:,:,0]) * 250
    invalid_white_mask = (image[:, :, 0] > threshold) & (image[:, :, 1] > threshold) & (image[:, :, 2] > threshold)
    invalid_color_mask_final = invalid_color_mask & invalid_white_mask
    color_mask = (1 - invalid_color_mask_final) > 0
    mask_pil = Image.fromarray((color_mask * 255).astype(np.uint8))
    
    return rgb_pil, mask_pil

def resize_and_padding(image, resolution=768):
    """
    将图片resize并padding成正方形
    Args:
        image: PIL Image对象
        resolution: 目标分辨率
    Returns:
        PIL Image对象
    """
    # 获取原始尺寸
    w, h = image.size
    
    # 计算缩放比例
    ratio = min(resolution/h, resolution/w)
    
    # 计算缩放后的尺寸
    new_h = int(h * ratio)
    new_w = int(w * ratio)
    
    # 先将图片按比例缩放
    resized_image = image.resize((new_w, new_h), Image.LANCZOS)
    
    # 创建目标尺寸的空白图片(白色背景)
    result = Image.new('RGB', (resolution, resolution), (255, 255, 255))
    
    # 计算粘贴的位置（居中）
    paste_x = (resolution - new_w) // 2
    paste_y = (resolution - new_h) // 2
    
    # 将缩放后的图片粘贴到居中位置
    result.paste(resized_image, (paste_x, paste_y))
    
    return result




def t2i(tensor):
    np_tensor = torch.clamp(tensor, -1, 1).permute(1, 2, 0).cpu().numpy()[:,:,:3]
    img = Image.fromarray(((np_tensor + 1) / 2 * 255).astype(np.uint8))
    return img

def to_world(normal_cam, view_angle):
    '''
        normal_cam : [4, h, w] tensor. normal map in camera view 
        view_angle: yaw angle of camera view in degree
    '''
    view_angle = torch.deg2rad(torch.tensor(view_angle))
    rot = torch.tensor([[torch.cos(view_angle), 0, torch.sin(view_angle)],
                        [0, 1, 0],
                        [-torch.sin(view_angle), 0, torch.cos(view_angle)]], dtype=normal_cam.dtype, device=normal_cam.device)
    mask = normal_cam[3, :, :] >= 0
    normal_world = normal_cam.clone()
    normal_world[:3, mask] = torch.einsum('ij, jk->ik', rot.T, normal_cam[:3, mask])

    return normal_world

def to_cam(normal_world, view_angle):
    '''
        normal_world : [4, h, w] tensor. normal map in world coordinate 
        view_angle: yaw angle of camera view in degree
    '''
    view_angle = torch.deg2rad(torch.tensor(view_angle))
    rot = torch.tensor([[torch.cos(view_angle), 0, torch.sin(view_angle)],
                        [0, 1, 0],
                        [-torch.sin(view_angle), 0, torch.cos(view_angle)]], dtype=normal_world.dtype, device=normal_world.device)
    mask = normal_world[3] >= 0
    normal_cam = normal_world.clone()
    normal_cam[:3, mask] = torch.einsum('ij, jk->ik', rot, normal_world[:3, mask])

    return normal_cam

def to_cam_B(normal_B_world, view_angle):
    '''
        normal_B_world : [4, h, w] tensor. backward normal map in world coordinate 
        view_angle: yaw angle of camera view in degree
    '''
    if normal_B_world.shape[0] != 4:
        raise ValueError
    view_angle_B = (view_angle + 180) % 360
    normal_B_cam = to_world(to_cam(normal_B_world, view_angle_B), 180)
    normal_B_cam_flipped = torch.flip(normal_B_cam, dims=[2])
    
    return normal_B_cam_flipped

def fliplr_nml(normal_map):
    '''
        args:
            normal_map: [4, H, W] torch.Tensor 
    '''
    normal_map_flipped = torch.flip(normal_map, dims=[2])
    mask = normal_map_flipped[3] >= 0
    normal_map_flipped[0, mask] = -normal_map_flipped[0, mask]  #  flip x

    return normal_map_flipped

def to_pil(tensor):
    '''
        tensor: [3 or 4, H, W] torch.Tensor with range [-1, 1] 
    '''
    return transforms.ToPILImage()((torch.clamp(tensor, -1, 1).detach().cpu() + 1) * 0.5)

def np_to_tensor(array):
    rgbd_image_tensor = torch.from_numpy(array).float() / 255.0
    rgbd_image_tensor = rgbd_image_tensor.permute(2, 0, 1) * 2.0 - 1.0
    return rgbd_image_tensor
    
def pil_concat_h(pil_list, h=512, w=512):
    dst = Image.new('RGBA', (w * len(pil_list), h), "black")
    for idx, image in enumerate(pil_list):
        dst.paste(image, (w * idx, 0))
    return dst

def pil_concat_v(pil_list, h=512, w=512):
    dst = Image.new('RGBA', (w, h * len(pil_list)), "black")
    for idx, image in enumerate(pil_list):
        dst.paste(image, (0, h * idx))
    return dst

def load_smpl_images(smpl_dirs, res=512, device='cpu'):
    smpl_images = []
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(res),
        transforms.Normalize(0.5, 0.5)
    ])

    for smpl_dir in smpl_dirs:
        smpl_image = np.array(Image.open(smpl_dir))
        smpl_image = transf(smpl_image).to(device)
        smpl_images.append(smpl_image)
    return smpl_images

def to_rgba(tensor):
    if tensor.shape[0] == 4:
        return tensor
    alpha = 2 * (tensor[2, :, :] >= 0) - 1
    rgba_tensor = torch.cat([tensor, alpha[None]], dim=0)
    return rgba_tensor

def merge_frontback(normal_F, normal_B, ch=4):
    normal_F = normal_F.permute(2,0,1)
    normal_B = normal_B.permute(2,0,1)
    normal_B = fliplr_nml(normal_B)
    normal_cam = torch.cat([normal_F[:ch,:,:], normal_B[:ch,:,:]], dim=0)
    return normal_cam

def frontback_img(normal, ch_slice=4):
    normal_f, normal_b = normal[:ch_slice,:,:], normal[ch_slice:,:,:]
    normal_fc, normal_bc = to_rgba(normal_f), torch.flip(to_rgba(normal_b), dims=[2])
    fc_mask, bc_mask = (normal_fc[3,:,:] >= 0),  (normal_bc[3,:,:] >= 0)

    normal_bc[0, bc_mask] *= -1  
    if ch_slice == 3: 
        normal_fc[:3, ~fc_mask] = -1
        normal_bc[2, bc_mask] *= -1
        normal_bc[:3, ~bc_mask] = -1

    normal_fi, normal_bi = t2i(normal_fc), t2i(normal_bc)
    normal_i = pil_concat_h([normal_fi, normal_bi])
    return normal_i, normal_fi, normal_bi

def resample_cam_input(normal_world_F, normal_world_B, angle, ch_slice=4):
    normal_cam_F = to_cam(normal_world_F, angle)
    normal_cam_B = fliplr_nml(to_cam(normal_world_B, (angle + 180) % 360))
    normal_cam = torch.cat([normal_cam_F[:ch_slice], normal_cam_B[:ch_slice]], dim=0)
    return normal_cam

def resample_img(normal_cam, normal_re, ch_slice=4):
    normal_fci, normal_bci = t2i(normal_cam[:ch_slice]), t2i(normal_cam[ch_slice:])
    normal_re_f, normal_re_b = normal_re[0, :ch_slice], normal_re[0, ch_slice:]

    normal_re_fi = t2i(to_rgba(normal_re_f))
    normal_re_bi = t2i(to_rgba(normal_re_b))
    normal_re_flip_bi = t2i(fliplr_nml(to_rgba(normal_re_b)))

    compare_fi = pil_concat_h([normal_fci, normal_re_fi])
    compare_bi = pil_concat_h([normal_bci, normal_re_bi])
    compare_i = pil_concat_v([compare_fi, compare_bi], w=compare_fi.size[0])
    return normal_re_fi, normal_re_flip_bi, compare_i