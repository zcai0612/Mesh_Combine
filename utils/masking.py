import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import trimesh
import json
from trimesh import Trimesh


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


#really good method for setting class attribute from dictionary
class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class SMPLXLiteSeg:
    smplx_dir = "./models/smplx-lite"
    smplx_segs = json.load(open(f"{smplx_dir}/spec_verts_idx.json"))
    head_ids = smplx_segs["head"]
    hand_ids = smplx_segs["hand"]
    hand_head_ids = head_ids + hand_ids

# class SMPLXSeg:
#     smplx_dir = "./models/smplx"
#     smplx_segs = json.load(open(f"{smplx_dir}/smplx_vert_segementation.json"))
#     head_ids = smplx_segs["head"]


# numpy -> numpy
class Masking:
    def __init__(self, vertices, faces, v_mask) -> None:
        if not isinstance(vertices, np.ndarray) or not isinstance(faces, np.ndarray) or not isinstance(v_mask, np.ndarray):
            raise Exception("Masker not initialized with all array!")
        self.vertices = vertices
        self.faces = faces
        self.v_mask = v_mask

    def get_triangle_mask(self):
        f = self.faces
        m = self.v_mask
        selected = []
        for i in range(f.shape[0]):
            l = f[i]
            valid = 0
            for j in range(3):
                if l[j] in m:
                    valid += 1
            if valid == 3:
                selected.append(i)

    def get_binary_triangle_mask(self):
        faces = self.faces
        mask = self.v_mask
        reduced_faces = []
        for f in faces:
            valid = 0
            for v in f:
                if v in mask:
                    valid += 1
                # print(f"3367 in mask:{3367 in mask}")
            reduced_faces.append(True if valid > 0 else False)
        return reduced_faces