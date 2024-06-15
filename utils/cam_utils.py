import numpy as np
from scipy.spatial.transform import Rotation as R

import torch
from .common import safe_normalize, make_rotate


def look_at(campos, target, opengl=True):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    if not opengl:
        # camera forward aligns with -z
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


# elevation & azimuth to pose (cam2world) matrix
def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
    # radius: scalar
    # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    # return: [4, 4], camera pose matrix
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T

def undo_orbit_camera(T, is_degree=True):
    # T: [4, 4], camera pose matrix
    # return: elevation, azimuth, radius
    campos = T[:3, 3]
    radius = np.linalg.norm(campos)
    elevation = np.arcsin(-campos[1] / radius)
    azimuth = np.arctan2(campos[0], campos[2])
    if is_degree:
        elevation = np.rad2deg(elevation)
        azimuth = np.rad2deg(azimuth)
    return elevation, azimuth, radius

class Camera:
    def __init__(self, phi=0, theta=0): 
        self.phi = phi
        self.theta = theta
        self.s = 1
 
    @property
    def extrinsic(self): 
        M = np.eye(4, dtype=np.float32)
        R = make_rotate(np.radians(self.theta), 0, 0) @ make_rotate(0, np.radians(self.phi), 0)
        M[:3, :3] = R
        return M
    
    @property
    def intrinsics(self):
        return np.array([
            [self.s, 0, 0, 0],
            [0, self.s, 0, 0],
            [0, 0, self.s, 0],
            [0, 0, 0, 1]], dtype=np.float32
            ) 
        
  
    @property
    def mvp(self):
        return self.intrinsics @  self.extrinsic  # [4, 4]

    def orbit(self, dx, dy): 
        self.phi += np.radians(dx)
        self.theta += np.radians(dy)
        
    def scale(self, delta):
        self.s *= 1.1 ** delta

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])