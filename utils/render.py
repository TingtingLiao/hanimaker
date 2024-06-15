import os 
import random 
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from utils.common import scale_img_nhwc


class Renderer(torch.nn.Module):
    def __init__(self, gui=False):
        super().__init__()  

        self.glctx = dr.RasterizeCudaContext()

        # if not gui or os.name == 'nt': 
        #     self.glctx = dr.RasterizeCudaContext()
        # else:
        #     self.glctx = dr.RasterizeGLContext()

    def forward(self, mesh, mvp,
                h=512,
                w=512,
                light_d=None,
                ambient_ratio=1.,
                mode='rgb',
                spp=1, 
                is_train=False):
        """
        Args:
            spp:
            return_normal:
            transform_nml:
            mesh: Mesh object
            mvp: [batch, 4, 4]
            h: int
            w: int
            light_d:
            ambient_ratio: float
            shading: str shading type albedo, normal,
            ssp: int
        Returns:
            color: [batch, h, w, 3]
            alpha: [batch, h, w, 1]
            depth: [batch, h, w, 1]

        """
        B = mvp.shape[0]
        v_clip = torch.bmm(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).expand(B, -1, -1),
                           torch.transpose(mvp, 1, 2)).float()  # [B, N, 4]
         

        res = (int(h * spp), int(w * spp)) if spp > 1 else (h, w)
        rast, rast_db = dr.rasterize(self.glctx, v_clip, mesh.f, res)

        ################################################################################
        # Interpolate attributes
        ################################################################################

        # Interpolate world space position
        alpha, _ = dr.interpolate(torch.ones_like(v_clip[..., :1]), rast, mesh.f)  # [B, H, W, 1]
        depth = rast[..., [2]]  # [B, H, W]

        if is_train:
            vn, _ = utils.compute_normal(v_clip[0, :, :3], mesh.f)
            normal, _ = dr.interpolate(vn[None, ...].float(), rast, mesh.f)
        else:
            normal, _ = dr.interpolate(mesh.vn[None, ...].float(), rast, mesh.f)

        # Texture coordinate
        if not mode == 'normal': 
            albedo = self.get_2d_texture(mesh, rast, rast_db)

        if mode == 'normal':
            color = (normal + 1) / 2.
        elif mode == 'rgb':
            color = albedo
        else:  # lambertian
            lambertian = ambient_ratio + (1 - ambient_ratio) * (normal @ light_d.view(-1, 1)).float().clamp(min=0)
            color = albedo * lambertian.repeat(1, 1, 1, 3)

        normal = (normal + 1) / 2.

        normal = dr.antialias(normal, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        color = dr.antialias(color, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        alpha = dr.antialias(alpha, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]

        # inverse super-sampling
        if spp > 1:
            color = scale_img_nhwc(color, (h, w))
            alpha = scale_img_nhwc(alpha, (h, w))
            normal = scale_img_nhwc(normal, (h, w))

        # return color, normal, alpha
        return {
            'image': color,
            'normal': normal,
            'alpha': alpha
        }
 

    def render_normal(self, vertex, faces, vertex_normals, mvp, h=512, w=512, spp=1):
        B = mvp.shape[0]
        v_clip = torch.bmm(F.pad(vertex, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).expand(B, -1, -1),
                           torch.transpose(mvp, 1, 2)).float()  # [B, N, 4]
        
        res = (int(h * spp), int(w * spp)) if spp > 1 else (h, w)
        rast, rast_db = dr.rasterize(self.glctx, v_clip, faces, res)

        alpha, _ = dr.interpolate(torch.ones_like(v_clip[..., :1]), rast, faces)  # [B, H, W, 1]
        depth = rast[..., [2]]  # [B, H, W]

        normal, _ = dr.interpolate(vertex_normals[None, ...].float(), rast, faces)
        normal = (normal + 1) / 2.

        normal = dr.antialias(normal, rast, v_clip, faces).clamp(0, 1)  # [H, W, 3]
        if spp > 1:
            normal = scale_img_nhwc(normal, (h, w))
            alpha = scale_img_nhwc(alpha, (h, w))
        return normal, alpha

    @staticmethod
    def get_2d_texture(mesh, rast, rast_db):
        texc, texc_db = dr.interpolate(mesh.vt[None, ...], rast, mesh.ft, rast_db=rast_db, diff_attrs='all')

        albedo = dr.texture(
            mesh.albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')  # [B, H, W, 3]
        albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device))  # remove background
        return albedo