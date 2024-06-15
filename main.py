import os
import cv2
import random
import pickle
import trimesh
import torch
import torch.nn as nn
import smplx as smplx  
import numpy as np  
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing, mesh_edge_loss, mesh_normal_consistency
from pytorch3d.ops.knn import knn_points
from pytorch3d.structures import Meshes  

from utils.common import warp_points, inverse_sigmoid, make_rotate
from utils.geo import * 
from utils.render import Renderer

from kiui.mesh import Mesh
     


class Trainer(nn.Module):
    def __init__(self, opt, num_betas=300):
        super(Trainer, self).__init__()  
        self.num_betas = num_betas
        self.albedo_res = opt.albedo_res 
        self.debug = opt.debug
        self.smplx_file = opt.smplx_file
        self.save_dir = opt.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
 
        self.renderer = Renderer()  
        self.offset = torch.tensor([0.0, 0.4, 0.0]).cuda() 

        # load smpl uv 
        mesh = Mesh.load("data/smplx/smplx_uv.obj", resize=False)
        # mesh = trimesh.load_mesh("data/smplx/smplx_uv.obj", process=False, maintain_order=True)
        self.vt = mesh.vt.cpu().numpy()
        self.ft = mesh.ft.cpu().numpy()

        model_init_params = dict(
            model_path="./data/smplx/SMPLX_NEUTRAL_2020.npz", 
            model_type='smplx', 
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            create_expression=True,
            create_jaw_pose=True,
            create_leye_pose=True,
            create_reye_pose=True,
            create_transl=True,
            num_pca_comps=45,
            num_betas=num_betas, 
            ext='npz', 
            use_pca=False
        )
        self.smpl_model = smplx.create(**model_init_params).cuda()  
        self.smplx_faces = torch.tensor(self.smpl_model.faces.astype(np.int32)) 
        self.remesh_mask = self.get_remesh_mask() 
        self.load_smplx_body(opt.smplx_file)
        self.load_scan(opt.obj_file)

        # init optimizer
        self.init_geo_optimizer()

        if self.debug:
            os.makedirs(f"{self.save_dir}/vis", exist_ok=True)
 
    def init_geo_optimizer(self):
        self.v_offsets = nn.Parameter(torch.zeros(self.N, 3).cuda())
        self.betas = nn.Parameter(torch.zeros(1, self.num_betas).cuda())  
        self.albedo = nn.Parameter(torch.ones(self.albedo_res, self.albedo_res, 3).cuda())

        params = [
            {"params": self.v_offsets, "lr": 0.0001},
            {"params": self.betas, "lr": 0.001}
        ]
        self.optimizer = torch.optim.Adam(params, lr=0.0001)

    def init_albedo_optimizer(self):  
        self.optimizer = torch.optim.Adam([self.albedo], lr=0.01)


    def get_remesh_mask(self):
        ids = list(set(SMPLXSeg.front_face_ids) - set(SMPLXSeg.forehead_ids))
        ids = ids + SMPLXSeg.ears_ids + SMPLXSeg.eyeball_ids + SMPLXSeg.hands_ids
        mask = ~np.isin(np.arange(10475), ids)
        mask = mask[self.smpl_model.faces].all(axis=1)
        return mask
     
    def load_smplx_body(self, file_path, subdevide_cache="./data/smplx/subdivide.npy"):
        param = np.load(file_path, allow_pickle=True) 
        for key in param.keys():
            param[key] = torch.as_tensor(param[key]).float().cuda()
        
        self.root_transform = self.smpl_model(global_orient=param['global_orient']).joints_transform[:, 0]
        self.smplx_scale = param['scale']
        self.smplx_transl = param['transl'] 
        self.smplx_params = dict(
            body_pose=param['body_pose'],
            left_hand_pose=param['left_hand_pose'],
            right_hand_pose=param['right_hand_pose'],
            jaw_pose=param['jaw_pose'],
            leye_pose=param['leye_pose'],
            reye_pose=param['reye_pose'],
            expression=param['expression'],  
            return_verts=True
        )
        output = self.smpl_model(**self.smplx_params) 
        v_cano = output.v_posed[0].detach().cpu().numpy() 

        if os.path.exists(subdevide_cache):
            # remesh 
            # self.sub_smplx_faces = [self.smplx_faces[self.remesh_mask]]
            self.sub_smplx_faces = [self.smplx_faces]
            self.sub_lbs_weights = [self.smpl_model.lbs_weights.detach()]
            self.sub_uniques = [] 

            # first-subdivide, do not subdived front head area  
            v_cano, smplx_faces, lbs_weights, unique = subdivide(v_cano, self.sub_smplx_faces[0].cpu().numpy(), self.sub_lbs_weights[0].cpu().numpy()) 
            # vt, ft, _ = subdivide(self.vt, self.ft[self.remesh_mask])
            # ft = np.concatenate([ft, self.ft[~self.remesh_mask]]) 
            # smplx_faces = np.concatenate([smplx_faces, self.smplx_faces[~self.remesh_mask]])

            vt, ft, _ = subdivide(self.vt, self.ft)
             
            self.sub_smplx_faces.append(torch.tensor(smplx_faces).int().cuda())
            self.sub_lbs_weights.append(torch.tensor(lbs_weights).float().cuda())
            self.sub_uniques.append(torch.tensor(unique).long().cuda())

            # second-subdivide
            for i in range(1):
                v_cano, smplx_faces, lbs_weights, unique = subdivide(v_cano, smplx_faces, lbs_weights)
                vt, ft, _ = subdivide(vt, ft)

                self.sub_smplx_faces.append(torch.tensor(smplx_faces).int().cuda())
                self.sub_lbs_weights.append(torch.tensor(lbs_weights).float().cuda())
                self.sub_uniques.append(torch.tensor(unique).long().cuda())
            self.N = v_cano.shape[0]
            self.vt = torch.tensor(vt).cuda()
            self.ft = torch.tensor(ft).int().cuda()

            subdivide_params = dict(
                sub_smplx_faces=self.sub_smplx_faces,
                sub_lbs_weights=self.sub_lbs_weights,
                sub_uniques=self.sub_uniques,
                N=self.N, 
                vt=self.vt.cpu().numpy(),
                ft=self.ft.cpu().numpy()
            )
            os.makedirs(os.path.dirname(subdevide_cache), exist_ok=True)
            np.save(subdevide_cache, subdivide_params)
        else:
            subdivide_params = np.load(subdevide_cache, allow_pickle=True).item()
            self.sub_smplx_faces = subdivide_params['sub_smplx_faces']
            self.sub_lbs_weights = subdivide_params['sub_lbs_weights']
            self.sub_uniques = subdivide_params['sub_uniques']
            self.N = subdivide_params['N']
            self.vt = torch.tensor(subdivide_params['vt']).cuda()
            self.ft = torch.tensor(subdivide_params['ft']).int().cuda()
        

    @torch.no_grad()
    def load_scan(self, scan_path):
        self.mesh = Mesh.load(scan_path, resize=False)  
        self.mesh.v = self.mesh.v / self.smplx_scale - self.smplx_transl 
         
        # remove global transform
        R, T = self.root_transform[:, :3, :3], self.root_transform[:, :3, 3]
        inverse_R = torch.inverse(R)
        self.mesh.v = (torch.bmm(self.mesh.v[None].cuda().float() - T, inverse_R.transpose(1, 2))).squeeze(0)

        # shift global offset 
        self.mesh.v += self.offset
        self.mesh.auto_normal()
        # render
        # self.scane_images, self.scane_normals, self.scan_alpha = self.render(self.mesh)
    
    def get_vertex(self, return_canonical=False, replace_hands_eyes=False):  
        # forward shape 
        output = self.smpl_model(betas=self.betas, **self.smplx_params) 
        v_cano = output.v_posed[0] 
        transform = output.joints_transform[:, :55] 

        # remeshing 
        for sub_f, sub_uniq in zip(self.sub_smplx_faces, self.sub_uniques):
            v_cano = subdivide_inorder(v_cano, sub_f, sub_uniq)
     
        # v_offsets = self.v_offsets
        # if replace_hands_eyes: # set hands and eyes offset=0 
        #     v_offsets[SMPLXSeg.eyeball_ids] = 0 
        #     v_offsets[SMPLXSeg.hands_ids] = 0 

        # add offset  
        if self.v_offsets.shape[1] == 1: 
            v_cano = v_cano + self.v_offsets * self.smplx_v_cano_nml 
        else:
            v_cano = v_cano + self.v_offsets
         
        # warp to target pose  
        v_posed = warp_points(v_cano, self.sub_lbs_weights[-1], transform).squeeze(0)
        
        if return_canonical:
            return v_posed, v_cano

        return v_posed 
    
    def render(self, mesh, thetas, phis):
        images, normals, alphas = [], [], []  
        for theta, phi in zip(thetas, phis):
            mvp = torch.eye(4)[None].cuda() 
            R = make_rotate(np.radians(180+theta), 0, 0) @ make_rotate(0, np.radians(phi), 0)
           
            mvp[:, :3, :3] = torch.tensor(R).float().cuda() 
            render_pkg = self.renderer(mesh, mvp, mode='rgb', spp=2) 
            images.append(render_pkg["image"][0])
            normals.append(render_pkg["normal"][0])
            alphas.append(render_pkg["alpha"][0])
        images = torch.cat(images, 1)
        normals = torch.cat(normals, 1)
        alphas = torch.cat(alphas, 1)
        # cv2.imwrite(f"normal.png", (normal.detach().cpu().numpy()[0, ..., ::-1]*255).astype(np.uint8))
        # exit()
        return images, normals, alphas 

    def train(self, iters=1000, random_view=True):
        pbar = tqdm(range(iters))
        for i in pbar: 
            smplx_v = self.get_vertex() + self.offset 

            # renderer 
            smplx_mesh = Mesh(v=smplx_v, f=self.sub_smplx_faces[-1], vt=self.vt, ft=self.ft, albedo=self.albedo.clamp(0, 1))
            smplx_mesh.auto_normal() 

            phis = random.randint(0, 90) + np.array([0, 90, 180, 270]) 
            thetas = random.sample(range(-30, 30), 4)
 
            smplx_images, smplx_normals, smplx_mask = self.render(smplx_mesh, thetas, phis) 
            self.scane_images, self.scane_normals, self.scan_mask = self.render(self.mesh,  thetas, phis)

            if i < 500:
                # reconstruction loss 
                chamfer_loss = chamfer_distance(smplx_v.unsqueeze(0), self.mesh.v.unsqueeze(0))[0] * 0.1 
                normal_render_loss = torch.mean((smplx_normals - self.scane_normals) ** 2) * 0.1
                mask_loss = torch.mean((smplx_mask - self.scan_mask) ** 2)   
                
                # regularization
                mesh = Meshes(verts=smplx_v.unsqueeze(0), faces=smplx_mesh.f.unsqueeze(0))
                lap_loss = mesh_laplacian_smoothing(mesh, method="uniform") * 1e-1
                edge_loss = mesh_edge_loss(mesh) * 10  
                normal_loss = mesh_normal_consistency(mesh) * 2e-2  
                
                loss = chamfer_loss + normal_loss + lap_loss + edge_loss + normal_render_loss  
            
            else:
                if i == 500: 
                    self.init_albedo_optimizer() 
                loss = torch.mean((smplx_images - self.scane_images) ** 2)  * 0.1

            if self.debug and i % 10 == 0:
                vis_rgb = torch.cat([smplx_images, self.scane_images], 0).detach().cpu().numpy()
                vis_nml = torch.cat([smplx_normals, self.scane_normals], 0).detach().cpu().numpy()
                vis = np.concatenate([vis_rgb, vis_nml], 1)
                cv2.imwrite(f"{self.save_dir}/vis/{i:04}.png", (vis[..., ::-1]*255).astype(np.uint8))

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            pbar.set_description(
                f"Chamf: {chamfer_loss.item():.4f} Lap: {lap_loss.item():.4f} Edge: {edge_loss.item():.6f} NC: {normal_loss.item():.4f} NR: {normal_render_loss.item():.4f}"
            )

        self.save_results()

    @torch.no_grad()
    def save_results(self): 
        smplx_v_pose, smplx_v_canon = self.get_vertex(return_canonical=True, replace_hands_eyes=False)
        smplx_f = self.sub_smplx_faces[-1].cpu().numpy() 

        # save canonical mesh 
        cano_mesh = Mesh(v=smplx_v_canon, f=self.sub_smplx_faces[-1], vt=self.vt, ft=self.ft, albedo=self.albedo.clamp(0, 1))
        cano_mesh.auto_normal()
        cano_mesh.write(f"{self.save_dir}/canon.obj")

        # save posed mesh
        posed_mesh = Mesh(v=smplx_v_pose, f=self.sub_smplx_faces[-1], vt=self.vt, ft=self.ft, albedo=self.albedo.clamp(0, 1))
        posed_mesh.auto_normal()
        posed_mesh.write(f"{self.save_dir}/posed.obj")
        
        # save lbs weights
        _, idx, _ = knn_points(self.mesh.v.unsqueeze(0), smplx_v_pose.unsqueeze(0), K=1)  
        lbs_weights = self.sub_lbs_weights[-1][idx.squeeze()] 

        # save smplx param
        smplx_param = np.load(self.smplx_file, allow_pickle=True) 
        smplx_param.update(
            betas=self.betas.cpu().numpy(), 
            v_offsets=self.v_offsets.cpu().numpy(), 
        )
 
        np.save(f"{self.save_dir}/smplx_param.npy", smplx_param)
  
        # inverse lbs 
        # mesh_v_cano = warp_points(self.mesh_v, lbs_weights, self.smplx_jT, inverse=True)[0]
        # print(mesh_v_cano.shape, self.mesh_f.shape)
        # trimesh.Trimesh(vertices=mesh_v_cano.cpu().detach().numpy(), faces=self.mesh_f.cpu().detach().numpy()).export('output_scan_cano.obj')
          
        # # rm large-edge-length triangles  
        # valid_tri_mask = remove_fly_triangles(canon_mesh.vertices, posed_mesh.vertices, canon_mesh.faces)
        # valid_tri = canon_mesh.faces[valid_tri_mask] 
        # trimesh.Trimesh(vertices=posed_mesh.vertices, faces=valid_tri).export(f'{save_dir}/output_posed_processed.obj')
        # trimesh.Trimesh(vertices=canon_mesh.vertices, faces=valid_tri).export(f'{save_dir}/output_cano_processed.obj')
        
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj', '--obj_file', type=str, required=True, help='Path to 3D scan obj file') 
    parser.add_argument('-smpl', '--smplx_file', type=str, required=True, help="Path to smplx param file")
    parser.add_argument('-out', '--save_dir', type=str, default="./out", help="Output directory")
    parser.add_argument('-albedo', '--albedo_res', type=int, default=1024, help="Albedo resolution")
    parser.add_argument('--debug', action='store_true', help="Debug mode")
    opt = parser.parse_args()
  
    trainer = Trainer(opt)
    trainer.train()  
   
     