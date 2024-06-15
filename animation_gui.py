import os
import cv2
import time
import tqdm
import pickle as pkl
import json
import numpy as np
import dearpygui.dearpygui as dpg 
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
import kiui
from kiui.mesh import Mesh

import smplx 
from utils.cam_utils import Camera 
from utils.render import Renderer
from utils.geo import subdivide_inorder, compute_normal
from utils.common import warp_points, make_rotate
 
  
class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = Camera()

        self.mode = "image"  

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.buffer_overlay = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.buffer_out = None  # for 2D to 3D projection

        self.need_update = True  # update buffer_image 

        self.mouse_loc = np.array([0, 0])

        self.idx = 0 
        self.motion_mode = 'canon'  # canon, posed, dancing   
        self.auto_rotate = True # auto rotate the camera
   
        self.device = torch.device("cuda")

        # renderer
        self.renderer = Renderer(gui=True)
  
        # load smplx motion
        mapping = list(open(f"./data/aist/cameras/mapping.txt", 'r').read().splitlines())
        self.motion_setting_dict = {}
        for pairs in mapping:
            motion, setting = pairs.split(" ")
            self.motion_setting_dict[motion] = setting  
        self.load_motions(opt.motion_file)

        # load smplx model 
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
            num_betas=300, 
            ext='npz', 
            use_pca=False
        )
        self.smpl_model = smplx.create(**model_init_params).to(self.device)
        subdivide_params = np.load("./data/smplx/subdivide.npy", allow_pickle=True).item()
        self.sub_smplx_faces = subdivide_params['sub_smplx_faces']
        self.sub_lbs_weights = subdivide_params['sub_lbs_weights']
        self.sub_uniques = subdivide_params['sub_uniques']
        self.vt = torch.tensor(subdivide_params['vt']).cuda()
        self.ft = torch.tensor(subdivide_params['ft']).int().cuda()

        self.smplx_params = np.load(opt.smplx_params, allow_pickle=True).item()
        self.betas = torch.tensor(self.smplx_params['betas']).float().to(self.device)
        self.body_pose = torch.tensor(self.smplx_params['body_pose']).float().to(self.device)
        self.expression = torch.tensor(self.smplx_params['expression']).float().to(self.device)
        self.v_offsets = torch.tensor(self.smplx_params['v_offsets']).float().to(self.device)

        # load albedo
        albedo = cv2.imread(os.path.join(os.path.dirname(opt.smplx_params), "canon_albedo.png"))    
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB) / 255.0  # BGR to RGB
        self.albedo = torch.tensor(albedo).float().to(self.device)

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()
    
    def load_motions(self, motion_file):
        motion_name = motion_file.split("/")[-1]
        smpl_data = pkl.load(open(motion_file, 'rb'))
        poses = smpl_data['smpl_poses']  # (N, 24, 3)
        self.scale = smpl_data['smpl_scaling'][0]  # (1,)
        trans = smpl_data['smpl_trans']  # (N, 3)
        self.poses = torch.from_numpy(poses).view(-1, 24, 3).float() 
        self.trans = torch.from_numpy(trans).cuda().float() 

        # load aist camera  
        setting = self.motion_setting_dict[motion_name[:-4]]
        camera_path = open(f"./data/aist/cameras/{setting}.json", 'r')
        camera_params = json.load(camera_path)[0]
        rvec = np.array(camera_params['rotation'])
        tvec = np.array(camera_params['translation'])
        matrix = np.array(camera_params['matrix']).reshape((3, 3))
        distortions = np.array(camera_params['distortions'])

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        w2c = np.eye(4)
        w2c[:3, :3] = rotation_matrix
        w2c[:3, 3] = tvec.flatten()  
        
        fx = matrix[0, 0]
        fy = matrix[1, 1]
        cx = matrix[0, 2]
        cy = matrix[1, 2]
        w, h = 1080, 1920
        project_mat = np.array([
            [fx, 0,  cx, 0],
            [0,  fy, cy, 0],
            [0,  0,  1,  0], 
            [0,  0,  0,  1]
        ])
        
        mvp = project_mat @ w2c 
        self.mvp = torch.tensor(project_mat @ w2c).float().to(self.device)  


    @property
    def mesh(self):  
        global_orient = torch.tensor([0, 0, 0], device=self.device).float().unsqueeze(0)

        if self.motion_mode == 'canon': 
            body_pose = torch.zeros_like(self.body_pose)
        elif self.motion_mode == 'posed': 
            body_pose = self.body_pose  
        else: 
            idx = self.idx // 2 % len(self.poses) 
            body_pose = torch.as_tensor(self.poses[idx, None, 1:22].view(1, 21, 3), device=self.device)
            global_orient = torch.as_tensor(self.poses[idx, None, :1], device=self.device)
             
        output = self.smpl_model(
            betas=self.betas, 
            global_orient=global_orient, 
            body_pose=body_pose, 
            expression=self.expression,
            return_verts=True) 
        v_cano = output.v_posed[0] 
        transform = output.joints_transform[:, :55] 
         
        # remeshing 
        for sub_f, sub_uniq in zip(self.sub_smplx_faces, self.sub_uniques):
            v_cano = subdivide_inorder(v_cano, sub_f, sub_uniq)
      
        # add offset  
        if self.v_offsets.shape[1] == 1: 
            v_cano = v_cano + self.v_offsets * self.smplx_v_cano_nml 
        else:
            v_cano = v_cano + self.v_offsets
         
        # cano-to-posed
        v_posed = warp_points(v_cano, self.sub_lbs_weights[-1], transform).squeeze(0)
        vn = compute_normal(v_posed, self.sub_smplx_faces[-1])[0]

        if self.motion_mode in ['canon', 'posed']:   
            if self.auto_rotate:
                R = make_rotate(np.radians(180), 0, 0) @ make_rotate(0, np.radians(self.idx), 0) 
            else:
                R = make_rotate(np.radians(180+self.cam.theta), 0, 0) @ make_rotate(0, np.radians(self.cam.phi), 0)  
            v_posed = v_posed @ torch.tensor(R).float().to(self.device).T
            v_posed[:, 1] -= 0.35 
            smplx_mesh = Mesh(v=v_posed, f=self.sub_smplx_faces[-1], vn=vn, vt=self.vt, ft=self.ft, albedo=self.albedo)
            return smplx_mesh
 
        vn = compute_normal(v_posed, self.sub_smplx_faces[-1])[0]

        # world to image coordinate 
        posed_points = v_posed * self.scale + self.trans[idx:idx+1, :] 
        posed_points = F.pad(posed_points, (0, 1), mode='constant', value=1)
        posed_points = torch.matmul(posed_points, self.mvp.T)[:, :3]
        posed_points[:, :2] = posed_points[:, :2] / posed_points[:, 2:3] 

        # image to ndc coordinates
        ndc_points = posed_points 
        ndc_points[:, :2] = ndc_points[:, :2] / 1920 * 2 - 1 
        ndc_points[:, 1] += 0.5 * 1080 / 1920 
        ndc_points[:, :2] *= 1920 / 512 
        zmax, zmin = ndc_points[:, 2].max(), ndc_points[:, 2].min()
        ndc_points[:, 2] = (ndc_points[:, 2] - zmin) / (zmax - zmin) * 2 - 1
 
        smplx_mesh = Mesh(v=ndc_points, f=self.sub_smplx_faces[-1], vn=vn, vt=self.vt, ft=self.ft, albedo=self.albedo)
        return smplx_mesh 
          
    def __del__(self):
        if self.gui:
            dpg.destroy_context()
   
    @torch.no_grad()
    def test_step(self):  
        self.idx = (self.idx + 1)  

        # ignore if no need to update
        if not self.need_update:
            return
 
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
 
        # render image    
        out = self.renderer(self.mesh, torch.eye(4).unsqueeze(0).to(self.device), self.H, self.W)
        buffer_image = out[self.mode]  # [H, W, 3]

        if not self.mode == 'alpha':
            buffer_image = buffer_image * out['alpha'] + (1 - out['alpha'])  
          
        if self.mode in ['depth', 'alpha']:
            buffer_image = buffer_image.repeat(1, 1, 1, 3)
            if self.mode == 'depth':
                buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)
            
        self.buffer_image = buffer_image.contiguous().clamp(0, 1).detach().cpu().numpy()

        self.buffer_out = out

        # self.need_update = False
          
        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui: 
            # mix image and overlay
            buffer = np.clip(
                self.buffer_image + self.buffer_overlay, 0, 1
            )  # mix mode, sometimes unclear

            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", buffer
            )  # buffer must be contiguous, else seg fault!
  
    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")
  
            # rendering options
            with dpg.collapsing_header(label="Render", default_open=True):
                
                # auto rotate camera 
                with dpg.group(horizontal=True):
                    def callback_toggle_auto_rotate(sender, app_data):
                            self.auto_rotate = not self.auto_rotate
                            self.need_update = True
                    dpg.add_checkbox(
                        label="auto rotate",
                        default_value=self.auto_rotate,
                        callback=callback_toggle_auto_rotate,
                    )

                # input motion file 
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        # dpg.set_value("_log_input", k) 
                        self.load_motions(v)
            
                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension(".*")
                    dpg.add_file_extension("*")
                    dpg.add_file_extension("motions{.pkl,}", color=(0, 255, 0, 255))
 
                # with dpg.group(horizontal=True):
                #     dpg.add_button(
                #         label="Motion Selector",
                #         callback=lambda: dpg.show_item("file_dialog_tag"),
                #     )
                #     dpg.add_text("", tag="_log_input")

                # render mode combo
                def callback_change_rendering_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True
                    self.idx = 0 

                dpg.add_combo(
                    ("image", "depth", "alpha", "normal"),
                    label="render mode",
                    default_value=self.mode,
                    callback=callback_change_rendering_mode,
                )

                # motion mode combo
                def callback_change_motion_mode(sender, app_data):
                    self.motion_mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("canon", "posed", "dancing"),
                    label="motion mode",
                    default_value=self.motion_mode,
                    callback=callback_change_motion_mode,
                )
   
        ### register camera handler
        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]
 
            self.cam.orbit(dx, dy)
            self.need_update = True 

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)
 
        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            ) 
            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc) 
 

        dpg.create_viewport(
            title="hanimaker",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            self.test_step()
            dpg.render_dearpygui_frame()
     

# python viewer.py --smplx_params out/0525/smplx_param.npy
if __name__ == "__main__":
    import argparse 
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--H', type=int, default=800, help='gui windown height') 
    parser.add_argument('--W', type=int, default=800, help="gui window width")
    parser.add_argument('--smplx_params', type=str, required=True, help="file to the smplx path")
    parser.add_argument('--gui', type=bool, default=True, help="gui mode or not")
    parser.add_argument('--motion_file', type=str, default="./data/aist/motions/gBR_sBM_cAll_d06_mBR3_ch06.pkl", help="Path to aist motion file") 

    opt = parser.parse_args()
      
    gui = GUI(opt)
    gui.render()
 