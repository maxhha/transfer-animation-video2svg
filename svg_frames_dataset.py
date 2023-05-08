from torch.utils.data import Dataset
from frames_dataset import read_video
import re
import numpy as np
import torch
import os
import pydiffvg

def rgba2rgb(img):
    return img[..., :3].permute(2, 0, 1) * img[..., 3] + (1 - img[..., 3])

# pydiffvg.set_use_gpu(torch.cuda.is_available())
# pydiffvg.set_use_gpu(False)

FILE_NAME_REG = re.compile(r'^(?P<name>.+?)(?P<index> \(\d+\))?\.(?P<type>mp4|svg)$')

class SVGFramesDataset(Dataset):
    def __init__(self, root_dir, frame_shape=(256, 256, 3), is_train=True, max_svg_dict=None):
        self.root_dir = root_dir
        self.is_train = is_train
        self.frame_shape = frame_shape
        
        self.max_svg_dict = {
            'shapes': 16,
            'points': 128,
            'num_control_points': 64,
            'shape_groups': 16,
            'shape_ids': 8
        }
        if max_svg_dict:
            self.max_svg_dict.update(max_svg_dict)
        
        self.video_dir = os.path.join(root_dir, 'video')
        self.svg_dir = os.path.join(root_dir, 'svg')
    
        self.videos = os.listdir(self.video_dir)
        self.svg_map = {
            video_name: FILE_NAME_REG.match(video_name).group('name')+'.svg'
            for video_name in self.videos
        }
        set_needed_svgs = set(self.svg_map.values())
        set_existing_svgs = set(os.listdir(self.svg_dir))
        assert set_needed_svgs.issubset(set_existing_svgs), f'Missing svgs: {",".join(s.et_needed_svgs.difference(set_existing_svgs))}'

    def __len__(self):
        return len(self.videos)

    @torch.no_grad()
    def _load_svg(self, svg_file_name):
        canvas_w, canvas_h = self.frame_shape[:2]
        orig_w, orig_h, shapes, shape_groups = pydiffvg.svg_to_scene(svg_file_name)

        scale = min(canvas_w / orig_w, canvas_h / orig_h)
        for s in shapes:
            s.points *= scale
            s.stroke_width *= scale

        return shapes, shape_groups
    
    @torch.no_grad()
    def _render_svg(self, shapes, shape_groups):
        canvas_w, canvas_h = self.frame_shape[:2]
        
        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_w, canvas_h, shapes, shape_groups)
        
        render = pydiffvg.RenderFunction.apply
        img = render(canvas_w, # width
                     canvas_h, # height
                     1,   # num_samples_x
                     1,   # num_samples_y
                     0,   # seed
                     None, # background_image
                     *scene_args)
        
        return rgba2rgb(img).numpy()

    @torch.no_grad()
    def _svg_params(self, shapes, shape_groups, svg_name):
        all_points = torch.zeros(self.max_svg_dict['shapes'], self.max_svg_dict['points'], 2, dtype=torch.float32)
        all_points_n = torch.zeros(self.max_svg_dict['shapes'], dtype=torch.int32)
        all_num_control_points = torch.zeros(max_svg_dict['shapes'], max_svg_dict['num_control_points'], dtype=torch.int32)
        all_num_control_points_n = torch.zeros(max_svg_dict['shapes'], dtype=torch.int32)
        all_stroke_width = torch.zeros(max_svg_dict['shapes'], dtype=torch.float32)
        all_ids = ['' for _ in range(max_svg_dict['shapes'])]

        assert len(shapes) <= self.max_svg_dict['shapes']
        for shape_i, shape in enumerate(shapes):
            assert isinstance(shape, (pydiffvg.shape.Path,)), f"Supports only path tags in svg, but get {shape.__class__.__name__} in {svg_name}"
            assert shape.is_closed, f"All paths must be closed in {svg_name}"
            assert shape.points.shape[0] <= self.max_svg_dict['points'], f"points in path({shape.points.shape[0]}) greater than size({self.max_svg_dict['points']}) in {svg_name}"

            all_points_n[shape_i] = n = shape.points.shape[0]
            all_points[shape_i, :n, ...] = shape.points
            all_num_control_points_n[shape_i] = n = shape.num_control_points.shape[0]
            all_num_control_points[shape_i, :n, ...] = shape.num_control_points
            all_stroke_width[shape_i] = shape.stroke_width
            all_ids[shape_i] = shape.id
        
        all_shape_ids = torch.zeros(max_svg_dict['shape_groups'], max_svg_dict['shape_ids'], dtype=torch.int32)
        all_shape_ids_n = torch.zeros(max_svg_dict['shape_groups'], dtype=torch.int32)
        all_fill_color = torch.zeros(max_svg_dict['shape_groups'], 4, dtype=torch.float32)
        all_use_even_odd_rule = torch.zeros(max_svg_dict['shape_groups'], dtype=torch.bool)
        all_stroke_color = torch.zeros(max_svg_dict['shape_groups'], 4, dtype=torch.float32)
        
        assert len(shape_groups) <= self.max_svg_dict['shape_groups']
        for g_i, g in enumerate(shape_groups):
            assert g.shape_ids.shape[0] <= self.max_svg_dict['shape_ids']
            
            all_shape_ids_n[g_i] = n = g.shape_ids.shape[0]
            all_shape_ids[g_i, :n] = g.shape_ids
            all_fill_color[g_i] = g.fill_color
            all_use_even_odd_rule[g_i] = g.use_even_odd_rule
            all_stroke_color[g_i] = torch.tensor([0.0, 0.0, 0.0, 1.0]) if g.stroke_color is None else g.stroke_color

        return {
            "shapes_n": len(shapes),
            "shape_groups_n": len(shape_groups),
            "points": all_points,
            "points_n": all_points_n,
            "num_control_points": all_num_control_points,
            "num_control_points_n": all_num_control_points_n,
            "stroke_width": all_stroke_width,
            "shape_ids": all_ids,
            "group_shape_ids": all_shape_ids,
            "group_shape_ids_n": all_shape_ids_n,
            "fill_color": all_fill_color,
            "use_even_odd_rule": all_use_even_odd_rule,
            "stroke_color": all_stroke_color
        }

    def __getitem__(self, idx):
        video_name = self.videos[idx]
        video_path = os.path.join(self.video_dir, video_name)

        video_array = read_video(video_path, frame_shape=self.frame_shape)
        num_frames = len(video_array)

        if self.is_train:
            frame_idx = np.random.choice(num_frames, replace=True, size=1)
        else:
            frame_idx = range(num_frames)
            

        out = {}
        if self.is_train:
            driving = np.array(video_array[0], dtype='float32').transpose((2, 0, 1))
            svg_name = self.svg_map[video_name]
            svg_path = os.path.join(self.svg_dir, svg_name)
            shapes, shape_groups = self._load_svg(svg_path)
            source = self._render_svg(shapes, shape_groups)

            out['driving'] = driving           
            out['source'] = source
#             out['shapes'] = shapes
            out['svg_name'] = svg_name
#             out['shape_groups'] = shape_groups
            out['svg'] = self._svg_params(shapes, shape_groups, svg_name)
        else:
            raise NotImplementedError() 

        return out
