from torch.utils.data import Dataset
from frames_dataset import read_video
import re
import numpy as np
from skimage.color import rgba2rgb
import torch
import os
import pydiffvg

# pydiffvg.set_use_gpu(torch.cuda.is_available())
# pydiffvg.set_use_gpu(False)

FILE_NAME_REG = re.compile(r'^(?P<name>.+?)(?P<index> \(\d+\))?\.(?P<type>mp4|svg)$')

class SVGFramesDataset(Dataset):
    def __init__(self, root_dir, frame_shape=(256, 256, 3), is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        self.frame_shape = frame_shape
        
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
        
        return rgba2rgb(img.numpy())

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
            driving = np.array(video_array[0], dtype='float32')
            svg_path = os.path.join(self.svg_dir, self.svg_map[video_name])
            shapes, shape_groups = self._load_svg(svg_path)
            source = self._render_svg(shapes, shape_groups)

            out['driving'] = driving.transpose((2, 0, 1))            
            out['source'] = source.transpose((2, 0, 1))
            out['shapes'] = shapes
            out['shape_groups'] = shape_groups
        else:
            raise NotImplementedError() 

        return out
