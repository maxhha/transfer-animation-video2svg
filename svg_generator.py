import torch
import pydiffvg

# pydiffvg.set_use_gpu(torch.cuda.is_available())
# pydiffvg.set_use_gpu(False)

def rgba2rgb(img):
    return img[..., :3].permute(2, 0, 1) * img[..., 3] + (1 - img[..., 3])

class SVGGenerator(torch.nn.Module):
    def __init__(self, num_channels, num_regions):
        super().__init__()
        self.num_channels = num_channels
        self.num_regions = num_regions
        self.sigma = torch.nn.Parameter(torch.tensor(1.))
        
    def _render_svg(self, canvas_size, points, points_n, shape_groups_n, shapes_n, num_control_points, num_control_points_n, stroke_width, shape_ids, group_shape_ids, group_shape_ids_n, fill_color, use_even_odd_rule, stroke_color):
        canvas_w, canvas_h = canvas_size
        batch_size = points.shape[0]
        
        results = []

        for b in range(batch_size):
            shapes = []
            for i in range(shapes_n[b]):
                shapes.append(pydiffvg.shape.Path(
                    num_control_points = num_control_points[b, i, :num_control_points_n[b, i]],
                    points = points[b, i, :points_n[b, i]],
                    is_closed = True,
                    stroke_width = stroke_width[b, i],
                    id = shape_ids[i][b],
                ))

            shape_groups = []
            for i in range(shape_groups_n[b]):
                shape_groups.append(pydiffvg.shape.ShapeGroup(
                    shape_ids = group_shape_ids[b, i, :group_shape_ids_n[b, i]],
                    fill_color = fill_color[b, i],
                    use_even_odd_rule = use_even_odd_rule[b, i],
                    stroke_color = stroke_color[b, i]
                ))

            scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_w, canvas_h, shapes, shape_groups)

            render = pydiffvg.RenderFunction.apply
            results.append(rgba2rgb(render(canvas_w, # width
                         canvas_h, # height
                         1,   # num_samples_x
                         1,   # num_samples_y
                         0,   # seed
                         None, # background_image
                         *scene_args)))
        
        return torch.stack(results)

    def forward(self, x, driving_region_params=None, source_region_params=None, svg_params=None, **kwargs):
        points = svg_params['points']
        batch_size, shapes_size, points_size, _ = points.shape
        image_size = x.shape[-2:]
        heatmap_size = source_region_params['heatmap'].shape[-2:]
        
        points_m = points.repeat(self.num_regions, 1, 1, 1, 1).permute(2, 1, 0, 3, 4) # -> Shapes Batches Regions Points xy
        s_shift = (source_region_params['shift'] + 1) / 2 * torch.tensor(image_size)
        d_shift = (driving_region_params['shift'] + 1) / 2 * torch.tensor(image_size)
        
        affine_m = torch.matmul(source_region_params['affine'], torch.inverse(driving_region_params['affine']))
        sign = torch.sign(affine_m[:, :, 0:1, 0:1])
        affine_m *= sign
    
        points_m -= s_shift.expand(points_size, -1, -1, -1).permute(1, 2, 0, 3)        
        points_m = points_m @ affine_m
        points_m += d_shift.expand(points_size, -1, -1, -1).permute(1, 2, 0, 3)
        
        down_scale_factor = torch.tensor(heatmap_size) / torch.tensor(image_size)
        
        w, h = heatmap_size
        xx, yy = torch.arange(w), torch.arange(h)
        yy = yy.view(-1, 1).repeat(1, w)
        xx = xx.view(1, -1).repeat(h, 1)
        grid = torch.stack([xx, yy])
        
        distance_field = grid - (points * down_scale_factor).expand(w, h, -1, -1, -1, -1).permute(2, 3, 4, 5, 0, 1) # -> Batches Shapes Points xy W H
        distance_field = (distance_field**2).sum(-3) # sum by xy
        distance_field = torch.exp(-distance_field/self.sigma**2)
        
        influence_k = (source_region_params['heatmap'].expand(shapes_size, points_size, -1, -1, -1, -1).permute(3, 2, 0, 1, 4, 5) * distance_field).sum(dim=(-2, -1)) # -> Regions Batches Shapes Points 
        influence_k /= influence_k.sum(dim=0)
        
        new_points = influence_k * points_m.permute(4, 2, 1, 0, 3) # -> xy Regions Batches Shapes Points
        new_points = new_points.permute(1, 2, 3, 4, 0).sum(dim=0) # sum by regions -> Batches Shapes Points xy
        
        svg = svg_params.copy()
        svg['points'] = new_points
        
        prediction = self._render_svg(image_size, **svg)
        
        return {'prediction': prediction, 'svg': svg }
