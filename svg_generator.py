import torch
import pydiffvg

def rgba2rgb(img):
    return img[..., :3].permute(2, 0, 1) * img[..., 3] + (1 - img[..., 3])

class SVGGenerator(torch.nn.Module):
    def __init__(self, num_channels, num_regions, use_cpu=False, with_raster=True, sigma=1., epsilon=1e-3):
        super().__init__()
        self.num_channels = num_channels
        self.num_regions = num_regions
        self.use_cuda = not use_cpu
        self.sigma = sigma
        self.with_raster = with_raster
        self.epsilon = epsilon

    @classmethod
    def from_svg_params(self, points, points_n, shape_groups_n, shapes_n, num_control_points, num_control_points_n, stroke_width, shape_ids, group_shape_ids, group_shape_ids_n, fill_color, use_even_odd_rule, stroke_color, **kwargs):
        batch_size = points.shape[0]

        for b in range(batch_size):
            shapes = []
            for i in range(shapes_n[b]):
                shapes.append(pydiffvg.Path(
                    num_control_points = num_control_points[b, i, :num_control_points_n[b, i]],
                    points = points[b, i, :points_n[b, i]],
                    is_closed = True,
                    stroke_width = stroke_width[b, i],
                    id = shape_ids[i][b],
                ))

            shape_groups = []
            for i in range(shape_groups_n[b]):
                shape_groups.append(pydiffvg.ShapeGroup(
                    shape_ids = group_shape_ids[b, i, :group_shape_ids_n[b, i]],
                    fill_color = fill_color[b, i],
                    use_even_odd_rule = use_even_odd_rule[b, i],
                    stroke_color = stroke_color[b, i]
                ))
            
            yield shapes, shape_groups

    def _render_svg(self, canvas_size, svg_params):
        canvas_w, canvas_h = canvas_size

        render = pydiffvg.RenderFunction.apply
        results = []

        for shapes, shape_groups in self.from_svg_params(**svg_params):
            scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_w, canvas_h, shapes, shape_groups)

            r = render(
                canvas_w, # width
                canvas_h, # height
                1,   # num_samples_x
                1,   # num_samples_y
                0,   # seed
                None, # background_image
                *scene_args
            )

            results.append(rgba2rgb(r))

        return torch.stack(results)

    def forward(self, x, driving_region_params=None, source_region_params=None, svg_params=None, **kwargs):
        
        points = svg_params['points']
        if self.use_cuda:
            points = points.cuda()
        
        batch_size, shapes_size, points_size, _ = points.shape
        image_size = x.shape[-2:]
        heatmap_size = source_region_params['heatmap'].shape[-2:]
        
        image_size_tensor = torch.tensor(image_size, device=torch.device('cuda:0' if self.use_cuda else 'cpu')) 

        points_m = points.repeat(self.num_regions, 1, 1, 1, 1).permute(2, 1, 0, 3, 4) # -> Shapes Batches Regions Points xy
        
        s_shift = source_region_params['shift']
        d_shift = driving_region_params['shift']
                
        s_shift = (s_shift + 1) / 2 * image_size_tensor
        d_shift = (d_shift + 1) / 2 * image_size_tensor
        
        s_affine = source_region_params['affine']
        d_affine = driving_region_params['affine']
        if self.use_cuda:
            s_affine = s_affine.cuda()
            d_affine = d_affine.cuda()

        affine_m = torch.matmul(s_affine, torch.inverse(d_affine))
        sign = torch.sign(affine_m[:, :, 0:1, 0:1])
        affine_m *= sign
    
        points_m -= s_shift.expand(points_size, -1, -1, -1).permute(1, 2, 0, 3)        
        points_m = points_m @ affine_m
        points_m += d_shift.expand(points_size, -1, -1, -1).permute(1, 2, 0, 3)
        
        down_scale_factor = torch.tensor(heatmap_size) / torch.tensor(image_size)
        if self.use_cuda:
            down_scale_factor = down_scale_factor.cuda()
        
        w, h = heatmap_size
        xx, yy = torch.arange(w, device=torch.device("cuda:0" if self.use_cuda else 'cpu')), torch.arange(h, device=torch.device("cuda:0" if self.use_cuda else 'cpu'))

        yy = yy.view(-1, 1).repeat(1, w)
        xx = xx.view(1, -1).repeat(h, 1)
        grid = torch.stack([xx, yy])
        
        distance_field = grid - (points * down_scale_factor).expand(w, h, -1, -1, -1, -1).permute(2, 3, 4, 5, 0, 1) # -> Batches Shapes Points xy W H
        distance_field = (distance_field**2).sum(-3) # sum by xy -> Batches Shapes Points W H
        distance_field = torch.exp(-distance_field/self.sigma**2)

        influence_k = (source_region_params['heatmap'].expand(shapes_size, points_size, -1, -1, -1, -1).permute(3, 2, 0, 1, 4, 5) * distance_field).sum(dim=(-2, -1)) # -> Regions Batches Shapes Points 
        influence_k_sum = influence_k.sum(dim=0) + self.epsilon # -> Batches Shapes Points 
        
        new_points = influence_k * points_m.permute(4, 2, 1, 0, 3) # -> xy Regions Batches Shapes Points
        new_points = new_points.permute(1, 2, 3, 4, 0).sum(dim=0) # sum by regions -> Batches Shapes Points xy
        new_points += points * self.epsilon
        new_points /= influence_k_sum[..., None]
        
        svg = svg_params.copy()
        svg['points'] = new_points

        result = {'svg': svg }

        if self.with_raster:
            prediction = self._render_svg(image_size, svg_params=svg)
            if self.use_cuda:
                prediction = prediction.cuda()
            result['prediction'] = prediction

        return result
