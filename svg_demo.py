import numpy as np
import xml.etree.ElementTree as etree
import torch
import pydiffvg
from scipy.spatial import ConvexHull
from tqdm.notebook import tqdm
from xml.dom import minidom
from svg_frames_dataset import SVGFramesDataset

def svg_string_to_scene(svg_str):
    root = etree.fromstring(svg_str)
    ret = pydiffvg.parse_scene(root)
    return ret

def get_animation_region_params(source_region_params, driving_region_params, driving_region_params_initial,
                                mode='standard', avd_network=None, adapt_movement_scale=True):
    assert mode in ['standard', 'relative', 'avd']
    new_region_params = {k: v for k, v in driving_region_params.items()}
    if mode == 'standard':
        return new_region_params
    elif mode == 'relative':
        source_area = ConvexHull(source_region_params['shift'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(driving_region_params_initial['shift'][0].data.cpu().numpy()).volume
        movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

        shift_diff = (driving_region_params['shift'] - driving_region_params_initial['shift'])
        shift_diff *= movement_scale
        new_region_params['shift'] = shift_diff + source_region_params['shift']

        affine_diff = torch.matmul(driving_region_params['affine'],
                                   torch.inverse(driving_region_params_initial['affine']))
        new_region_params['affine'] = torch.matmul(affine_diff, source_region_params['affine'])
        return new_region_params
    elif mode == 'avd':
        new_region_params = avd_network(source_region_params, driving_region_params)
        return new_region_params

@torch.no_grad()
def make_animation(canvas_w, canvas_h, shapes, shape_groups, svg_name, region_predictor, generator, animation_mode='relative'):
    source_image = SVGFramesDataset.render_svg(shapes, shape_groups, (canvas_w, canvas_h)).numpy()
    max_svg_dict = SVGFramesDataset.get_max_svg_dict(shapes, shape_groups, svg_name)
    svg_params = SVGFramesDataset.svg_params(shapes, shape_groups, svg_name, max_svg_dict)

    source = torch.tensor(source_image[np.newaxis].astype(np.float32))
    driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
    svg_params = {
        k: v[None, ...] if isinstance(v, torch.Tensor) 
        else [[i] for i in v] if isinstance(v, list) 
        else [v]
        for k, v in svg_params.items()
    }
    source_region_params = region_predictor(source)
    driving_region_params_initial = region_predictor(driving[:, :, 0])
    predictions = []
    for frame_idx in tqdm(range(driving.shape[2])):
        driving_frame = driving[:, :, frame_idx]
        driving_region_params = region_predictor(driving_frame)
        new_region_params = get_animation_region_params(source_region_params, driving_region_params,
                                                        driving_region_params_initial,
                                                        mode=animation_mode)
        out = generator(source, source_region_params=source_region_params, driving_region_params=new_region_params, svg_params=svg_params)

        predictions.append(out['svg'])
    
    return predictions

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = etree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def shape2d(shape):
    num_segments = shape.num_control_points.shape[0]
    num_control_points = shape.num_control_points.data.cpu().numpy()
    points = shape.points.data.cpu().numpy()
    num_points = shape.points.shape[0]
    path_str = 'M {} {}'.format(points[0, 0], points[0, 1])
    point_id = 1
    for j in range(0, num_segments):
        if num_control_points[j] == 0:
            p = point_id % num_points
            path_str += ' L {} {}'.format(\
                    points[p, 0], points[p, 1])
            point_id += 1
        elif num_control_points[j] == 1:
            p1 = (point_id + 1) % num_points
            path_str += ' Q {} {} {} {}'.format(\
                    points[point_id, 0], points[point_id, 1],
                    points[p1, 0], points[p1, 1])
            point_id += 2
        elif num_control_points[j] == 2:
            p2 = (point_id + 2) % num_points
            path_str += ' C {} {} {} {} {} {}'.format(\
                    points[point_id, 0], points[point_id, 1],
                    points[point_id + 1, 0], points[point_id + 1, 1],
                    points[p2, 0], points[p2, 1])
            point_id += 3

    return path_str

def save_svg(width, height, shapes, shape_groups, animated_shapes):
    root = etree.Element('svg')
    root.set('version', '1.1')
    root.set('xmlns', 'http://www.w3.org/2000/svg')
    root.set('width', str(width))
    root.set('height', str(height))
    defs = etree.SubElement(root, 'defs')
    g = etree.SubElement(root, 'g')
   
    # Store color
    for i, shape_group in enumerate(shape_groups):
        def add_color(shape_color, name):
            if isinstance(shape_color, pydiffvg.LinearGradient):
                lg = shape_color
                color = etree.SubElement(defs, 'linearGradient')
                color.set('id', name)
                color.set('x1', str(lg.begin[0].item()))
                color.set('y1', str(lg.begin[1].item()))
                color.set('x2', str(lg.end[0].item()))
                color.set('y2', str(lg.end[1].item()))
                offsets = lg.offsets.data.cpu().numpy()
                stop_colors = lg.stop_colors.data.cpu().numpy()
                for j in range(offsets.shape[0]):
                    stop = etree.SubElement(color, 'stop')
                    stop.set('offset', str(offsets[j]))
                    c = lg.stop_colors[j, :]
                    stop.set('stop-color', 'rgb({}, {}, {})'.format(\
                        int(255 * c[0]), int(255 * c[1]), int(255 * c[2])))
                    stop.set('stop-opacity', '{}'.format(c[3]))

        if shape_group.fill_color is not None:
            add_color(shape_group.fill_color, 'shape_{}_fill'.format(i))
        if shape_group.stroke_color is not None:
            add_color(shape_group.stroke_color, 'shape_{}_stroke'.format(i))
    
    for i, shape_group in enumerate(tqdm(shape_groups)):
        shape_node = etree.SubElement(g, 'path')
        if shape_group.use_even_odd_rule:
            shape_node.set('style', "fill-rule: evenodd")
        
        path_str = []
        for shape_i in shape_group.shape_ids:
            shape = shapes[shape_i]
            assert isinstance(shape, pydiffvg.Path)
            path_str.append(shape2d(shape))
        
        shape_node.set('d', ' '.join(path_str))
            
#             if isinstance(shape, pydiffvg.Circle):
#                 shape_node = etree.SubElement(g, 'circle')
#                 shape_node.set('r', str(shape.radius.item()))
#                 shape_node.set('cx', str(shape.center[0].item()))
#                 shape_node.set('cy', str(shape.center[1].item()))
#             elif isinstance(shape, pydiffvg.Polygon):
#                 shape_node = etree.SubElement(g, 'polygon')
#                 points = shape.points.data.cpu().numpy()
#                 path_str = ''
#                 for j in range(0, shape.points.shape[0]):
#                     path_str += '{} {}'.format(points[j, 0], points[j, 1])
#                     if j != shape.points.shape[0] - 1:
#                         path_str +=  ' '
#                 shape_node.set('points', path_str)
#             elif isinstance(shape, pydiffvg.Path):
#                 shape_node = etree.SubElement(g, 'path')
                
#                 shape_node.set('d', shape2d(shape))
#             elif isinstance(shape, pydiffvg.Rect):
#                 shape_node = etree.SubElement(g, 'rect')
#                 shape_node.set('x', str(shape.p_min[0].item()))
#                 shape_node.set('y', str(shape.p_min[1].item()))
#                 shape_node.set('width', str(shape.p_max[0].item() - shape.p_min[0].item()))
#                 shape_node.set('height', str(shape.p_max[1].item() - shape.p_min[1].item()))
#             else:
#                 assert(False)

        shape_node.set('stroke-width', str(2 * shape.stroke_width.data.cpu().item()))
        if shape_group.fill_color is not None:
            if isinstance(shape_group.fill_color, pydiffvg.LinearGradient):
                shape_node.set('fill', 'url(#shape_{}_fill)'.format(i))
            else:
                c = shape_group.fill_color.data.cpu().numpy()
                shape_node.set('fill', 'rgb({}, {}, {})'.format(\
                    int(255 * c[0]), int(255 * c[1]), int(255 * c[2])))
                shape_node.set('opacity', str(c[3]))
        else:
            shape_node.set('fill', 'none')
        if shape_group.stroke_color is not None:
            if isinstance(shape_group.stroke_color, pydiffvg.LinearGradient):
                shape_node.set('stroke', 'url(#shape_{}_stroke)'.format(i))
            else:
                c = shape_group.stroke_color.data.cpu().numpy()
                shape_node.set('stroke', 'rgb({}, {}, {})'.format(\
                    int(255 * c[0]), int(255 * c[1]), int(255 * c[2])))
                shape_node.set('stroke-opacity', str(c[3]))
            shape_node.set('stroke-linecap', 'round')
            shape_node.set('stroke-linejoin', 'round')

        animate_node = etree.SubElement(shape_node, 'animate')
        animate_node.set('attributeName', 'd')
        animate_node.set('attributeType', 'XML')
        animate_node.set('values', ";".join(' '.join(shape2d(a[shape_i]) for shape_i in shape_group.shape_ids) for a in animated_shapes))
        animate_node.set('dur', '1s') # FIXME
        animate_node.set('repeatCount', 'indefinite')

    return prettify(root)
