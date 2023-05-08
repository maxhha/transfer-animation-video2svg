import os
from picosvg.svg import SVG
from tqdm import tqdm

root_dir = os.path.join('data', 'solid-icons')
src_dir = os.path.join(root_dir, 'svg-old')
dest_dir = os.path.join(root_dir, 'svg')

for filename in tqdm(os.listdir(src_dir)):
    src_path =  os.path.join(src_dir, filename)
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path):
        continue
    
    svg = SVG.parse(src_path).topicosvg()
    output = svg.tostring()
    with open(dest_path, "w") as f:
        f.write(output)
