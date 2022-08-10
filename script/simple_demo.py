from confmap.confmap import BFF
from confmap.io_utils import read_obj, write_obj

vertices, faces = read_obj('../data/bumpcap.obj')
cm = BFF(vertices, faces)
image = cm.layout()
write_obj('bumpcap_with_uv.obj', cm.vertices, cm.faces, image.vertices, image.faces)
