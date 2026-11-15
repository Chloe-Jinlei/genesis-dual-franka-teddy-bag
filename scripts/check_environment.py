"""Check the exported runtime and portable assets without starting a simulation."""
from pathlib import Path
import json
import platform
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'genesis-world'))
sys.path.insert(0, str(ROOT / 'teddy_bag_scene/vendor'))
assert sys.version_info[:2] == (3, 12), 'CPython 3.12 required for the native wheel'
assert platform.system() == 'Linux' and platform.machine() == 'x86_64', 'Bundled native wheel requires Linux x86-64'

import torch
import genesis
import uipc
import mujoco

assert torch.__version__ == '2.14.0+cu130', torch.__version__
assert torch.version.cuda == '13.0', torch.version.cuda
assert torch.cuda.is_available(), 'A working NVIDIA CUDA driver/GPU is required'
xml = ROOT / 'teddy_bag_scene/assets/panda_tactile.xml'
tree = ET.parse(xml).getroot()
meshdir = xml.parent / tree.find('compiler').attrib['meshdir']
for mesh in tree.findall('.//mesh'):
    if 'file' in mesh.attrib:
        assert (meshdir / mesh.attrib['file']).is_file(), mesh.attrib['file']
model = mujoco.MjModel.from_xml_path(str(xml))
print(json.dumps({'python': platform.python_version(), 'torch': torch.__version__,
                  'torch_cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                  'genesis_source': str(Path(genesis.__file__).resolve()),
                  'uipc_source': str(Path(uipc.__file__).resolve()),
                  'franka_model': {'nq': model.nq, 'nv': model.nv, 'nu': model.nu},
                  'scope': 'Import, GPU availability and MJCF check; not a complete simulation or clean-machine install test.'},
                 indent=2))

