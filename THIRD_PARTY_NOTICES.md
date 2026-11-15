# Third-party resources

- **Genesis 1.4.3**: https://github.com/Genesis-Embodied-AI/genesis-world at commit `ffd119c30118ebb9745fc009349efb2112a8cb76`, Apache-2.0. License copied to `third_party/GENESIS_LICENSE` and retained in the source release archive. The upstream working tree was clean; scene-specific physics adapters live in `teddy_bag_scene/physics_config.py` and the accompanying scene code.
- **Franka Emika Panda resources**: copied unchanged from that Genesis commit's `genesis/assets/xml/franka_emika_panda/assets/`. All 67 meshes referenced by the scene are in `teddy_bag_scene/assets/franka_meshes/`, alongside the upstream README and Apache-2.0 license. The scene's `panda_tactile.xml` adds tactile finger pad geometry; in the portable export only mesh resource paths were changed to relative paths. The frozen historical XML remains unchanged in the original evidence.
- **pyuipc 0.0.29 / libuipc**: Apache-2.0; all supplied LICENSE and NOTICE files remain inside the native-vendor release archive, including `pyuipc-0.0.29.dist-info/licenses/`.
- **Polyscope 2.6.1**: MIT; its distribution license remains in the native-vendor archive.
- **huggingface_hub 2.4.0**: Apache-2.0; its distribution license remains in the native-vendor archive. No user login or token cache is included.
- Other Python packages are recorded by version in `environment/python-packages.lock`; their own licenses apply when installed. The original Python virtual environment is not redistributed.
- The scene's teddy, cloth meshes and procedural textures are retained with their generators and provenance files. Reference imagery is retained as the original task's visual reference, without granting a separate third-party image license.

Third-party notices do not assign a new blanket license to the custom project code.
