# 复现环境记录

`python-packages.lock` 保存成功运行机器中 155 个 Python 分发的精确版本，仅去掉 Genesis 的本机 editable 路径。Genesis 单独使用 `ffd119c30118ebb9745fc009349efb2112a8cb76` 源码安装。`environment.json` 记录平台、原生依赖与验证范围。本次没有修改现有仿真或依赖环境。

这是一份已安装环境快照，**尚未在干净机器完成安装测试**，也不是带所有 wheel 哈希的离线依赖仓库。锁文件包括当时环境内的辅助包，未声称每个包都是此场景的最小必需依赖。

## 平台与原生依赖

- 使用 Linux x86-64、CPython 3.12；原生 vendor wheel 要求 glibc 至少 2.34。
- 实测 GPU 为 RTX 5070 Laptop 8 GB，NVIDIA 驱动 595.99.02。Genesis 使用 CPU 前端，libuipc 接触求解仍使用 CUDA。
- `vendor/uipc/compatibility.json` 说明 pyuipc wheel 以 CUDA 12.8 编译，不要求单独安装系统 CUDA toolkit。其驱动门槛不能当作 CUDA 13.0 Torch 的完整驱动门槛。
- 无窗口渲染需要 EGL/OpenGL 驱动；窗口模式需要 GLX 和可用显示服务。
- 将仓库 Release 提供的原生依赖包核对 SHA-256 后恢复到 `teddy_bag_scene/vendor/`，保留 `.dist-info`、LICENSE、NOTICE 和共享库。它含 pyuipc 0.0.29、Polyscope 2.6.1、huggingface_hub 2.4.0，未列入 venv 锁文件。可选远程资源下载及 Hub CLI 不属于已验证流程。

## 安装顺序

以下从仓库根目录运行，需要先恢复同级 `genesis-world/` 源码和原生 vendor。推荐由仓库安装脚本执行同样顺序。这里使用 `uv` 创建环境和安装，源机器的 venv 本身没有 pip 模块。

```bash
uv venv --python 3.12 .venv
requirements_without_torch="$(mktemp)"
sed '/^torch==/d' environment/python-packages.lock > "$requirements_without_torch"
uv pip install --python .venv/bin/python --no-deps -r "$requirements_without_torch"
rm -f -- "$requirements_without_torch"
uv pip install --python .venv/bin/python --no-deps \
  --index-url https://download.pytorch.org/whl/cu130 'torch==2.14.0'
uv pip install --python .venv/bin/python --no-deps -e ./genesis-world
uv pip check --python .venv/bin/python
```

这里的 `--no-deps` 用于按完整快照安装每一条已记录依赖，防止解析过程另行选择 Torch 构建。Genesis 构建隔离环境仍可能安装 `pyproject.toml` 声明的构建依赖；这些临时构建依赖的完整版本和哈希未锁定。若索引缺少某个精确版本，应报告安装失败，不能静默改版本。

**Torch 有两个不同但都真实的版本字段：** 分发元数据是 `2.14.0`，运行时 `torch.__version__` 是 `2.14.0+cu130`，`torch.version.cuda` 是 `13.0`。锁文件忠实保留 `torch==2.14.0`；必须从单独的官方 cu130 索引安装并核对运行时，不能仅看这个无后缀的版本号。上述索引与版本依据 [PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/#v2-14-0)，本次没有下载或安装该 wheel。

```bash
.venv/bin/python - <<'PYVERIFY'
import importlib.metadata as md
import torch
assert md.version('torch') == '2.14.0'
assert torch.__version__ == '2.14.0+cu130'
assert torch.version.cuda == '13.0'
assert md.version('genesis-world') == '1.4.3'
print('Python/Torch/Genesis versions match the recorded build.')
PYVERIFY
PYTHONPATH="$PWD/teddy_bag_scene/vendor" .venv/bin/python - <<'PYVERIFY'
import importlib.metadata as md
import uipc
assert md.version('pyuipc') == '0.0.29'
assert str(uipc.__version__) == '0.0.29'
print('Bundled native pyuipc imports successfully.')
PYVERIFY
```

这些版本与导入检查不替代完整仿真验收。原机器的 `uv pip check` 已通过；历史 full40 视频、输入哈希和审计报告属于原始成功运行证据，导出环境尚未重新执行完整任务。
