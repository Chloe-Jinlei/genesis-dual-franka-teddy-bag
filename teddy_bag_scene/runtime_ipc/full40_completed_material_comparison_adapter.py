"""One-run offline filename compatibility adapter; no numerical changes.

Preserve the original completed comparator and reference method byte-for-byte.
Correct only its historical launch filename, and record that correction before
the comparator creates its exclusive output. All original readiness checks run.
"""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "runtime_ipc/compare_cloth_material_run.py"
REFERENCE = ROOT / "runtime_ipc/full30_cloth_material_baseline.py"
BASELINE = ROOT / "runtime_ipc/full30-cloth-material-baseline.json"
EXPECTED = {
    WRAPPER: "a93753e0059512f8ed654fa0fff20303aca345afbdffe09f1e1e40e27d4d4aea",
    REFERENCE: "2bc32f735837f90f7cb7f44f21616f8194db9985317293cfbe541307ebb44b47",
    BASELINE: "ec66543e57231e8b544ec110b7fcc9c7f8470ccc049dcb5612dd9452c383d177",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 1:
        raise ValueError("This adapter is exclusively bound to two_panel_full_40")
    if not all(sha(path) == digest for path, digest in EXPECTED.items()):
        raise ValueError("Immutable comparator or reference changed")
    source = WRAPPER.read_text()
    old = 'launch_path = ROOT/"runtime_ipc"/(run.name.removeprefix("two_panel_")+"-launch.json")'
    new = 'launch_path = ROOT/"runtime_ipc"/(run.name.removeprefix("two_panel_").replace("full_", "full", 1)+"-launch.json")'
    if source.count(old) != 1:
        raise ValueError("Unexpected launch filename expression")
    source = source.replace(old, new, 1)
    provenance = {
        "adapter_path": str(Path(__file__).resolve()),
        "adapter_sha256": sha(Path(__file__).resolve()),
        "run": "two_panel_full_40",
        "original_wrapper_and_reference_hashes": {str(p): h for p, h in EXPECTED.items()},
        "original_invocation": "Full40 has not been executed; proactively correct the filename bug observed in the historical full38 invocation",
        "single_functional_correction": {"from": "runtime_ipc/full_40-launch.json", "to": "runtime_ipc/full40-launch.json"},
        "launch_and_completion_readiness_checks_still_executed": True,
        "numeric_method_or_sampling_changes": False,
        "output_name_left_as_original_wrapper_computes": "runtime_ipc/full_40-cloth-material-baseline.json",
    }
    # Reporting-only insertion; every computational/readiness statement remains.
    marker = '    with output.open("x") as file:'
    if source.count(marker) != 1:
        raise ValueError("Unexpected exclusive report-writing structure")
    source = source.replace(marker, '    measured["completed_invocation_compatibility"] = ADAPTER_PROVENANCE\n' + marker, 1)
    namespace = {"__name__": "immutable_completed_comparator_with_filename_adapter", "__file__": str(WRAPPER), "ADAPTER_PROVENANCE": provenance}
    exec(compile(source, str(WRAPPER), "exec"), namespace)
    sys.argv = [str(WRAPPER), "two_panel_full_40"]
    namespace["main"]()
    if not all(sha(path) == digest for path, digest in EXPECTED.items()):
        raise ValueError("Immutable inputs changed during comparison")


if __name__ == "__main__":
    main()
