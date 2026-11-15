"""Launch one fresh, recorded native trial after its independent checks pass."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
config = ROOT / "config_gravity_seating_trial.json"
audit = ROOT / "runtime_ipc/independent-gravity-release-review.json"
preflight = ROOT / "runtime_ipc/full25-gravity-release-preflight-final.json"
assert sha(config) == "a07e36055578a5aeb7bd27262112a6dbbdf88c0db41ecc8a7c578113bf33a861"
assert sha(ROOT/"run.py") == "f774826d687eb6662bd1d4c491382214eac2fce22e554fd0ebdf9bf6fba0a2a6"
assert sha(ROOT/"gravity_release.py") == "d88f4a7cd34853fb8f07eb34d42e137007cd553f69ddc920a0856a0710c96dee"
a = read(audit)
assert a["all_checks_passed"] and not a["failed_checks"]
assert all(sha(ROOT/name) == digest for name, digest in a["production_source_sha256"].items())
p = read(preflight)
assert p["status"] == "candidate" and not p["physical_success"]
assert all(sha(ROOT/name) == digest for name, digest in p["input_sha256"].items())
cfg = read(config); old = read(ROOT/"runs/two_panel_full_24/config.json")
stripped = json.loads(json.dumps(cfg))
policy = stripped["task"]["contact_trial"].pop("gravity_release")
assert stripped == old
baseline = read(ROOT/"runs/two_panel_full_24/input_sha256.json")
assert all(sha(ROOT/name.removeprefix("source/")) == digest
           for name, digest in baseline.items() if name.startswith("source/") and name != "source/run.py")
out = ROOT/"runs/two_panel_full_25"
log = ROOT/"runs/two_panel_full_25.log"
decision = ROOT/"runtime_ipc/full25-gravity-release-decision.json"
launch = ROOT/"runtime_ipc/full25-launch.json"
assert not any(x.exists() for x in (out, log, decision, launch))
evidence = [audit, preflight,
    ROOT/"runs/two_panel_full_24/independent-contact-loss-window-review.json",
    ROOT/"runs/two_panel_full_24/independent-release-window-containment-review.json",
    ROOT/"runs/two_panel_full_24/independent-release-window-interpretation.json",
    ROOT/"runs/two_panel_full_24/independent-late-contact-review/review.json"]
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
record = {"kind": "fresh_native_gravity_seating_release_trial", "created_utc": now,
    "native_run_started": False, "physical_success": False, "config": str(config),
    "config_sha256": sha(config), "only_config_addition_from_full24": policy,
    "source_sha256": {p.name: sha(p) for p in sorted(ROOT.glob("*.py"))},
    "evidence_sha256": {str(p.relative_to(ROOT)): sha(p) for p in evidence},
    "rationale": "Full24 first recorded actual COM inclusion and over 50 percent node inclusion at TCP1.086432m while both fingers still reacted above2N. It later lost one-sided contact before the original deep endpoint. This new controller deliberately stops at TCP1.085m, checks the actual stopped state, and releases only if its explicit partial-capture conditions pass.",
    "limits": [
        "Previous moving-through geometry does not predict the new stopped state or prove stable support. Cloth panels and the free rear handle are in contact with the teddy.",
        "The original full-depth preflight remains required and unchanged. Partial insertion executes its own exact checked C2 joint curve with zero endpoint velocity/acceleration; no abrupt interruption or soft-state injection.",
        "Actual release requires valid analysis cap, mass COM inside, at least half of ALL1450 nodes inside, slow actual endpoint, correct pose and original two-sided contact. Partial capture is not physical task success.",
        "Opening remains the continuous original motor target0 to0.04m over0.5s, so actual unloading may lag the declared release start. No weakening of force, torque, cloth, tetra, table or left-holding limits.",
        "The original materials, upper-arm3N grasp, -30degree yaw and XY target remain unchanged; the bag remains two sewn cloth panels with zero-width bottom seam.",
        "The analysis cap is not simulated. Its validity flag does not certify every possible body self-intersection.",
        "Release and withdrawal use native contact; final complete teddy/COM containment and retention plus physical audits remain mandatory.",
        "Runs from time zero with original assets. Default desktop candidate is not changed before a complete validated success."]}
decision.write_text(json.dumps(record, indent=2)+"\n")
command = ["bash", str(ROOT/"run.sh"), "--headless", "--task", "full", "--config", str(config), "--output", out.name]
with log.open("x") as stream:
    process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
launch.write_text(json.dumps({"started_utc": now, "pid": process.pid, "command": command,
    "log": str(log), "output": str(out), "config_sha256": sha(config),
    "pretrial_decision": str(decision.relative_to(ROOT)), "pretrial_decision_sha256": sha(decision),
    "native_success_established": False}, indent=2)+"\n")
manifest = ROOT/"dependency_manifest.json"
data = read(manifest)
data.update(active_run=str(out.relative_to(ROOT)), updated_utc=now,
    gravity_seating_candidate=config.name, latest_full_task_passed=False)
manifest.write_text(json.dumps(data, indent=2)+"\n")
print(json.dumps({"pid": process.pid, "launch": str(launch), "decision_sha256": sha(decision)}))
