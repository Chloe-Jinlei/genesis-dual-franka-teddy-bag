"""CPU-only summary of saved controller and native IPC feedback evidence.

This reports measurements, not a force-closure or task-success certificate.
Run only against a finished run directory; no Genesis import or simulation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def audit(run: Path):
    config = json.loads((run / "config.json").read_text())
    events = json.loads((run / "events.json").read_text())
    if (run / "report.json").exists():
        final_events = json.loads((run / "report.json").read_text()).get("events", [])
        if len(final_events) > len(events):
            events = final_events
    trace = np.load(run / "joint_trace.npy", allow_pickle=False)
    states = np.load(run / "states.npz", allow_pickle=False)
    initial = int(np.argmin(states["time"]))
    if float(states["time"][initial]) < float(trace[0, 0]) - 1e-9:
        trace = np.vstack([np.r_[states["time"][initial], states["q_left"][initial],
                                states["q_right"][initial]], trace])
    raw = json.loads((run / "coupling_trace.json").read_text())
    if trace.ndim != 2 or trace.shape[1] != 19 or not np.isfinite(trace).all():
        raise ValueError("Expected finite per-step trace [time, 9 left joints, 9 right joints].")
    dt = np.diff(trace[:, 0])
    if np.any(dt <= 0):
        raise ValueError("Per-step trace time is not strictly increasing.")
    # Captures at phase ends can duplicate the regular visualization sample.
    diagnostics = {round(float(row["time"]), 10): row for row in raw if row["time"] > 0}
    diagnostic_times = np.asarray(sorted(diagnostics))
    rows = [diagnostics[t] for t in diagnostic_times]
    phases = []
    for event in events:
        start = float(event["start"])
        end = min(start + float(event["duration"]), float(trace[-1, 0]))
        item = {"phase": event["phase"], "start_s": start, "observed_end_s": end}
        for side, first_column in (("left", 1), ("right", 10)):
            q = trace[:, first_column + 7:first_column + 9]
            inside = (trace[:, 0] >= start - 1e-9) & (trace[:, 0] <= end + 1e-9)
            # A finite-difference speed belongs to the interval ending at t[i].
            intervals = (trace[1:, 0] > start + 1e-9) & (trace[1:, 0] <= end + 1e-9)
            if not inside.any():
                continue
            values = {"finger_q_min_m": q[inside].min(0).tolist(),
                      "finger_q_max_m": q[inside].max(0).tolist(),
                      "finger_q_last_m": q[inside][-1].tolist(),
                      "peak_finite_difference_finger_speed_m_s":
                          np.max(np.abs(np.diff(q, axis=0)[intervals] / dt[intervals, None]), axis=0).tolist()
                          if intervals.any() else None}
            state_inside = (states["time"] >= start - 1e-9) & (states["time"] <= end + 1e-9)
            if state_inside.any() and "force_" + side in states:
                pd = states["force_" + side][state_inside, -2:]
                values["sampled_pd_force_min_n"] = pd.min(0).tolist()
                values["sampled_pd_force_max_n"] = pd.max(0).tolist()
                values["sampled_pd_force_last_n"] = pd[-1].tolist()
            samples = []
            translation_errors, rotation_errors = [], []
            for t, row in zip(diagnostic_times, rows):
                if start - 1e-9 <= t <= end + 1e-9:
                    fingers = {z["link"]: z for z in row["links"]
                               if z["entity"] == side + "_franka" and "finger" in z["link"]}
                    if len(fingers) != 2:
                        continue
                    ordered = [fingers["left_finger"], fingers["right_finger"]]
                    # This is the actual native STC reaction, projected onto
                    # each finger's positive-opening prismatic axis at aim.
                    normals = [float(np.dot(z["last_feedback_force_world_n"],
                                            z["aim_opening_axis_world"])) for z in ordered]
                    force = np.sum([z["last_feedback_force_world_n"] for z in ordered], axis=0)
                    samples.append([t, *normals, *force])
                    translation_errors.extend(z["translation_error_m"] for z in ordered)
                    rotation_errors.extend(z["rotation_error_rad"] for z in ordered)
            if samples:
                a = np.asarray(samples)
                values.update(native_reaction_samples=len(a),
                              native_opening_reaction_min_n=a[:, 1:3].min(0).tolist(),
                              native_opening_reaction_max_n=a[:, 1:3].max(0).tolist(),
                              native_opening_reaction_last_n=a[-1, 1:3].tolist(),
                              sampled_fraction_both_opening_reactions_above_0p1n=float(np.mean(np.all(a[:, 1:3] > .1, axis=1))),
                              summed_finger_reaction_world_last_n=a[-1, 3:6].tolist(),
                              maximum_finger_proxy_translation_error_m=max(translation_errors),
                              maximum_finger_proxy_rotation_error_rad=max(rotation_errors))
            item[side] = values
        phases.append(item)
    handoff = next((e for e in events if e["phase"] == "release_right_handle"), None)
    displacement = None
    if handoff:
        mask = states["time"] >= handoff["start"] - 1e-9
        tcp = states["tcp_left"][mask]
        if len(tcp):
            displacement = float(np.linalg.norm(tcp - tcp[0], axis=1).max())
    return {"run": str(run), "duration_s": float(trace[-1, 0]),
            "task_passed": None, "interpretation": "Measurement summary; does not assign task success.",
            "maximum_trace_interval_s": float(dt.max()),
            "finger_kp": config["robots"]["finger_kp"],
            "finger_kv": config["robots"]["finger_kv"],
            "maximum_left_tcp_displacement_after_handoff_m": displacement,
            "phases": phases,
            "limitations": ["IPC feedback is a proxy reaction, not individual contact impulse data.",
                            "Diagnostic forces are sampled more sparsely than the joint trace; unrecorded peaks may be higher.",
                            "Positive normal reaction does not alone prove frictional grasp or prevent material slip.",
                            "Pose, mesh, material, and controller changes must not be mistaken for an isolated damping experiment."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run / "independent-contact-evidence.json"
    result = audit(args.run)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(output), "duration_s": result["duration_s"],
                      "phases": len(result["phases"])}))
