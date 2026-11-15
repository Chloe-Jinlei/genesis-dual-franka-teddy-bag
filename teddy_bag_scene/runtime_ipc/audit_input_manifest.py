"""Read-only input manifest verification; external model files are inventoried separately."""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET


def audit(run):
    run=Path(run).resolve()
    manifest=json.loads((run/'input_sha256.json').read_text())
    failures=[]
    for name,expected in manifest.items():
        path=(run/name).resolve()
        if not path.is_relative_to(run) or not path.is_file():
            failures.append({'path':name,'reason':'missing or outside run'})
            continue
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected:failures.append({'path':name,'expected':expected,'actual':actual})
    files={'config.json'}|{str(p.relative_to(run)) for p in (run/'source').rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    missing=sorted(files-set(manifest))
    root=ET.parse(run/'source/assets/panda_tactile.xml').getroot()
    external=[]
    for mesh in root.findall('asset/mesh'):
        file=mesh.get('file')
        if not file:continue
        path=Path(file)
        if not path.is_absolute():
            directory=Path(root.find('compiler').get('meshdir',''))
            path=directory/path
            if not path.is_absolute():path=run/'source/assets'/path
        path=path.resolve()
        if path.is_relative_to(run):continue
        external.append(dict(path=str(path),exists=path.is_file(),
            sha256_at_review=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None))
    external=list({r['path']:r for r in external}.values())
    config=json.loads((run/'config.json').read_text())
    result=dict(kind='independent_input_manifest_review',
        manifest_entry_count=len(manifest),snapshot_file_count=len(files),
        all_declared_hashes_match=not failures,hash_or_path_failures=failures,
        all_snapshot_files_in_manifest=not missing,unmanifested_snapshot_files=missing,
        missing_external_model_files=[r['path'] for r in external if not r['exists']],
        external_model_file_count=len(external),external_model_files=external,
        self_contained_snapshot=not external,
        integration_dt_s=config['simulation']['dt'],
        declared_left_sweep_gap_m=config['task'].get('contact_trial',{}).get('left_robot_soft_sweep_clearance_m'),
        limitations=['External robot mesh hashes are measured at this review, not established as startup hashes.',
            'The input manifest covers scene source/config/copied assets, not installed Genesis/libuipc binaries or all environment dependencies.',
            'Hash consistency proves recorded inputs match the manifest; it does not prove physical correctness or task success.'])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    args=parser.parse_args();result=audit(args.run)
    out=args.run/'independent-input-manifest-review.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('external_model_files','limitations')},indent=2))
