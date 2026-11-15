"""Restore checksum-verified source and native vendor archives, without overwrites."""
from pathlib import Path
import hashlib
import json
import tarfile
import tempfile
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    index = json.loads((ROOT / 'environment/release-assets.json').read_text())
    for key, relative in [('genesis_source', Path('genesis-world')),
                          ('native_vendor', Path('teddy_bag_scene/vendor'))]:
        record = index[key]
        archive = ROOT / 'release-assets' / record['file']
        if archive.stat().st_size != record['bytes'] or digest(archive) != record['sha256']:
            raise ValueError(f'Archive checksum mismatch: {archive.name}')
        destination = ROOT / relative
        if destination.exists():
            print(f'Preserving existing directory: {relative}')
            continue
        with tempfile.TemporaryDirectory(prefix='genesis-restore-', dir=ROOT) as temporary:
            staging = Path(temporary)
            with tarfile.open(archive, 'r:gz') as stream:
                for member in stream.getmembers():
                    path = Path(member.name)
                    if path.is_absolute() or '..' in path.parts:
                        raise ValueError('Unsafe archive path')
                    if path != relative and relative not in path.parents:
                        raise ValueError(f'Unexpected archive member: {member.name}')
                stream.extractall(staging, filter='data')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging / relative), str(destination))
        print(f'Restored: {relative}')


if __name__ == '__main__':
    main()

