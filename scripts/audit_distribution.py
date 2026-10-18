#!/usr/bin/env python3
"""Check a reviewed manifest and optionally build/verify its code-only ZIP.

The manifest is an author-reviewed inventory, never populated by directory glob.
Run this helper with Python 3.9+; no third-party libraries are required.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


TOP_LEVEL = {
    '.gitignore', 'README.md', 'REVIEW_ONLY_NOTICE.txt', 'pyproject.toml',
    'requirements.txt', 'requirements-tested.txt',
}
CONFIG = {'analysis.yaml', 'fields.yaml', 'paths.example.yaml', 'prompt.txt'}
SECRET = re.compile(
    r'(?:sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|'
    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)'
)
MANIFEST = 'FILE_MANIFEST_SHA256.txt'
PREFIX = 'environmental-noise-ageing/'


def allowed(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        return False
    if name in TOP_LEVEL:
        return True
    parts = path.parts
    if len(parts) == 2 and parts[0] == 'config':
        return parts[1] in CONFIG
    if len(parts) == 2 and parts[0] == 'docs':
        return path.suffix == '.md' or name == 'docs/script_inventory.csv'
    if len(parts) == 2 and parts[0] in {'scripts', 'tests'}:
        return path.suffix in ({'.py', '.R'} if parts[0] == 'scripts' else {'.py'})
    if len(parts) >= 3 and parts[:2] == ('src', 'noise_agegap_pipeline'):
        return path.suffix == '.py' and all(part.isidentifier() for part in parts[2:-1])
    return name in {
        f'examples/example_{i:02d}/{kind}.json'
        for i in range(1, 4) for kind in ('input', 'output')
    }


def inspect_content(name, payload):
    text = payload.decode('utf-8-sig')
    if '\x00' in text or SECRET.search(text):
        raise ValueError(f'Binary content or credential pattern: {name}')
    if name.startswith('examples/'):
        value = json.loads(text)
        if name.endswith('/input.json'):
            example = PurePosixPath(name).parts[1].upper()
            if value.get('eid') != 'SYNTHETIC_' + example:
                raise ValueError(f'Unexpected example identifier: {name}')
        else:
            explanations = [v for k, v in value.items() if k.startswith('inference process - ')]
            if len(explanations) != 8 or any(
                not isinstance(v, str) or 'fictional' not in v.lower() for v in explanations
            ):
                raise ValueError(f'Unexpected example explanations: {name}')
    if name == 'docs/script_inventory.csv':
        rows = list(csv.DictReader(io.StringIO(text)))
        expected = ['kind', 'name', 'source_path', 'output_path', 'scope']
        if not rows or list(rows[0]) != expected:
            raise ValueError('Unexpected script-inventory schema')
        if any(row['kind'] not in {'module', 'script', 'aggregate_export'} for row in rows):
            raise ValueError('Unexpected inventory row type')


def manifest_entries(payload):
    entries = {}
    for line in payload.decode('utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if not re.fullmatch('[a-f0-9]{64}', digest) or not allowed(name) or name in entries:
            raise ValueError(f'Invalid or unapproved manifest entry: {name}')
        entries[name] = digest
    if not entries:
        raise ValueError('Empty manifest')
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--build', type=Path, help='Create a new ZIP; never overwrite an existing file')
    group.add_argument('--check-archive', type=Path, help='Verify a ZIP against this reviewed local manifest')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / MANIFEST
    if manifest_path.is_symlink():
        raise ValueError('Symlink manifest')
    manifest = manifest_path.read_bytes()
    entries = manifest_entries(manifest)
    payloads = {}
    for name, expected in entries.items():
        path = root / name
        if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
            raise ValueError(f'Symlink in release path: {name}')
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f'Content changed since review: {name}')
        inspect_content(name, payload)
        payloads[name] = payload
    payloads[MANIFEST] = manifest
    if args.build:
        # Exclusive creation preserves earlier reviewed archives.
        with zipfile.ZipFile(args.build, 'x', zipfile.ZIP_DEFLATED) as archive:
            for name, payload in sorted(payloads.items()):
                archive.writestr(PREFIX + name, payload)
    archive_path = args.check_archive or args.build
    if archive_path:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            expected_names = {PREFIX + name for name in payloads}
            if len(names) != len(set(names)) or set(names) != expected_names:
                raise ValueError('Archive contains missing, duplicate, or unreviewed files')
            if archive.testzip() is not None:
                raise ValueError('Archive CRC check failed')
            for info in archive.infolist():
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError('Archive contains a symlink')
                if archive.read(info) != payloads[info.filename[len(PREFIX):]]:
                    raise ValueError(f'Archive content differs from reviewed file: {info.filename}')
    print(json.dumps({
        'status': 'passed', 'reviewed_files': len(payloads),
        'archive_checked': bool(archive_path),
        'scope': 'Reviewed code-only distribution; new content requires review before manifest updates',
    }, indent=2))


if __name__ == '__main__':
    main()
