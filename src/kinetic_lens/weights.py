"""Authenticated, revision-pinned retrieval of private learned artifacts."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'models/private_artifacts.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('visibility_requirement') != 'private':
        raise ValueError('A version-1 private artifact manifest is required')
    if manifest.get('repo_type') != 'model' or not re.fullmatch(r'[0-9a-f]{40}', manifest.get('revision', '')):
        raise ValueError('The model repository must be pinned to a full commit revision')
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', manifest.get('repo_id', '')):
        raise ValueError('Invalid model repository ID')
    seen = set()
    for entry in manifest.get('files', []):
        for field in ('path', 'hf_path'):
            path = PurePosixPath(entry[field])
            if not entry[field] or path == PurePosixPath('.') or path.is_absolute() or '..' in path.parts or '\\' in entry[field] or ':' in entry[field]:
                raise ValueError('Artifact paths must be relative POSIX paths')
        if not (root / entry['path']).resolve().is_relative_to(root):
            raise ValueError('Artifact destination escapes the repository')
        if entry['path'] in seen:
            raise ValueError('Duplicate artifact destination')
        seen.add(entry['path'])
        if entry['group'] not in ('default', 'cuda', 'research'):
            raise ValueError('Unknown artifact group')
        if not re.fullmatch(r'[0-9a-f]{64}', entry.get('sha256', '')):
            raise ValueError('An exact SHA256 is required')
        if not isinstance(entry.get('bytes'), int) or entry['bytes'] <= 0:
            raise ValueError('A positive artifact byte count is required')
    if not seen:
        raise ValueError('Empty private artifact manifest')
    return manifest


def verify_file(path, entry):
    return Path(path).is_file() and Path(path).stat().st_size == entry['bytes'] and sha256(path) == entry['sha256']


def ensure_artifacts(root, groups=('default',), *, from_workspace=None, verify_only=False):
    root = Path(root).resolve()
    manifest = load_manifest(root)
    groups = set(groups)
    if not groups or not groups.issubset({'default', 'cuda', 'research'}):
        raise ValueError('Select default, cuda and/or research artifact groups')
    selected = [entry for entry in manifest['files'] if entry['group'] in groups]
    missing = []
    for entry in selected:
        target = root / entry['path']
        if target.exists():
            if not verify_file(target, entry):
                raise ValueError(f'Existing artifact has an unexpected hash; move it aside explicitly: {entry["path"]}')
        else:
            missing.append(entry)
    if verify_only and missing:
        raise FileNotFoundError(f'{len(missing)} private artifacts are absent; run scripts/setup_models.py with the required groups')
    if from_workspace is not None:
        workspace = Path(from_workspace).resolve()
        for entry in list(missing):
            candidates = [workspace / 'private_weight_migration/backup' / entry['path'],
                          workspace / 'kinetic-lens-har-xai' / entry['path']]
            source = next((p for p in candidates if verify_file(p, entry)), None)
            if source is None:
                raise FileNotFoundError(f'No verified local source for {entry["path"]}')
            target = root / entry['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.partial')
            shutil.copyfile(source, temporary)
            if not verify_file(temporary, entry):
                temporary.unlink()
                raise ValueError('Local artifact copy failed its hash check')
            temporary.replace(target)
            missing.remove(entry)
    if missing:
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError:
            raise RuntimeError('Install huggingface-hub, then run hf auth login to access the private weights') from None
        api = HfApi()
        try:
            info = api.model_info(manifest['repo_id'], revision=manifest['revision'])
        except Exception:
            raise RuntimeError('Private weight access failed. Run hf auth login or configure HF_TOKEN locally; do not put a token in Git') from None
        if info.private is not True or info.sha != manifest['revision']:
            raise RuntimeError('Refusing an unpinned or non-private model repository')
        cache = root / 'models/.hub-cache'
        if not cache.resolve().is_relative_to(root):
            raise ValueError('Private download cache escapes the repository')
        for entry in missing:
            downloaded = Path(hf_hub_download(repo_id=manifest['repo_id'], repo_type='model',
                revision=manifest['revision'], filename=entry['hf_path'], cache_dir=cache))
            if not verify_file(downloaded, entry):
                raise ValueError(f'Private download failed its hash check: {entry["path"]}')
            target = root / entry['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.partial')
            shutil.copyfile(downloaded, temporary)
            if not verify_file(temporary, entry):
                temporary.unlink()
                raise ValueError('Private artifact copy failed its hash check')
            temporary.replace(target)
    return {'repo_id': manifest['repo_id'], 'revision': manifest['revision'],
            'groups': sorted(groups), 'verified_files': len(selected),
            'bytes': sum(entry['bytes'] for entry in selected)}
