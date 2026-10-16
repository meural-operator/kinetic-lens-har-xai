"""Fetch Google's official MoViNet SavedModels and provenance into this workspace."""
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
MODELS = ['a0_stream','a2_stream','a2_base']
records = []
for name in MODELS:
    model,mode = name.split('_')
    url = f'https://tfhub.dev/tensorflow/movinet/{model}/{mode}/kinetics-600/classification/3?tf-hub-format=compressed'
    archive = ROOT/(name+'.tar.gz')
    folder = ROOT/'models'/name
    folder.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        print('DOWNLOAD '+name,flush=True)
        with urllib.request.urlopen(url,timeout=120) as response,archive.with_suffix('.partial').open('wb') as output:
            final_source = response.geturl().split('?')[0]
            while chunk := response.read(1024*1024):
                output.write(chunk)
        archive.with_suffix('.partial').replace(archive)
    else:
        final_source = 'Previously downloaded official TF Hub handle; verify archive hash below.'
    with tarfile.open(archive,'r:gz') as bundle:
        # Python's data filter rejects absolute paths, traversal and unsafe links.
        bundle.extractall(folder,filter='data')
    saved_models = list(folder.rglob('saved_model.pb'))
    assert len(saved_models)==1,(name,saved_models)
    weights = list(folder.rglob('variables.data-*'))
    assert weights,(name,'No variable checkpoint in archive')
    record = {'name':name,'provider':'Google/TensorFlow Model Garden',
              'requested_handle':url.split('?')[0], 'resolved_archive':final_source,
              'archive_path':str(archive),'archive_bytes':archive.stat().st_size,
              'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
              'saved_model_directory':str(saved_models[0].parent),
              'unpacked_bytes':sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()),
              'variable_bytes':sum(p.stat().st_size for p in weights),
              'variables_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in weights},
              'pretraining_dataset':'Kinetics-600','license_reference':
              'https://github.com/tensorflow/models/blob/master/official/projects/movinet/README.md#license'}
    records.append(record)
    print('READY '+json.dumps(record),flush=True)

urls = {'kinetics_600_labels.txt':'https://raw.githubusercontent.com/tensorflow/models/master/official/projects/movinet/files/kinetics_600_labels.txt',
        'official_README.md':'https://raw.githubusercontent.com/tensorflow/models/master/official/projects/movinet/README.md'}
for filename,url in urls.items():
    with urllib.request.urlopen(url,timeout=60) as response:
        (ROOT/filename).write_bytes(response.read())
assert len((ROOT/'kinetics_600_labels.txt').read_text().splitlines())==600
(ROOT/'downloads.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('DOWNLOADS_COMPLETE',flush=True)
