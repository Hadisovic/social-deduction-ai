"""Standard-library-only package integrity, safe exports and isolated imports."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path,PurePosixPath


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def safe_path(root,name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('Unsafe archive path: '+name)
    result=Path(root).joinpath(*p.parts).resolve()
    if not result.is_relative_to(Path(root).resolve()):raise ValueError('Escaping package path')
    return result


def verify(root,*,controls_only=False):
    root=Path(root);manifest=json.loads((root/'portable.manifest.json').read_text(encoding='utf-8'))
    for name,entry in manifest['files'].items():
        if controls_only and (name.startswith(('payloads/','origin/')) or Path(name).suffix not in ('.py','.ps1','.json')):continue
        p=safe_path(root,name)
        if p.stat().st_size!=entry['size'] or digest(p)!=entry['sha256']:raise ValueError('Package integrity failure: '+name)
    return manifest


def export_results(root,folder):
    root=Path(root).resolve();folder=Path(folder).resolve()
    latest=json.loads((folder/'latest.json').read_text());update=latest['counters']['update']
    registration=json.loads((folder/'registration.json').read_text()) if (folder/'registration.json').exists() else {}
    regime='investigation-a2-v1' if registration.get('reward_regime')=='investigation-a2-v1' else 'task-pursuit-v1'
    output=root/'exports';output.mkdir(exist_ok=True)
    number=1
    while (output/f'{regime}-update-{update:06d}-export-{number:03d}.zip').exists():number+=1
    target=output/f'{regime}-update-{update:06d}-export-{number:03d}.zip'
    files={}
    for p in folder.rglob('*'):
        if p.is_file() and p.name!='writer.lock' and not p.name.endswith(('.tmp','.writing')):files['run/'+p.relative_to(folder).as_posix()]=p
    manifest=json.loads((root/'portable.manifest.json').read_text())
    for name in manifest['files']:
        if not name.startswith('payloads/'):files['source/'+name]=safe_path(root,name)
    entries={}
    record=dict(schema='patrick-jane-results-export-v1',update=update,lifetime_decisions=latest['counters']['individual_decisions'],files=entries)
    temporary=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        # Hash the bytes actually archived: the independent viewer may still
        # replace its heartbeat or append a log while the export is written.
        for name,p in sorted(files.items()):
            h=hashlib.sha256();size=0
            with p.open('rb') as src,z.open(name,'w',force_zip64=True) as dst:
                for block in iter(lambda:src.read(1024*1024),b''):h.update(block);size+=len(block);dst.write(block)
            entries[name]=dict(sha256=h.hexdigest(),size=size)
        z.writestr('export-manifest.json',json.dumps(record,indent=2)+'\n')
    temporary.replace(target)
    (target.with_suffix('.zip.sha256')).write_text(digest(target)+'  '+target.name+'\n')
    return target


def import_results(archive,destination):
    archive=Path(archive);destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Import destination must be new; existing models are never replaced')
    with zipfile.ZipFile(archive) as z:
        if len(set(z.namelist()))!=len(z.namelist()):raise ValueError('Duplicate ZIP members')
        manifest=json.loads(z.read('export-manifest.json'))
        if manifest.get('schema')!='patrick-jane-results-export-v1':raise ValueError('Unknown results export')
        if set(z.namelist())!=set(manifest['files'])|{'export-manifest.json'}:raise ValueError('Unexpected export members')
        for name,entry in manifest['files'].items():
            safe_path(destination,name)
            h=hashlib.sha256();size=0
            with z.open(name) as src:
                for block in iter(lambda:src.read(1024*1024),b''):h.update(block);size+=len(block)
            if h.hexdigest()!=entry['sha256'] or size!=entry['size']:raise ValueError('Corrupt export member: '+name)
        destination.mkdir(parents=True,exist_ok=False)
        for name in manifest['files']:
            target=safe_path(destination,name);target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(name) as src,target.open('xb') as out:shutil.copyfileobj(src,out)
        (destination/'export-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return destination


def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--export',action='store_true')
    p.add_argument('--root',default=str(Path(__file__).resolve().parent));p.add_argument('--folder');p.add_argument('--import-archive');p.add_argument('--destination')
    a=p.parse_args()
    if a.verify:print('PACKAGE VERIFIED:',len(verify(a.root)['files']),'files')
    elif a.export:print(export_results(a.root,a.folder))
    elif a.import_archive:print(import_results(a.import_archive,a.destination))

if __name__=='__main__':main()
