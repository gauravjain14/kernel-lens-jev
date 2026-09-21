"""Create a source archive from explicit project paths; never include local secrets."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent.parent
files = ['package.json', 'package-lock.json', 'tsconfig.json', '.gitignore', '.vscodeignore', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md']
folders = ['src', 'media', 'scripts', 'test', 'docs', 'examples', '.vscode', '.github']
paths = [root / name for name in files]
extensions = {'.ts', '.js', '.mjs', '.json', '.py', '.cu', '.md', '.html', '.css', '.svg', '.yml', '.yaml'}
for folder in folders:
    paths.extend(p for p in (root / folder).rglob('*') if p.is_file() and p.suffix in extensions and '__pycache__' not in p.parts and not p.name.startswith('.env'))
destination = root / 'artifacts' / 'kernel-lens-source.zip'
destination.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
    for p in sorted(paths):
        if p.name.startswith('.env'):
            continue
        archive.write(p, Path('kernel-lens-jev') / p.relative_to(root))
print(f'Created {destination.name} ({len(paths)} source files).')
