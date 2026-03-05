#!/usr/bin/env python
"""Start TorchCodeV2 JupyterLab server."""

import os
import subprocess
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    notebooks_dir = project_root / "notebooks"
    
    # Create notebooks directory if needed
    notebooks_dir.mkdir(exist_ok=True)
    
    # Copy templates and solutions if notebooks directory is empty
    templates = list((project_root / "templates").glob("*.ipynb"))
    solutions = list((project_root / "solutions").glob("*.ipynb"))
    existing = list(notebooks_dir.glob("*.ipynb"))
    
    if len(existing) < len(templates) + len(solutions):
        print("Copying templates and solutions to notebooks directory...")
        import shutil
        for f in templates:
            shutil.copy(f, notebooks_dir / f.name)
        for f in solutions:
            shutil.copy(f, notebooks_dir / f.name)
        print(f"Copied {len(templates)} templates and {len(solutions)} solutions.")
    
    print("\n" + "=" * 50)
    print("📓 TorchCodeV2 JupyterLab")
    print("=" * 50)
    print("\n  Open the displayed URL in your browser")
    print("  Press Ctrl+C to stop\n")
    print("=" * 50 + "\n")
    
    # Start JupyterLab
    subprocess.run([
        sys.executable, "-m", "jupyter", "lab",
        f"--notebook-dir={notebooks_dir}",
        "--NotebookApp.token=''",
        "--no-browser"
    ])

if __name__ == "__main__":
    main()
