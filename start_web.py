#!/usr/bin/env python
"""Start HappyTorch web server."""

import os
import sys
from pathlib import Path

# Prevent OpenMP duplicate library crash on Windows (numpy + torch both bundle libiomp5md.dll)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    try:
        import fastapi
    except ImportError:
        missing.append("fastapi")

    try:
        import uvicorn
    except ImportError:
        missing.append("uvicorn")

    try:
        import torch
    except ImportError:
        missing.append("torch")

    return missing


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🔥 HappyTorch Web Server")
    print("=" * 50)

    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\nPlease install them with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

    # Import after dependency check
    import uvicorn
    from web.app import app

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    db_path = os.environ.get("HAPPYTORCH_DB_PATH", "data/happytorch.db")
    public_origin = os.environ.get("PUBLIC_ORIGIN", "").strip()

    if public_origin:
        browser_hint = public_origin
    elif host in {"0.0.0.0", "::"}:
        browser_hint = f"http://<server-ip>:{port}"
    else:
        browser_hint = f"http://{host}:{port}"

    print(f"\n  Bind: {host}:{port}")
    print(f"  Database: {db_path}")
    print(f"  Open: {browser_hint}")
    if not public_origin:
        print("  Tip: set PUBLIC_ORIGIN=https://your-domain for clearer remote deployment hints")
    print("  Press Ctrl+C to stop\n")
    print("=" * 50 + "\n")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
