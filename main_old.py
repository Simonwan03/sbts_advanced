"""
Compatibility shim for the old entrypoint.

The project now uses a single active experiment pipeline implemented in
`main.py`. This file is intentionally kept lightweight so older scripts that
still call `python main_old.py` continue to work without maintaining a second
training stack.
"""

from main import main


if __name__ == "__main__":
    main()
