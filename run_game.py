# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from riverborn.mgl_terrain import main
if __name__ == "__main__":
    sys.exit(main())
