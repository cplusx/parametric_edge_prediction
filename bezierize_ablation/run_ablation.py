import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bezierization.bezierize_ablation.run_ablation import *

if __name__ == '__main__':
    from bezierization.bezierize_ablation.run_ablation import main
    main()
