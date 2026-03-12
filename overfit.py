import argparse
import sys

from train import main as train_main


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a focused overfit experiment for the parametric edge detector.')
    parser.add_argument('--config', default='configs/parametric_edge/default.yaml')
    parser.add_argument('--override-config', default='configs/parametric_edge/overfit.yaml')
    args = parser.parse_args()
    sys.argv = [sys.argv[0], '--config', args.config, '--override-config', args.override_config]
    train_main()


if __name__ == '__main__':
    main()
