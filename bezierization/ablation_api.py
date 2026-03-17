from importlib import import_module

from bezierization.bezier_versions import CANONICAL_VERSIONS, VERSION_REGISTRY


def list_versions():
    return list(CANONICAL_VERSIONS)


def load_version(version_name):
    if version_name not in VERSION_REGISTRY:
        raise ValueError(f'Unknown version: {version_name}')
    module = import_module(VERSION_REGISTRY[version_name])
    return module


def run_version(version_name, image_path=None, output_dir=None, image_array=None, **overrides):
    module = load_version(version_name)
    return module.run_refiner(image_path=image_path, output_dir=output_dir, image_array=image_array, **overrides)
