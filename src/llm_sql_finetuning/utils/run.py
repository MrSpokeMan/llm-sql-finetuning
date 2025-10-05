import argparse
import yaml
from typing import Any


def parse_args_and_config() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run script")

    p.add_argument("--config-file", type=str, help="Path to YAML config file")
    args = vars(p.parse_args())

    config = {}
    if args.get("config_file"):
        with open(args["config_file"]) as f:
            config = yaml.safe_load(f)

    def flatten(d: Any, parent_key: str = "", sep: str = "_") -> dict[str, Any]:
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    flat_config = flatten(config)

    for k, v in flat_config.items():
        if args.get(k) is None:
            args[k] = v

    return argparse.Namespace(**args)
