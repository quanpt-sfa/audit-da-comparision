from __future__ import annotations

import argparse

from _next_diag_common import load_config, resolve
from audit_da.diag_common import write_tables
from audit_da.diag_cfs_proxy_validation import inventory_and_line_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory and map detailed cash-flow-statement line items")
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    settings = dict(config["cfs_shifting_validation"])
    tables = inventory_and_line_items(
        resolve(config_path, config["paths"]["raw_input"]),
        settings,
    )
    write_tables(tables, resolve(config_path, config["paths"]["output_dir"]))


if __name__ == "__main__":
    main()
