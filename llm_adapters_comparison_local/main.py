#!/usr/bin/env python3
"""
Local entry point for running adapter experiments without relying on notebooks.

Usage example:
    python main.py --config wo_privacy
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

RUN_CONFIGS = {
    "wo_privacy": {
        "config_file": "hyper_parameter_config_wo_privacy.json",
        "default_tag": "eps_inf",
    },
    "w_privacy": {
        "config_file": "hyper_parameter_config_w_privacy.json",
        "default_tag": "eps_8",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GLUE adapter experiments using a Python entry point."
    )
    parser.add_argument(
        "--config",
        choices=["wo_privacy", "w_privacy", "both"],
        default="wo_privacy",
        help="Which hyper-parameter configuration to run.",
    )
    parser.add_argument(
        "--datasets",
        default="sst2,qnli,mnli,qqp",
        help="Comma separated GLUE tasks to download and run on.",
    )
    parser.add_argument(
        "--adapter-methods",
        default="",
        help="Optional comma separated adapter methods to run. "
        "Defaults to the adapters defined in the selected config.",
    )
    parser.add_argument(
        "--model-name",
        default="prajjwal1/bert-tiny",
        help="Base Hugging Face model to fine-tune.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Relative or absolute path where prepared datasets are stored.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Relative or absolute path for experiment outputs.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload GLUE datasets even if a saved copy exists.",
    )
    parser.add_argument(
        "--file-tag",
        default="",
        help="Optional prefix for output filenames.",
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Disable the linear learning rate scheduler.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay used by the optimizer.",
    )
    parser.add_argument(
        "--hf-token-env",
        default="",
        help="Environment variable containing a Hugging Face token for login.",
    )
    return parser.parse_args()


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def maybe_login(token_env: str) -> None:
    token_names: Iterable[str]
    if token_env:
        token_names = (token_env,)
    else:
        token_names = ("HF_TOKEN", "HUGGINGFACE_TOKEN")

    for env_name in token_names:
        token = os.getenv(env_name)
        if token:
            try:
                from huggingface_hub import login

                login(token=token, add_to_git_credential=False)
                print(f"Hugging Face login succeeded using ${env_name}.")
            except Exception as exc:  # pragma: no cover - defensive guardrail
                print(f"Warning: could not login to Hugging Face ({exc}).")
            return

    print("Proceeding without Hugging Face login (no token environment variable set).")


def prepare_glue_datasets(
    tasks: Iterable[str],
    data_dir: Path,
    force_download: bool = False,
) -> Dict[str, object]:
    try:
        from datasets import load_dataset, load_from_disk
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'datasets' package is required. Install it with "
            "'pip install datasets'."
        ) from exc

    data_dir.mkdir(parents=True, exist_ok=True)
    datasets_map: Dict[str, object] = {}

    for task in tasks:
        task_dir = data_dir / task
        if force_download or not task_dir.exists():
            print(f"Downloading GLUE/{task} to {task_dir} ...")
            glue_dataset = load_dataset("glue", task)
            glue_dataset.save_to_disk(str(task_dir))
        else:
            print(f"Found cached GLUE/{task} dataset at {task_dir}.")

        print(f"Loading GLUE/{task} from disk.")
        datasets_map[task] = load_from_disk(str(task_dir))

    return datasets_map


def resolve_file_tag(base_tag: str, override: str, suffix: str | None = None) -> str:
    tag = override or base_tag
    if override and suffix:
        return f"{override}_{suffix}"
    if not override and suffix:
        return f"{base_tag}_{suffix}"
    return tag


def run_experiment(
    config_key: str,
    project_root: Path,
    datasets_map: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    sys.path.insert(0, str(project_root / "src"))
    try:
        import utils
        from train import train_test_all
    finally:
        sys.path.pop(0)

    spec = RUN_CONFIGS[config_key]
    config_path = project_root / spec["config_file"]
    hyper_parameter_config = utils.load_json_file(str(config_path))
    if hyper_parameter_config is None:
        raise FileNotFoundError(f"Missing configuration file: {config_path}")

    adapter_methods = (
        parse_csv_list(args.adapter_methods)
        if args.adapter_methods
        else list(hyper_parameter_config.keys())
    )

    dataset_list = parse_csv_list(args.datasets)
    missing_datasets = sorted(set(dataset_list) - set(datasets_map.keys()))
    if missing_datasets:
        raise ValueError(
            f"Datasets {missing_datasets} not prepared. "
            "Use --datasets and --force-download to make them available."
        )

    active_config = {}
    for adapter, adapter_params in hyper_parameter_config.items():
        filtered = {
            dataset_name: params
            for dataset_name, params in adapter_params.items()
            if dataset_name in dataset_list
        }
        if filtered:
            active_config[adapter] = filtered
    if not active_config:
        raise ValueError(
            "No overlapping datasets between CLI selection and configuration."
        )
    adapter_methods = [adapter for adapter in adapter_methods if adapter in active_config]
    if not adapter_methods:
        raise ValueError(
            "Adapter selection does not match any adapters present in the configuration."
        )

    output_base = (project_root / args.output_dir).resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M")
    file_tag = resolve_file_tag(
        base_tag=spec["default_tag"],
        override=args.file_tag,
        suffix=config_key if args.config == "both" else None,
    )
    output_path = output_base / f"{file_tag}_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Writing results to {output_path}")
    train_test_all(
        model_name=args.model_name,
        hyper_parameter_config=active_config,
        datasets=datasets_map,
        output_path=str(output_path),
        adapter_method_list=adapter_methods,
        dataset_list=dataset_list,
        scheduler=not args.no_scheduler,
        weight_decay=args.weight_decay,
        file_tag=file_tag,
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    maybe_login(args.hf_token_env)

    dataset_list = parse_csv_list(args.datasets)
    datasets_map = prepare_glue_datasets(
        tasks=dataset_list,
        data_dir=(project_root / args.data_dir).resolve(),
        force_download=args.force_download,
    )

    configs_to_run = (
        list(RUN_CONFIGS.keys()) if args.config == "both" else [args.config]
    )
    for config_key in configs_to_run:
        run_experiment(
            config_key=config_key,
            project_root=project_root,
            datasets_map=datasets_map,
            args=args,
        )


if __name__ == "__main__":
    main()
