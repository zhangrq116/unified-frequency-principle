#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import datetime as _datetime
import hashlib
import json
import math
import os
import pickle
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SCRIPT_VERSION = "1.0.0"
SCRIPT_NAME = Path(__file__).name
PI = math.pi

NoiseTriple = Tuple[float, float, float]


@dataclass(frozen=True)
class Preset:

    dataset: str
    source_script: str
    circuit_profile: str
    noise_enabled: bool
    encoder_noise: NoiseTriple
    single_gate_noise: NoiseTriple
    double_gate_noise: NoiseTriple
    epochs: int


ZERO_NOISE: NoiseTriple = (0.0, 0.0, 0.0)
HISTORICAL_SINGLE_GATE_NOISE: NoiseTriple = (0.0001 / 4,) * 3
HISTORICAL_DOUBLE_GATE_NOISE: NoiseTriple = (0.01 / 4,) * 3


PRESETS: Mapping[str, Preset] = {
    "main_noiseless": Preset(
        dataset="QNN_FP_20250704_010607",
        source_script="QNN_initial_noise_equal_noiseless_parfor7.py",
        circuit_profile="encoding-only",
        noise_enabled=False,
        encoder_noise=(0.0, 0.15, 0.0),
        single_gate_noise=ZERO_NOISE,
        double_gate_noise=ZERO_NOISE,
        epochs=800,
    ),
    "main_y_015": Preset(
        dataset="QNN_FP_20250704_010758",
        source_script="QNN_initial_noise_equal_noiseless_parfor8.py",
        circuit_profile="encoding-only",
        noise_enabled=True,
        encoder_noise=(0.0, 0.15, 0.0),
        single_gate_noise=ZERO_NOISE,
        double_gate_noise=ZERO_NOISE,
        epochs=800,
    ),
    "main_depolarizing_005": Preset(
        dataset="QNN_FP_20250704_121259",
        source_script="QNN_initial_noise_equal_noiseless_parfor9.py",
        circuit_profile="encoding-only",
        noise_enabled=True,
        encoder_noise=(0.05, 0.05, 0.05),
        single_gate_noise=ZERO_NOISE,
        double_gate_noise=ZERO_NOISE,
        epochs=800,
    ),
    "main_y_030": Preset(
        dataset="QNN_FP_20250704_205731",
        source_script="QNN_initial_noise_equal_noiseless_parfor10.py",
        circuit_profile="encoding-only",
        noise_enabled=True,
        encoder_noise=(0.0, 0.30, 0.0),
        single_gate_noise=ZERO_NOISE,
        double_gate_noise=ZERO_NOISE,
        epochs=800,
    ),
    "main_depolarizing_015": Preset(
        dataset="QNN_FP_20250705_212628",
        source_script="QNN_initial_noise_equal_noiseless_parfor12.py",
        circuit_profile="encoding-only",
        noise_enabled=True,
        encoder_noise=(0.15, 0.15, 0.15),
        single_gate_noise=ZERO_NOISE,
        double_gate_noise=ZERO_NOISE,
        epochs=800,
    ),
    "main_depolarizing_010": Preset(
        dataset="QNN_FP_20250715_121736",
        source_script="QNN_initial_noise_equal_noiseless_new_v2_parfor6_0.py",
        circuit_profile="all-gates",
        noise_enabled=True,
        encoder_noise=(0.10, 0.10, 0.10),
        single_gate_noise=ZERO_NOISE,
        double_gate_noise=ZERO_NOISE,
        epochs=1600,
    ),
    "allgate_y_015": Preset(
        dataset="QNN_FP_20250709_181400",
        source_script="QNN_initial_noise_equal_noiseless_new_v2_0.py",
        circuit_profile="all-gates",
        noise_enabled=True,
        encoder_noise=(0.0, 0.15, 0.0),
        single_gate_noise=HISTORICAL_SINGLE_GATE_NOISE,
        double_gate_noise=HISTORICAL_DOUBLE_GATE_NOISE,
        epochs=1600,
    ),
    "allgate_encoder_noiseless": Preset(
        dataset="QNN_FP_20250709_181438",
        source_script="QNN_initial_noise_equal_noiseless_new_v2_parfor2_0.py",
        circuit_profile="all-gates",
        noise_enabled=True,
        encoder_noise=ZERO_NOISE,
        single_gate_noise=HISTORICAL_SINGLE_GATE_NOISE,
        double_gate_noise=HISTORICAL_DOUBLE_GATE_NOISE,
        epochs=1600,
    ),
    "allgate_y_030": Preset(
        dataset="QNN_FP_20250709_181512",
        source_script="QNN_initial_noise_equal_noiseless_new_v2_parfor3_0.py",
        circuit_profile="all-gates",
        noise_enabled=True,
        encoder_noise=(0.0, 0.30, 0.0),
        single_gate_noise=HISTORICAL_SINGLE_GATE_NOISE,
        double_gate_noise=HISTORICAL_DOUBLE_GATE_NOISE,
        epochs=1600,
    ),
    "allgate_depolarizing_005": Preset(
        dataset="QNN_FP_20250709_181542",
        source_script="QNN_initial_noise_equal_noiseless_new_v2_parfor4_0.py",
        circuit_profile="all-gates",
        noise_enabled=True,
        encoder_noise=(0.05, 0.05, 0.05),
        single_gate_noise=HISTORICAL_SINGLE_GATE_NOISE,
        double_gate_noise=HISTORICAL_DOUBLE_GATE_NOISE,
        epochs=1600,
    ),
    "allgate_depolarizing_0025": Preset(
        dataset="QNN_FP_20250710_114037",
        source_script="QNN_initial_noise_equal_noiseless_new_v2_parfor5_0.py",
        circuit_profile="all-gates",
        noise_enabled=True,
        encoder_noise=(0.025, 0.025, 0.025),
        single_gate_noise=HISTORICAL_SINGLE_GATE_NOISE,
        double_gate_noise=HISTORICAL_DOUBLE_GATE_NOISE,
        epochs=1600,
    ),
}


@dataclass
class RunConfig:
    preset: str
    historical_dataset: str
    historical_source_script: str
    circuit_profile: str
    noise_enabled: bool
    encoder_noise: NoiseTriple
    single_gate_noise: NoiseTriple
    double_gate_noise: NoiseTriple
    qubits: int
    encoder_layers: int
    ansatz_layers: int
    ansatz_layers_per_block: int
    target_max_frequency: int
    target_normalization: float
    epochs: int
    grid_size: int
    batch_size: int
    evaluation_points: int
    repeats: int
    initial_method: str
    optimizer: str
    learning_rate: Optional[float]
    seed: int
    output_dir: str
    run_name: Optional[str]
    save_every_repeat: bool
    resume: Optional[str]
    plot_interval: int
    log_interval: int
    verify_sgd_step: bool


def _add_optional_number(parser: argparse.ArgumentParser, name: str, kind: Any, help_text: str) -> None:
    parser.add_argument(name, type=kind, default=None, help=help_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one parameterized QNN experiment with Adam or full-batch SGD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=tuple(PRESETS), default="main_noiseless")
    parser.add_argument("--list-presets", action="store_true", help="List historical presets and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved configuration and exit without writing files.")

    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--noise-enabled", action="store_true", dest="noise_enabled", default=None)
    profile.add_argument("--noise-disabled", action="store_false", dest="noise_enabled")
    parser.add_argument("--circuit-profile", choices=("encoding-only", "all-gates"), default=None)

    for prefix, label in (("encoder", "encoding rotation"), ("single", "other single-qubit gates"), ("double", "two-qubit gates")):
        for axis in "xyz":
            _add_optional_number(parser, f"--{prefix}-p{axis}", float, f"Pauli-{axis.upper()} probability for {label}.")

    _add_optional_number(parser, "--qubits", int, "Number of qubits.")
    _add_optional_number(parser, "--encoder-layers", int, "Number of repeated encoder blocks.")
    _add_optional_number(parser, "--ansatz-layers", int, "Number of final ansatz layers.")
    _add_optional_number(parser, "--ansatz-layers-per-block", int, "Ansatz layers after each encoder layer.")
    _add_optional_number(parser, "--target-max-frequency", int, "Largest target frequency. Historical code used 39, not 40.")
    _add_optional_number(parser, "--target-normalization", float, "Divisor applied to the target Fourier sum.")

    _add_optional_number(parser, "--epochs", int, "Training iterations per repeat.")
    _add_optional_number(parser, "--grid-size", int, "Number of training points on [0, 2*pi].")
    _add_optional_number(parser, "--batch-size", int, "Batch size; SGD requires this to equal grid-size.")
    _add_optional_number(parser, "--evaluation-points", int, "Number of points saved before every parameter update.")
    _add_optional_number(parser, "--repeats", int, "Number of independently initialized training trajectories.")
    parser.add_argument("--initial-method", choices=("ones", "normal", "random"), default=None)

    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="SGD step size eta. It must be explicit for SGD; Adam compatibility mode defaults to 0.1.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output-dir", default=None, help="New-output directory; created only for a real run.")
    parser.add_argument("--run-name", default=None, help="Output stem without .json/.pkl.")
    parser.add_argument("--resume", default=None, help="Resume after completed repeats from an existing unified .pkl file.")
    parser.add_argument("--save-every-repeat", action="store_true", default=True)
    parser.add_argument("--save-at-end", action="store_false", dest="save_every_repeat")
    parser.add_argument("--plot-interval", type=int, default=0, help="Save a diagnostic spectrum every N epochs; 0 disables it.")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--verify-sgd-step", action="store_true", help="Numerically verify theta_new = theta_old - eta*gradient on the first update.")
    parser.add_argument("--noise-family", choices=("pauli", "thermal"), default="pauli",
                        help="Pauli channels or CZ-only thermal relaxation.")
    parser.add_argument("--thermal-label", choices=("weak", "matched"), default=None)
    parser.add_argument("--thermal-t1", type=float, default=None)
    parser.add_argument("--thermal-t2", type=float, default=None)
    parser.add_argument("--thermal-gate-time", type=float, default=None)
    parser.add_argument("--preflight", action="store_true",
                        help="Check thermal channels and gradients in memory; thermal branch only.")
    return parser


def _triple_from_args(args: argparse.Namespace, prefix: str, default: NoiseTriple) -> NoiseTriple:
    values: List[float] = []
    for index, axis in enumerate("xyz"):
        value = getattr(args, f"{prefix}_p{axis}")
        values.append(default[index] if value is None else float(value))
    return tuple(values)  # type: ignore[return-value]


def resolve_config(args: argparse.Namespace) -> RunConfig:
    preset = PRESETS[args.preset]
    optimizer = args.optimizer.lower()
    learning_rate = args.learning_rate
    if optimizer == "adam" and learning_rate is None:
        learning_rate = 0.1
    output_dir = args.output_dir or ("data_gd" if optimizer == "sgd" else "data_unified")
    config = RunConfig(
        preset=args.preset,
        historical_dataset=preset.dataset,
        historical_source_script=preset.source_script,
        circuit_profile=args.circuit_profile or preset.circuit_profile,
        noise_enabled=preset.noise_enabled if args.noise_enabled is None else bool(args.noise_enabled),
        encoder_noise=_triple_from_args(args, "encoder", preset.encoder_noise),
        single_gate_noise=_triple_from_args(args, "single", preset.single_gate_noise),
        double_gate_noise=_triple_from_args(args, "double", preset.double_gate_noise),
        qubits=args.qubits if args.qubits is not None else 4,
        encoder_layers=args.encoder_layers if args.encoder_layers is not None else 10,
        ansatz_layers=args.ansatz_layers if args.ansatz_layers is not None else 1,
        ansatz_layers_per_block=args.ansatz_layers_per_block if args.ansatz_layers_per_block is not None else 4,
        target_max_frequency=args.target_max_frequency if args.target_max_frequency is not None else 39,
        target_normalization=args.target_normalization if args.target_normalization is not None else 40.0,
        epochs=args.epochs if args.epochs is not None else preset.epochs,
        grid_size=args.grid_size if args.grid_size is not None else 2048,
        batch_size=args.batch_size if args.batch_size is not None else 2048,
        evaluation_points=args.evaluation_points if args.evaluation_points is not None else 1000,
        repeats=args.repeats if args.repeats is not None else 40,
        initial_method=args.initial_method or "normal",
        optimizer=optimizer,
        learning_rate=learning_rate,
        seed=int(args.seed),
        output_dir=output_dir,
        run_name=args.run_name,
        save_every_repeat=bool(args.save_every_repeat),
        resume=args.resume,
        plot_interval=int(args.plot_interval),
        log_interval=int(args.log_interval),
        verify_sgd_step=bool(args.verify_sgd_step),
    )
    thermal_values = (args.thermal_label, args.thermal_t1, args.thermal_t2, args.thermal_gate_time)
    if args.noise_family == "thermal":
        if any(value is None for value in thermal_values):
            raise ValueError("thermal runs require --thermal-label, --thermal-t1, --thermal-t2 and --thermal-gate-time")
        if not all(math.isfinite(value) for value in thermal_values[1:]):
            raise ValueError("thermal time parameters must be finite")
        if args.thermal_t1 <= 0 or args.thermal_t2 <= 0 or args.thermal_gate_time < 0:
            raise ValueError("T1/T2 must be positive and gate time non-negative")
        attach_thermal_config(config, args)
    elif any(value is not None for value in thermal_values) or args.preflight:
        raise ValueError("thermal arguments and --preflight require --noise-family thermal")
    return config


def validate_config(config: RunConfig, *, require_learning_rate: bool) -> None:
    for label, triple in (
        ("encoder", config.encoder_noise),
        ("single", config.single_gate_noise),
        ("double", config.double_gate_noise),
    ):
        if any(value < 0 for value in triple):
            raise ValueError(f"{label} noise probabilities must be non-negative: {triple}")
        if sum(triple) > 1.0 + 1e-12:
            raise ValueError(f"{label} Pauli probabilities sum to more than one: {triple}")
    for label, value in (
        ("qubits", config.qubits),
        ("encoder_layers", config.encoder_layers),
        ("ansatz_layers", config.ansatz_layers),
        ("ansatz_layers_per_block", config.ansatz_layers_per_block),
        ("target_max_frequency", config.target_max_frequency),
        ("epochs", config.epochs),
        ("grid_size", config.grid_size),
        ("batch_size", config.batch_size),
        ("evaluation_points", config.evaluation_points),
        ("repeats", config.repeats),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be positive, got {value}")
    if config.target_normalization <= 0:
        raise ValueError("target_normalization must be positive")
    if config.circuit_profile == "encoding-only" and (
        any(config.single_gate_noise) or any(config.double_gate_noise)
    ):
        raise ValueError("encoding-only profile requires zero noise on other single/double-qubit gates")
    if config.batch_size != config.grid_size:
        raise ValueError("this unified workflow is full-batch: --batch-size must equal --grid-size")
    if require_learning_rate and config.learning_rate is None:
        raise ValueError("--learning-rate must be provided explicitly for SGD training")
    if config.learning_rate is not None and config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.plot_interval < 0 or config.log_interval <= 0:
        raise ValueError("plot_interval must be non-negative and log_interval must be positive")
    if config.run_name and Path(config.run_name).name != config.run_name:
        raise ValueError("run_name must be a plain file stem, not a path")
    if hasattr(config, "thermal_t1"):
        validate_thermal_config(config)


def config_as_jsonable(config: RunConfig) -> Dict[str, Any]:
    data = asdict(config)
    data["encoder_noise"] = list(config.encoder_noise)
    data["single_gate_noise"] = list(config.single_gate_noise)
    data["double_gate_noise"] = list(config.double_gate_noise)
    if hasattr(config, "thermal_t1"):
        data["thermal_relaxation"] = {
            "label": config.thermal_label,
            "t1": config.thermal_t1,
            "t2": config.thermal_t2,
            "gate_time": config.thermal_gate_time,
            "gate_time_over_t1": config.thermal_gate_time_over_t1,
            "t2_over_t1": config.thermal_t2_over_t1,
            "gamma_1": config.thermal_gamma_1,
            "phase_flip_probability": config.thermal_phase_flip_probability,
            "average_gate_infidelity": config.thermal_average_gate_infidelity,
            "noisy_cz_count": config.noisy_cz_count,
            "local_thermal_channel_count": config.local_thermal_channel_count,
        }
    return data


def print_presets() -> None:
    print("preset\tdataset\tsource_script\tprofile\tencoder(px,py,pz)\tepochs")
    for name, preset in PRESETS.items():
        print(
            f"{name}\t{preset.dataset}\t{preset.source_script}\t{preset.circuit_profile}\t"
            f"{preset.encoder_noise}\t{preset.epochs}"
        )


def _fingerprint_payload(config: RunConfig) -> Dict[str, Any]:
    payload = config_as_jsonable(config)
    for key in ("output_dir", "run_name", "save_every_repeat", "resume", "plot_interval", "log_interval", "verify_sgd_step", "repeats"):
        payload.pop(key, None)
    return payload


def config_fingerprint(config: RunConfig) -> str:
    encoded = json.dumps(_fingerprint_payload(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def metadata_from_config(config: RunConfig, run_name: str, status: str, completed_repeats: int) -> Dict[str, Any]:
    px, py, pz = config.encoder_noise
    sx, sy, sz = config.single_gate_noise
    dx, dy, dz = config.double_gate_noise
    metadata = {
        "schema_version": 2,
        "script": {"filename_this": SCRIPT_NAME, "version": SCRIPT_VERSION},
        "historical_provenance": {
            "preset": config.preset,
            "dataset": config.historical_dataset,
            "source_script": config.historical_source_script,
        },
        "noise_settings": {
            "Is_noise": config.noise_enabled,
            "circuit_profile": config.circuit_profile,
            "noise_rate_x": px,
            "noise_rate_y": py,
            "noise_rate_z": pz,
            "noise_rate_single_x": sx,
            "noise_rate_single_y": sy,
            "noise_rate_single_z": sz,
            "noise_rate_double_x": dx,
            "noise_rate_double_y": dy,
            "noise_rate_double_z": dz,
        },
        "circuit_settings": {
            "qubit_num": config.qubits,
            "encoder_layer_num": config.encoder_layers,
            "ansatz_layer_num": config.ansatz_layers,
            "ansatz_layer_num_in_block": config.ansatz_layers_per_block,
            "filename_this": SCRIPT_NAME,
        },
        "target_settings": {
            "max_frequency": config.target_max_frequency,
            "normalization": config.target_normalization,
            "domain": [0.0, 2.0 * PI],
        },
        "training_settings": {
            "epochs": config.epochs,
            "grid_size": config.grid_size,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "initial_method": config.initial_method,
            "repeat_num": config.repeats,
            "seed": config.seed,
        },
        "optimizer_settings": {
            "name": config.optimizer,
            "learning_rate": config.learning_rate,
            "momentum": 0.0 if config.optimizer == "sgd" else None,
            "weight_decay": 0.0,
            "nesterov": False,
            "update_mode": "full-batch",
        },
        "run_state": {
            "run_name": run_name,
            "status": status,
            "requested_repeats": config.repeats,
            "completed_repeats": completed_repeats,
            "config_fingerprint": config_fingerprint(config),
            "updated_at": _datetime.datetime.now().isoformat(timespec="seconds"),
        },
    }
    if hasattr(config, "thermal_t1"):
        metadata["historical_provenance"]["role"] = "architecture and final-ansatz placement only"
        metadata["noise_settings"].update(
            {
                "noise_family": "thermal_relaxation_zero_temperature",
                "training_implementation": "amplitude_damping_plus_calibrated_pauli_z_dephasing",
                "native_channel_role": "forward_preflight_only",
                "placement": "after_repeated-block_CZ_on_both_participants",
                "single_qubit_rotations_noiseless": True,
                "final_ansatz_noiseless": True,
                "correlated_two_qubit_noise": False,
                "thermal_label": config.thermal_label,
                "thermal_t1": config.thermal_t1,
                "thermal_t2": config.thermal_t2,
                "thermal_gate_time": config.thermal_gate_time,
                "thermal_gate_time_over_t1": config.thermal_gate_time_over_t1,
                "thermal_t2_over_t1": config.thermal_t2_over_t1,
                "thermal_gamma_1": config.thermal_gamma_1,
                "thermal_phase_flip_probability": config.thermal_phase_flip_probability,
                "thermal_local_average_gate_infidelity": config.thermal_average_gate_infidelity,
                "noisy_cz_count": config.noisy_cz_count,
                "local_thermal_channel_count": config.local_thermal_channel_count,
            }
        )
    return metadata


def _atomic_json_dump(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_pickle_dump(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=str(path.parent), prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _target_values(np: Any, values: Any, max_frequency: int, normalization: float) -> Any:
    result = np.zeros_like(values, dtype=np.float64)
    for frequency in range(1, max_frequency + 1):
        result += np.sin(frequency * values) + np.cos(frequency * values)
    return result / normalization


def _build_circuits(runtime: Mapping[str, Any], config: RunConfig) -> Tuple[Any, Any]:
    if hasattr(config, "thermal_t1"):
        return _build_thermal_circuits(runtime, config, "decomposed")
    Circuit = runtime["Circuit"]
    RY = runtime["RY"]
    Z = runtime["Z"]
    PauliChannel = runtime["PauliChannel"]
    PRGenerator = runtime["PRGenerator"]

    clean = Circuit()
    clean_prg = PRGenerator("beta")
    for _ in range(config.encoder_layers):
        encoder = Circuit()
        for qubit in range(config.qubits):
            encoder += RY("alpha").on(qubit)
        for qubit in range(config.qubits - 1):
            encoder += Z.on(qubit + 1, qubit)
        clean += encoder.as_encoder()
        for _ in range(config.ansatz_layers_per_block):
            ansatz = Circuit()
            for qubit in range(config.qubits):
                ansatz += RY(clean_prg.new()).on(qubit)
            for qubit in range(config.qubits - 1):
                ansatz += Z.on(qubit + 1, qubit)
            clean += ansatz.as_ansatz()
    clean.barrier()
    for _ in range(config.ansatz_layers):
        ansatz = Circuit()
        for qubit in range(config.qubits):
            ansatz += RY(clean_prg.new()).on(qubit)
        for qubit in range(config.qubits - 1):
            ansatz += Z.on(qubit + 1, qubit)
        clean += ansatz.as_ansatz()

    noisy = Circuit()
    noisy_prg = PRGenerator("beta")
    ex, ey, ez = config.encoder_noise
    sx, sy, sz = config.single_gate_noise
    dx, dy, dz = config.double_gate_noise
    for _ in range(config.encoder_layers):
        encoder = Circuit()
        for qubit in range(config.qubits):
            encoder += PauliChannel(ex, ey, ez).on(qubit)
            encoder += RY("alpha").on(qubit)
        for qubit in range(config.qubits - 1):
            encoder += Z.on(qubit + 1, qubit)
            if config.circuit_profile == "all-gates":
                encoder += PauliChannel(dx, dy, dz).on(qubit + 1)
                encoder += PauliChannel(dx, dy, dz).on(qubit)
        noisy += encoder.as_encoder()

        for _ in range(config.ansatz_layers_per_block):
            ansatz = Circuit()
            for qubit in range(config.qubits):
                ansatz += RY(noisy_prg.new()).on(qubit)
                if config.circuit_profile == "all-gates":
                    ansatz += PauliChannel(sx, sy, sz).on(qubit)
            for qubit in range(config.qubits - 1):
                ansatz += Z.on(qubit + 1, qubit)
                if config.circuit_profile == "all-gates":
                    ansatz += PauliChannel(dx, dy, dz).on(qubit + 1)
                    ansatz += PauliChannel(dx, dy, dz).on(qubit)
            noisy += ansatz.as_ansatz()

    noisy.barrier()
    for _ in range(config.ansatz_layers):
        ansatz = Circuit()
        for qubit in range(config.qubits):
            ansatz += RY(noisy_prg.new()).on(qubit)
        for qubit in range(config.qubits - 1):
            ansatz += Z.on(qubit + 1, qubit)
        noisy += ansatz.as_ansatz()
    return clean, noisy


def _load_runtime() -> Dict[str, Any]:
    try:
        import numpy as np
        import mindspore as ms
        import mindspore.nn as nn
        from mindspore import Parameter
        from mindspore.common.initializer import initializer
        from mindquantum.core.circuit import Circuit
        from mindquantum.core.gates import PauliChannel, RY, Z, AmplitudeDampingChannel, ThermalRelaxationChannel
        from mindquantum.core.operators import Hamiltonian, QubitOperator
        from mindquantum.core.parameterresolver import PRGenerator
        from mindquantum.framework import MQOps
        from mindquantum.simulator import Simulator
    except ImportError as exc:
        raise RuntimeError(
            "A real run needs MindSpore and MindQuantum. On the inspected server, "
            "use the existing mindspore_cpu or mindspore_gpu conda environment."
        ) from exc
    return locals()


def _create_model_class(runtime: Mapping[str, Any]) -> Any:
    nn = runtime["nn"]
    ms = runtime["ms"]
    Parameter = runtime["Parameter"]
    initializer = runtime["initializer"]
    MQOps = runtime["MQOps"]

    class MQLayer(nn.Cell):
        def __init__(self, expectation_with_grad: Any, weight: str = "normal") -> None:
            super().__init__()
            self.evolution = MQOps(expectation_with_grad)
            weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
            self.weight = Parameter(
                initializer(weight, weight_size, dtype=ms.float32),
                name="ansatz_weight",
            )
            self.dense_relu_sequential = nn.SequentialCell(
                nn.Dense(1, 1, weight_init="normal", bias_init="zeros")
            )

        def construct(self, values: Any) -> Any:
            return self.dense_relu_sequential(self.evolution(values, self.weight))

    return MQLayer


def _make_optimizer(runtime: Mapping[str, Any], model: Any, config: RunConfig) -> Any:
    nn = runtime["nn"]
    if config.learning_rate is None:
        raise ValueError("learning_rate is required for a real training run")
    if config.optimizer == "sgd":
        return nn.SGD(
            model.trainable_params(),
            learning_rate=config.learning_rate,
            momentum=0.0,
            dampening=0.0,
            weight_decay=0.0,
            nesterov=False,
        )
    return nn.Adam(
        model.trainable_params(),
        learning_rate=config.learning_rate,
        weight_decay=0.0,
    )


def _parameter_snapshot(model: Any) -> List[Dict[str, Any]]:
    return [
        {"name": parameter.name, "shape": tuple(parameter.shape), "value": parameter.asnumpy().copy()}
        for parameter in model.trainable_params()
    ]


def _save_diagnostic(np: Any, output_dir: Path, run_name: str, repeat_index: int, epoch: int, values: Any) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(values)
    spacing = 2.0 * np.pi / count
    frequencies = np.fft.fftfreq(count, spacing)[: count // 2]
    amplitudes = 2.0 / count * np.abs(np.fft.fft(values)[: count // 2])
    diagnostic_dir = output_dir / f"{run_name}_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(frequencies[:40], amplitudes[:40])
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Amplitude")
    axis.set_title(f"repeat={repeat_index + 1}, epoch={epoch}")
    figure.tight_layout()
    figure.savefig(diagnostic_dir / f"repeat_{repeat_index + 1:03d}_epoch_{epoch:05d}.png", dpi=120)
    plt.close(figure)


def _initial_result_state() -> Dict[str, Any]:
    return {
        "result_data": {
            "ans_x_list_list": [],
            "ans_y_list_list": [],
            "target_x": None,
            "target_y": None,
            "model_final_x": None,
            "model_final_y": None,
            "model_final_x_list": [],
            "model_final_y_list": [],
            "loss_history_list": [],
            "gradient_norm_history_list": [],
        },
        "model_para_list_list": [],
        "sgd_step_verification": None,
    }


def _load_resume(path: Path, expected_fingerprint: str) -> Dict[str, Any]:
    with path.open("rb") as handle:
        state = pickle.load(handle)
    saved = state.get("run_state", {}).get("config_fingerprint")
    if saved != expected_fingerprint:
        raise ValueError(
            "Resume file configuration does not match the requested run. "
            f"saved={saved}, requested={expected_fingerprint}"
        )
    return state


def run_training(config: RunConfig) -> Tuple[Path, Path]:
    runtime = _load_runtime()
    np = runtime["np"]
    ms = runtime["ms"]
    nn = runtime["nn"]
    Simulator = runtime["Simulator"]
    Hamiltonian = runtime["Hamiltonian"]
    QubitOperator = runtime["QubitOperator"]

    output_dir = Path(config.output_dir).expanduser().resolve()
    timestamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.run_name or f"QNN_{config.optimizer.upper()}_{timestamp}_{config.preset}"
    json_path = output_dir / f"{run_name}.json"
    pickle_path = output_dir / f"{run_name}.pkl"

    if config.resume:
        pickle_path = Path(config.resume).expanduser().resolve()
        if pickle_path.suffix != ".pkl":
            raise ValueError("--resume must point to a .pkl file")
        json_path = pickle_path.with_suffix(".json")
        run_name = pickle_path.stem
        state = _load_resume(pickle_path, config_fingerprint(config))
    else:
        if json_path.exists() or pickle_path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing output: {json_path} or {pickle_path}")
        state = _initial_result_state()

    result_data = state["result_data"]
    completed = len(result_data["ans_y_list_list"])
    if completed > config.repeats:
        raise ValueError(f"Resume file already has {completed} repeats, exceeding requested {config.repeats}")

    clean_circuit, noisy_circuit = _build_circuits(runtime, config)
    active_circuit = noisy_circuit if config.noise_enabled else clean_circuit
    simulator_name = "mqmatrix" if config.noise_enabled else "mqvector"
    hamiltonian = Hamiltonian(QubitOperator("Z0"))
    gradient_operator = Simulator(simulator_name, active_circuit.n_qubits).get_expectation_with_grad(
        hamiltonian,
        active_circuit,
    )
    MQLayer = _create_model_class(runtime)

    train_x = np.linspace(0.0, 2.0 * np.pi, config.grid_size).reshape(-1, 1).astype(np.float32)
    train_y = _target_values(
        np,
        train_x,
        config.target_max_frequency,
        config.target_normalization,
    ).reshape(-1, 1).astype(np.float32)
    train_x_tensor = ms.Tensor(train_x, ms.float32)
    train_y_tensor = ms.Tensor(train_y, ms.float32)
    evaluation_x = np.linspace(0.0, 2.0 * np.pi, config.evaluation_points).astype(np.float32)
    target_x = evaluation_x.astype(np.float64)
    target_y = _target_values(np, target_x, config.target_max_frequency, config.target_normalization)
    result_data["target_x"] = target_x
    result_data["target_y"] = target_y

    def persist(status: str) -> None:
        metadata = metadata_from_config(config, run_name, status, len(result_data["ans_y_list_list"]))
        state.update(metadata)
        _atomic_json_dump(json_path, metadata)
        _atomic_pickle_dump(pickle_path, state)

    persist("running")
    for repeat_index in range(completed, config.repeats):
        repeat_seed = config.seed + repeat_index
        np.random.seed(repeat_seed)
        ms.set_seed(repeat_seed)
        model = MQLayer(gradient_operator, weight=config.initial_method)
        model.set_train()
        loss_function = nn.MSELoss()
        optimizer = _make_optimizer(runtime, model, config)

        def forward_fn(data: Any, label: Any) -> Tuple[Any, Any]:
            prediction = model(data)
            return loss_function(prediction, label), prediction

        gradient_fn = ms.value_and_grad(
            forward_fn,
            None,
            optimizer.parameters,
            has_aux=True,
        )

        answer_x: List[Any] = []
        answer_y: List[Any] = []
        losses: List[float] = []
        gradient_norms: List[float] = []
        first_step_verification: Optional[Dict[str, float]] = None

        print(f"Repeat {repeat_index + 1}/{config.repeats} (seed={repeat_seed})")
        for epoch in range(config.epochs):
            prediction_before = model(ms.Tensor(evaluation_x.reshape(-1, 1), ms.float32)).asnumpy().reshape(-1)
            answer_x.append(evaluation_x.copy())
            answer_y.append(prediction_before.copy())

            parameters_before = None
            if config.verify_sgd_step and config.optimizer == "sgd" and repeat_index == completed and epoch == 0:
                parameters_before = [parameter.asnumpy().copy() for parameter in optimizer.parameters]

            (loss, _), gradients = gradient_fn(train_x_tensor, train_y_tensor)
            gradient_arrays = [gradient.asnumpy() for gradient in gradients]
            gradient_norm = float(math.sqrt(sum(float(np.sum(array * array)) for array in gradient_arrays)))
            optimizer(gradients)
            loss_value = float(loss.asnumpy())
            if not math.isfinite(loss_value) or not math.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"Non-finite training value at repeat={repeat_index + 1}, epoch={epoch}: "
                    f"loss={loss_value}, gradient_norm={gradient_norm}"
                )
            losses.append(loss_value)
            gradient_norms.append(gradient_norm)

            if parameters_before is not None:
                residuals = []
                for before, gradient, parameter in zip(parameters_before, gradient_arrays, optimizer.parameters):
                    expected = before - float(config.learning_rate) * gradient
                    residuals.append(float(np.max(np.abs(parameter.asnumpy() - expected))))
                first_step_verification = {
                    "max_abs_update_residual": max(residuals, default=0.0),
                    "learning_rate": float(config.learning_rate),
                }
                state["sgd_step_verification"] = first_step_verification
                print(f"SGD one-step max residual: {first_step_verification['max_abs_update_residual']:.3e}")

            if epoch % config.log_interval == 0 or epoch + 1 == config.epochs:
                print(
                    f"  epoch={epoch + 1}/{config.epochs} "
                    f"loss={loss_value:.8g} grad_norm={gradient_norm:.8g}"
                )
            if config.plot_interval and epoch % config.plot_interval == 0:
                _save_diagnostic(np, output_dir, run_name, repeat_index, epoch, prediction_before)

        final_y = model(ms.Tensor(evaluation_x.reshape(-1, 1), ms.float32)).asnumpy().reshape(-1)
        result_data["ans_x_list_list"].append(answer_x)
        result_data["ans_y_list_list"].append(answer_y)
        result_data["model_final_x"] = evaluation_x.copy()
        result_data["model_final_y"] = final_y.copy()
        result_data["model_final_x_list"].append(evaluation_x.copy())
        result_data["model_final_y_list"].append(final_y.copy())
        result_data["loss_history_list"].append(losses)
        result_data["gradient_norm_history_list"].append(gradient_norms)
        state["model_para_list_list"].append(_parameter_snapshot(model))
        if config.save_every_repeat:
            persist("running")

    persist("complete")
    return json_path, pickle_path


def thermal_parameters(t1: float, t2: float, gate_time: float) -> dict[str, float]:
    gamma_1 = 1.0 - math.exp(-gate_time / t1)
    pure_dephasing_rate = 1.0 / t2 - 1.0 / (2.0 * t1)
    if pure_dephasing_rate < -1e-15:
        raise ValueError("T2 must not exceed 2*T1 for the thermal decomposition")
    phase_flip_probability = (1.0 - math.exp(-gate_time * max(pure_dephasing_rate, 0.0))) / 2.0
    average_infidelity = (3.0 - 2.0 * math.exp(-gate_time / t2) - math.exp(-gate_time / t1)) / 6.0
    return {
        "gamma_1": gamma_1,
        "phase_flip_probability": phase_flip_probability,
        "average_gate_infidelity": average_infidelity,
        "gate_time_over_t1": gate_time / t1,
        "t2_over_t1": t2 / t1,
    }

def attach_thermal_config(config: Any, args: argparse.Namespace) -> None:
    config.thermal_label = str(args.thermal_label)
    config.thermal_t1 = float(args.thermal_t1)
    config.thermal_t2 = float(args.thermal_t2)
    config.thermal_gate_time = float(args.thermal_gate_time)
    derived = thermal_parameters(config.thermal_t1, config.thermal_t2, config.thermal_gate_time)
    for key, value in derived.items():
        setattr(config, f"thermal_{key}", float(value))
    config.noisy_cz_count = config.encoder_layers * (config.qubits - 1) * (
        1 + config.ansatz_layers_per_block
    )
    config.local_thermal_channel_count = 2 * config.noisy_cz_count

def validate_thermal_config(config: Any) -> None:
    if not config.noise_enabled:
        raise ValueError("thermal experiment requires --noise-enabled")
    if config.circuit_profile != "all-gates":
        raise ValueError("thermal experiment requires --circuit-profile all-gates")
    if any(config.encoder_noise) or any(config.single_gate_noise) or any(config.double_gate_noise):
        raise ValueError("legacy Pauli triples must all be zero in the thermal-only experiment")
    if config.thermal_t1 <= 0.0 or config.thermal_t2 <= 0.0:
        raise ValueError("T1 and T2 must be positive")
    if config.thermal_gate_time < 0.0:
        raise ValueError("gate time must be non-negative")
    if config.thermal_t2 >= 2.0 * config.thermal_t1:
        raise ValueError("MindQuantum ThermalRelaxationChannel requires T2 < 2*T1")
    if config.qubits != 4:
        raise ValueError("this controlled experiment is locked to four qubits")

def _build_thermal_circuits(runtime: Mapping[str, Any], config: Any, implementation: str) -> tuple[Any, Any]:
    Circuit = runtime["Circuit"]
    RY = runtime["RY"]
    Z = runtime["Z"]
    PauliChannel = runtime["PauliChannel"]
    AmplitudeDampingChannel = runtime["AmplitudeDampingChannel"]
    ThermalRelaxationChannel = runtime["ThermalRelaxationChannel"]
    PRGenerator = runtime["PRGenerator"]

    def append_thermal(circuit: Any, qubit: int) -> None:
        if implementation == "native":
            circuit += ThermalRelaxationChannel(
                config.thermal_t1,
                config.thermal_t2,
                config.thermal_gate_time,
            ).on(qubit)
        elif implementation == "decomposed":
            circuit += AmplitudeDampingChannel(config.thermal_gamma_1).on(qubit)
            circuit += PauliChannel(0.0, 0.0, config.thermal_phase_flip_probability).on(qubit)
        else:
            raise ValueError(f"unknown thermal implementation: {implementation}")

    clean = Circuit()
    clean_prg = PRGenerator("beta")
    for _ in range(config.encoder_layers):
        encoder = Circuit()
        for qubit in range(config.qubits):
            encoder += RY("alpha").on(qubit)
        for qubit in range(config.qubits - 1):
            encoder += Z.on(qubit + 1, qubit)
        clean += encoder.as_encoder()
        for _ in range(config.ansatz_layers_per_block):
            ansatz = Circuit()
            for qubit in range(config.qubits):
                ansatz += RY(clean_prg.new()).on(qubit)
            for qubit in range(config.qubits - 1):
                ansatz += Z.on(qubit + 1, qubit)
            clean += ansatz.as_ansatz()
    clean.barrier()
    for _ in range(config.ansatz_layers):
        ansatz = Circuit()
        for qubit in range(config.qubits):
            ansatz += RY(clean_prg.new()).on(qubit)
        for qubit in range(config.qubits - 1):
            ansatz += Z.on(qubit + 1, qubit)
        clean += ansatz.as_ansatz()

    noisy = Circuit()
    noisy_prg = PRGenerator("beta")
    for _ in range(config.encoder_layers):
        encoder = Circuit()
        for qubit in range(config.qubits):
            encoder += RY("alpha").on(qubit)
        for qubit in range(config.qubits - 1):
            encoder += Z.on(qubit + 1, qubit)
            append_thermal(encoder, qubit + 1)
            append_thermal(encoder, qubit)
        noisy += encoder.as_encoder()

        for _ in range(config.ansatz_layers_per_block):
            ansatz = Circuit()
            for qubit in range(config.qubits):
                ansatz += RY(noisy_prg.new()).on(qubit)
            for qubit in range(config.qubits - 1):
                ansatz += Z.on(qubit + 1, qubit)
                append_thermal(ansatz, qubit + 1)
                append_thermal(ansatz, qubit)
            noisy += ansatz.as_ansatz()

    noisy.barrier()
    for _ in range(config.ansatz_layers):
        ansatz = Circuit()
        for qubit in range(config.qubits):
            ansatz += RY(noisy_prg.new()).on(qubit)
        for qubit in range(config.qubits - 1):
            ansatz += Z.on(qubit + 1, qubit)
        noisy += ansatz.as_ansatz()
    return clean, noisy

def run_thermal_preflight(config: Any) -> dict[str, Any]:
    runtime = _load_runtime()
    np = runtime["np"]
    Simulator = runtime["Simulator"]
    Hamiltonian = runtime["Hamiltonian"]
    QubitOperator = runtime["QubitOperator"]
    AmplitudeDampingChannel = runtime["AmplitudeDampingChannel"]
    PauliChannel = runtime["PauliChannel"]
    ThermalRelaxationChannel = runtime["ThermalRelaxationChannel"]

    _, decomposed = _build_thermal_circuits(runtime, config, "decomposed")
    _, native = _build_thermal_circuits(runtime, config, "native")
    ad_count = sum(isinstance(gate, AmplitudeDampingChannel) for gate in decomposed)
    pz_count = sum(
        isinstance(gate, PauliChannel)
        and math.isclose(float(gate.px), 0.0, abs_tol=1e-15)
        and math.isclose(float(gate.py), 0.0, abs_tol=1e-15)
        and math.isclose(float(gate.pz), config.thermal_phase_flip_probability, rel_tol=0.0, abs_tol=1e-15)
        for gate in decomposed
    )
    native_count = sum(isinstance(gate, ThermalRelaxationChannel) for gate in native)
    expected = config.local_thermal_channel_count
    if (ad_count, pz_count, native_count) != (expected, expected, expected):
        raise RuntimeError(
            f"unexpected channel counts: AD={ad_count}, PZ={pz_count}, native={native_count}, expected={expected}"
        )

    ad_ops = AmplitudeDampingChannel(config.thermal_gamma_1).matrix()
    pz_ops = PauliChannel(0.0, 0.0, config.thermal_phase_flip_probability).matrix()
    native_ops = ThermalRelaxationChannel(
        config.thermal_t1,
        config.thermal_t2,
        config.thermal_gate_time,
    ).matrix()
    states = (
        np.diag([1.0, 0.0]).astype(complex),
        np.diag([0.0, 1.0]).astype(complex),
        np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex),
        np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=complex),
    )
    channel_map_max_abs = 0.0
    trace_max_abs = 0.0
    minimum_eigenvalue = 1.0
    for rho in states:
        after_ad = sum(k @ rho @ k.conj().T for k in ad_ops)
        decomposed_rho = sum(k @ after_ad @ k.conj().T for k in pz_ops)
        native_rho = sum(k @ rho @ k.conj().T for k in native_ops)
        channel_map_max_abs = max(channel_map_max_abs, float(np.max(np.abs(decomposed_rho - native_rho))))
        trace_max_abs = max(trace_max_abs, float(abs(np.trace(decomposed_rho) - 1.0)))
        minimum_eigenvalue = min(minimum_eigenvalue, float(np.min(np.linalg.eigvalsh(decomposed_rho))))

    ham = Hamiltonian(QubitOperator("Z0"))
    decomp_fn = Simulator("mqmatrix", config.qubits).get_expectation_with_grad(ham, decomposed)
    native_fn = Simulator("mqmatrix", config.qubits).get_expectation_with_grad(ham, native)
    inputs = np.asarray([[0.2], [1.1], [2.4], [5.5]], dtype=float)
    parameter_count = len(decomp_fn.ansatz_params_name)
    theta = np.linspace(-0.2, 0.2, parameter_count, dtype=float)
    decomp_result = decomp_fn(inputs, theta)
    native_result = native_fn(inputs, theta)
    values = np.asarray(decomp_result[0]).real.reshape(len(inputs))
    jacobian = np.asarray(decomp_result[-1]).real.reshape(len(inputs), parameter_count)
    native_values = np.asarray(native_result[0]).real.reshape(len(inputs))
    full_circuit_forward_max_abs = float(np.max(np.abs(values - native_values)))

    selected = sorted(set((0, 1, parameter_count // 4, parameter_count // 2, parameter_count - 1)))
    shift_max_abs = 0.0
    loss_finite_difference_max_abs = 0.0
    target = _target_values(np, inputs.reshape(-1), config.target_max_frequency, config.target_normalization)
    for index in selected:
        shift = np.zeros(parameter_count, dtype=float)
        shift[index] = np.pi / 2.0
        plus = np.asarray(decomp_fn(inputs, theta + shift)[0]).real.reshape(len(inputs))
        minus = np.asarray(decomp_fn(inputs, theta - shift)[0]).real.reshape(len(inputs))
        shifted = (plus - minus) / 2.0
        shift_max_abs = max(shift_max_abs, float(np.max(np.abs(shifted - jacobian[:, index]))))

        analytic_loss_gradient = float(np.mean(2.0 * (values - target) * jacobian[:, index]))
        epsilon = 1e-4
        delta = np.zeros(parameter_count, dtype=float)
        delta[index] = epsilon
        plus_value = np.asarray(decomp_fn(inputs, theta + delta)[0]).real.reshape(len(inputs))
        minus_value = np.asarray(decomp_fn(inputs, theta - delta)[0]).real.reshape(len(inputs))
        plus_loss = float(np.mean((plus_value - target) ** 2))
        minus_loss = float(np.mean((minus_value - target) ** 2))
        finite = (plus_loss - minus_loss) / (2.0 * epsilon)
        loss_finite_difference_max_abs = max(
            loss_finite_difference_max_abs,
            abs(finite - analytic_loss_gradient),
        )

    checks = {
        "amplitude_damping_gate_count": ad_count,
        "channel_map_max_abs": channel_map_max_abs,
        "full_circuit_forward_max_abs": full_circuit_forward_max_abs,
        "jacobian_parameter_shift_max_abs": shift_max_abs,
        "local_thermal_channel_count": expected,
        "loss_gradient_finite_difference_max_abs": loss_finite_difference_max_abs,
        "minimum_density_eigenvalue": minimum_eigenvalue,
        "native_thermal_gate_count": native_count,
        "noisy_cz_count": config.noisy_cz_count,
        "parameter_count": parameter_count,
        "phase_flip_gate_count": pz_count,
        "selected_parameter_indices": selected,
        "trace_max_abs": trace_max_abs,
    }
    if channel_map_max_abs > 1e-12:
        raise RuntimeError(f"native/decomposed channel mismatch: {channel_map_max_abs}")
    if full_circuit_forward_max_abs > 1e-10:
        raise RuntimeError(f"native/decomposed circuit mismatch: {full_circuit_forward_max_abs}")
    if trace_max_abs > 1e-12 or minimum_eigenvalue < -1e-12:
        raise RuntimeError("channel is not numerically CPTP")
    if shift_max_abs > 1e-8:
        raise RuntimeError(f"parameter-shift Jacobian mismatch: {shift_max_abs}")
    if loss_finite_difference_max_abs > 1e-6:
        raise RuntimeError(f"loss-gradient finite-difference mismatch: {loss_finite_difference_max_abs}")
    return {
        "passed": True,
        "script": Path(__file__).name,
        "thermal_label": config.thermal_label,
        "thermal_parameters": config_as_jsonable(config)["thermal_relaxation"],
        "checks": checks,
    }

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_presets:
        print_presets()
        return 0
    try:
        config = resolve_config(args)
        validate_config(config, require_learning_rate=not (args.dry_run or args.preflight))
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        payload = {
            "script": SCRIPT_NAME,
            "version": SCRIPT_VERSION,
            "writes_files": False,
            "training_ready": config.learning_rate is not None,
            "config_fingerprint": config_fingerprint(config),
            "resolved_config": config_as_jsonable(config),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.preflight:
        print(json.dumps(run_thermal_preflight(config), indent=2, sort_keys=True))
        return 0

    try:
        json_path, pickle_path = run_training(config)
    except (FileExistsError, FloatingPointError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"JSON: {json_path}")
    print(f"PKL:  {pickle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
