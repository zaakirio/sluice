from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

TIERS = ("simple", "moderate", "complex")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class BackendConfig:
    name: str
    type: str
    model: str
    base_url: str
    input_cost_per_mtok: float
    output_cost_per_mtok: float
    timeout_s: float
    api_key_env: str | None

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


@dataclass(frozen=True)
class ReliabilityConfig:
    max_retries: int
    backoff_base_s: float
    backoff_max_s: float
    circuit_failure_threshold: int
    circuit_reset_s: float


@dataclass(frozen=True)
class Config:
    backends: dict[str, BackendConfig]
    policies: dict[str, dict[str, list[str]]]
    reliability: ReliabilityConfig
    default_policy: str
    ledger_path: str


def _require(mapping: dict, key: str, ctx: str):
    if not isinstance(mapping, dict) or key not in mapping:
        raise ConfigError(f"{ctx}: missing required key {key!r}")
    return mapping[key]


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: config file is empty or not a mapping")

    backends: dict[str, BackendConfig] = {}
    for name, b in _require(raw, "backends", str(path)).items():
        ctx = f"backend {name!r}"
        btype = _require(b, "type", ctx)
        if btype not in ("openai", "anthropic"):
            raise ConfigError(f"{ctx}: unknown type {btype!r}")
        backends[name] = BackendConfig(
            name=name,
            type=btype,
            model=_require(b, "model", ctx),
            base_url=_require(b, "base_url", ctx).rstrip("/"),
            input_cost_per_mtok=float(_require(b, "input_cost_per_mtok", ctx)),
            output_cost_per_mtok=float(_require(b, "output_cost_per_mtok", ctx)),
            timeout_s=float(b.get("timeout_s", 60.0)),
            api_key_env=b.get("api_key_env"),
        )

    policies: dict[str, dict[str, list[str]]] = {}
    for pname, tiers in _require(raw, "policies", str(path)).items():
        policy: dict[str, list[str]] = {}
        for tier in TIERS:
            if tier not in tiers:
                raise ConfigError(f"policy {pname!r} missing tier {tier!r}")
            chain = list(tiers[tier])
            if not chain:
                raise ConfigError(f"policy {pname!r} tier {tier!r} has empty chain")
            for backend_name in chain:
                if backend_name not in backends:
                    raise ConfigError(
                        f"policy {pname!r} tier {tier!r} references unknown backend {backend_name!r}"
                    )
            policy[tier] = chain
        policies[pname] = policy

    default_policy = raw.get("default_policy", "balanced")
    if default_policy not in policies:
        raise ConfigError(f"default_policy {default_policy!r} is not a defined policy")

    rel = raw.get("reliability", {})
    reliability = ReliabilityConfig(
        max_retries=int(rel.get("max_retries", 2)),
        backoff_base_s=float(rel.get("backoff_base_s", 0.25)),
        backoff_max_s=float(rel.get("backoff_max_s", 4.0)),
        circuit_failure_threshold=int(rel.get("circuit_failure_threshold", 5)),
        circuit_reset_s=float(rel.get("circuit_reset_s", 30.0)),
    )

    return Config(
        backends=backends,
        policies=policies,
        reliability=reliability,
        default_policy=default_policy,
        ledger_path=raw.get("ledger_path", "sluice.db"),
    )
