from __future__ import annotations

from dataclasses import dataclass

from .config import BackendConfig, Config

# Deterministic tier thresholds (estimated prompt tokens, chars/4).
COMPLEX_TOKENS = 1500
MODERATE_TOKENS = 300

CODE_MARKERS = (
    "```",
    "def ",
    "class ",
    "import ",
    "#include",
    "function ",
    "SELECT ",
    "=> ",
    "traceback",
    "Traceback",
)

# Assumed completion length for pre-flight cost estimates when the client
# doesn't set max_tokens.
DEFAULT_COMPLETION_ESTIMATE = 512


class NoRouteError(Exception):
    pass


@dataclass(frozen=True)
class RouteDecision:
    policy: str
    tier: str
    chain: list[str]
    reason: str
    est_costs: dict[str, float]


def cost_usd(cfg: BackendConfig, prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * cfg.input_cost_per_mtok
        + completion_tokens * cfg.output_cost_per_mtok
    ) / 1_000_000


def _prompt_text(messages: list[dict]) -> str:
    parts: list[str] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(p.get("text", "") for p in content if isinstance(p, dict))
    return "\n".join(parts)


def estimate_prompt_tokens(messages: list[dict]) -> int:
    return max(1, len(_prompt_text(messages)) // 4)


def classify(messages: list[dict], has_tools: bool) -> tuple[str, list[str]]:
    text = _prompt_text(messages)
    tokens = max(1, len(text) // 4)
    signals = [f"prompt~{tokens}tok"]
    has_code = any(marker in text for marker in CODE_MARKERS)
    if has_tools:
        signals.append("tools-requested")
    if has_code:
        signals.append("code-detected")

    if has_tools or tokens >= COMPLEX_TOKENS:
        tier = "complex"
    elif has_code or tokens >= MODERATE_TOKENS:
        tier = "moderate"
    else:
        tier = "simple"
    return tier, signals


def route(
    config: Config,
    policy_name: str,
    messages: list[dict],
    has_tools: bool,
    max_cost_usd: float | None,
    max_tokens: int | None,
) -> RouteDecision:
    if policy_name not in config.policies:
        raise NoRouteError(
            f"unknown policy {policy_name!r}; available: {sorted(config.policies)}"
        )

    tier, signals = classify(messages, has_tools)
    prompt_tokens = estimate_prompt_tokens(messages)
    completion_estimate = max_tokens or DEFAULT_COMPLETION_ESTIMATE

    kept: list[str] = []
    est_costs: dict[str, float] = {}
    dropped: list[str] = []
    for name in config.policies[policy_name][tier]:
        est = cost_usd(config.backends[name], prompt_tokens, completion_estimate)
        if max_cost_usd is not None and est > max_cost_usd:
            dropped.append(f"{name}(est=${est:.6f}>cap=${max_cost_usd:.6f})")
            continue
        kept.append(name)
        est_costs[name] = est

    if not kept:
        raise NoRouteError(
            f"no backend in policy {policy_name!r} tier {tier!r} fits "
            f"max-cost cap ${max_cost_usd:.6f}: dropped {', '.join(dropped)}"
        )

    reason = (
        f"policy={policy_name} tier={tier} signals=[{','.join(signals)}] "
        f"chain={'>'.join(kept)}"
    )
    if dropped:
        reason += f" dropped=[{','.join(dropped)}]"

    return RouteDecision(
        policy=policy_name, tier=tier, chain=kept, reason=reason, est_costs=est_costs
    )
