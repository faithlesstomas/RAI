"""User-managed profile consent rules, reloaded for every approval."""
from pathlib import Path
import fcntl
import os

from rai.kernel.records import CapabilityRequest
from .models import ApprovalRule
from .storage import ConfigurationError, load_settings, save_settings, selected_path


def matching_rule(profile: str, request: CapabilityRequest, path: Path) -> ApprovalRule | None:
    settings = load_settings(str(path))
    configured = settings.agents.get(profile)
    if configured is None:
        return None
    matches = [rule for rule in configured.approval_rules
               if rule.capability == request.capability and rule.data_class == request.data_class
               and (rule.query is None or rule.query == request.arguments.get("query"))]
    # Denials cannot be bypassed by a more specific allow; ask also beats allow.
    for outcome in ("deny", "ask", "allow"):
        candidates = [rule for rule in matches if rule.decision == outcome]
        if candidates:
            return sorted(candidates, key=lambda rule: (rule.query is None, rule.id))[0]
    return None


def update_rule(profile: str, rule_id: str, rule: ApprovalRule | None, path: Path | None = None) -> None:
    target = path or selected_path()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(str(target) + ".approvals.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = load_settings(str(target)).runtime_mapping()
        if profile not in data.get("agents", {}):
            raise ConfigurationError(f"Unknown profile: {profile}")
        configured = data["agents"][profile]
        rules = configured.get("approval_rules", [])
        if rule is None and not any(item["id"] == rule_id for item in rules):
            raise ConfigurationError(f"Unknown approval rule: {rule_id}")
        configured["approval_rules"] = [item for item in rules if item["id"] != rule_id]
        if rule is not None:
            configured["approval_rules"].append(rule.model_dump())
        save_settings(data, str(target))
