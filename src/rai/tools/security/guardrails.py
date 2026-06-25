"""
Guardrails and static analysis validation for shell command safety.
"""
import re
from typing import Tuple

# Blacklisted dangerous regex patterns
DANGEROUS_PATTERNS = [
    # Fork bomb attempt
    r"(:\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:)",
    # Destructive rm commands trying to wipe major system directories
    r"rm\s+-[rfv]*\s+/(etc|boot|usr|var|lib|dev|sys|proc|bin|sbin|home|opt)?(\s|$)",
    # Attempting to overwrite block devices or raw disk access
    r"dd\s+.*if=.*of=/dev/(sd[a-z]|nvme[0-9]n[0-9]|hd[a-z])",
    # Modifying filesystem mounts/partitions
    r"(mkfs[^\s]*|fdisk|parted|mount|umount)\s+",
    # Modifying absolute system directories permissions
    r"chmod\s+.*777\s+/",
    # Host firewall shutdown or altering networking tables
    r"(iptables\s+-(F|X)|ufw\s+disable)",
]

# Blacklisted dangerous command prefixes
DANGEROUS_KEYWORDS = [
    "reboot",
    "shutdown",
    "poweroff",
    "init 0",
    "init 6",
    "passwd root",
]

def validate_command(command_str: str) -> Tuple[bool, str]:
    """
    Validates a command string against guardrails.
    Returns (is_safe, error_message).
    """
    clean_command = command_str.strip()

    # 1. Empty command check
    if not clean_command:
        return False, "Command is empty."

    # 2. Check for exact dangerous keyword matches
    for keyword in DANGEROUS_KEYWORDS:
        if keyword in clean_command.lower():
            return False, f"Command contains a blocked destructive keyword: '{keyword}'."

    # 3. Pattern-based regex matching for destructive actions
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, clean_command, re.IGNORECASE):
            return False, "Command matches a blocked dangerous pattern (potential system damage)."

    return True, ""
