from rai.tools.security.guardrails import validate_command

def test_validate_command_empty():
    is_safe, error = validate_command("")
    assert not is_safe
    assert "Command is empty" in error

    is_safe, error = validate_command("   ")
    assert not is_safe
    assert "Command is empty" in error

def test_validate_command_safe():
    safe_commands = [
        "ls -la",
        "echo 'hello world'",
        "cat README.md",
        "python3 -c 'print(1+1)'",
        "git status",
    ]
    for cmd in safe_commands:
        is_safe, error = validate_command(cmd)
        assert is_safe, f"Command '{cmd}' should be safe, but failed: {error}"

def test_validate_command_blocked_keywords():
    blocked_commands = [
        "reboot",
        "shutdown -h now",
        "poweroff",
        "sudo init 0",
        "init 6",
        "passwd root",
    ]
    for cmd in blocked_commands:
        is_safe, error = validate_command(cmd)
        assert not is_safe, f"Command '{cmd}' should be blocked by keyword"
        assert "blocked destructive keyword" in error.lower()

def test_validate_command_blocked_patterns():
    dangerous_patterns = [
        ":(){ :|:& };:",  # Fork bomb
        "rm -rf /",
        "rm -rf /etc",
        "dd if=/dev/zero of=/dev/sda",
        "mkfs.ext4 /dev/sdb1",
        "fdisk -l",
        "mount /dev/sdc /mnt",
        "chmod 777 /",
        "iptables -F",
        "ufw disable",
    ]
    for cmd in dangerous_patterns:
        is_safe, error = validate_command(cmd)
        assert not is_safe, f"Command '{cmd}' should be blocked by regex pattern"
        assert "blocked dangerous pattern" in error.lower()
