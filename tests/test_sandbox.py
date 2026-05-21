import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from rai.tools.security.sandbox import BubblewrapRunner, GuixContainerRunner, get_sandbox_runner

def test_get_sandbox_runner():
    runner = get_sandbox_runner()
    assert runner is not None

def test_bubblewrap_availability():
    runner = BubblewrapRunner()
    # Check that is_available returns a boolean
    available = runner.is_available()
    assert isinstance(available, bool)

@patch("rai.tools.security.sandbox.subprocess.run")
@patch("rai.tools.security.sandbox.shutil.which")
def test_bubblewrap_runner_arguments_construction(mock_which, mock_run):
    # Force bubblewrap to be available for this test
    mock_which.return_value = "/usr/bin/bwrap"
    
    # Mock subprocess run to return successfully
    mock_res = MagicMock()
    mock_res.returncode = 0
    mock_res.stdout = "hello"
    mock_res.stderr = ""
    mock_run.return_value = mock_res

    # 1. Run without network and without rw_dir
    runner = BubblewrapRunner(workspace_dir="/tmp/mock-workspace")
    result = runner.run(["echo", "hello"], allow_network=False)
    
    assert result.is_success()
    assert result.stdout == "hello"
    
    # Check constructed args
    args, _ = mock_run.call_args
    bwrap_args = args[0]
    
    assert "/usr/bin/bwrap" in bwrap_args
    assert "--unshare-all" in bwrap_args
    assert "--share-net" not in bwrap_args
    assert "--tmpfs" in bwrap_args
    # It must bind the workspace read-only
    assert "--ro-bind" in bwrap_args
    assert "/tmp/mock-workspace" in bwrap_args
    assert "/workspace" in bwrap_args

    # 2. Run with network and rw_dir
    runner = BubblewrapRunner(workspace_dir="/tmp/mock-workspace")
    result = runner.run(["echo", "hello"], allow_network=True, rw_dir="/tmp/mock-rw")
    
    args, _ = mock_run.call_args
    bwrap_args = args[0]
    
    assert "--share-net" in bwrap_args
    assert "--bind" in bwrap_args
    assert "/tmp/mock-rw" in bwrap_args
    assert "/output" in bwrap_args

@patch("rai.tools.security.sandbox.subprocess.run")
@patch("rai.tools.security.sandbox.shutil.which")
def test_guix_container_runner_arguments_construction(mock_which, mock_run):
    mock_which.return_value = "/usr/bin/guix"
    
    mock_res = MagicMock()
    mock_res.returncode = 0
    mock_res.stdout = "guix-success"
    mock_res.stderr = ""
    mock_run.return_value = mock_res

    runner = GuixContainerRunner()
    result = runner.run(["echo", "hello"], allow_network=True, rw_dir="/tmp/mock-rw")
    
    assert result.is_success()
    
    args, _ = mock_run.call_args
    guix_args = args[0]
    
    assert "/usr/bin/guix" in guix_args
    assert "shell" in guix_args
    assert "--container" in guix_args
    assert "--network" in guix_args
    assert "--expose=/tmp/mock-rw" in guix_args

def test_bubblewrap_runner_actual_execution_if_available():
    runner = BubblewrapRunner()
    if not runner.is_available():
        pytest.skip("Bubblewrap is not installed on this system.")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run a simple echo command in sandbox
        result = runner.run(["echo", "hello from sandbox"], rw_dir=tmpdir)
        assert result.is_success()
        assert "hello from sandbox" in result.stdout

        # Verify we can write to /output/test.txt which resolves to tmpdir/test.txt
        result_write = runner.run([
            "/bin/bash", "-c", "echo 'secret' > /output/test.txt"
        ], rw_dir=tmpdir)
        
        assert result_write.is_success()
        
        # Verify file exists on host
        host_file = os.path.join(tmpdir, "test.txt")
        assert os.path.exists(host_file)
        with open(host_file, "r") as f:
            assert f.read().strip() == "secret"

        # Verify we CANNOT write directly to the workspace (which is mounted read-only)
        result_write_fail = runner.run([
            "/bin/bash", "-c", "touch /workspace/failed.txt"
        ], rw_dir=tmpdir)
        
        assert not result_write_fail.is_success()
        assert "Read-only file system" in result_write_fail.stderr
