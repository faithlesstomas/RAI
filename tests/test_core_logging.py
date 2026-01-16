
import pytest
from unittest.mock import MagicMock, patch
from rai.core import setup_tools

class TestCoreLogging:
    @patch('builtins.__import__')
    @patch('rai.core.logger')
    def test_setup_tools_logging_crash(self, mock_logger, mock_import):
        """
        Test that setup_tools handles ImportError without crashing the logger.
        Regression test for: TypeError: The fill character must be exactly one character long
        """
        # Simulate ImportError when importing a tool
        mock_import.side_effect = ImportError("Test error")
        
        # We need to ensure we are testing the path where a tool is processed
        # TOOL_REGISTRY has tools, so calling setup_tools(enable_tools=True, quiet=False)
        # should iterate and try to import.
        
        setup_tools(enable_tools=True, quiet=False)
        
        # Verify logger.debug was called with a single string argument (f-string)
        # instead of multiple args which caused the crash
        for call in mock_logger.debug.call_args_list:
            args = call.args
            # The crash happened when args was > 1 (msg, arg)
            # We expect either 1 arg, or if 2 args, the second one wasn't the exception string
            if "Could not import" in args[0]:
                assert len(args) == 1, f"Logger called with multiple args: {args}"
                assert "Test error" in args[0]
