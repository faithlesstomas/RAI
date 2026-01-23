from unittest.mock import MagicMock, patch
from ollama import ResponseError
from rai.cli import check_model_tool_support

def test_check_model_tool_support_ministral():
    """Test detection with ministral-style template."""
    mock_details = {
        "modelfile": 'TEMPLATE """{{- if .Tools }}...{{ end }}"""'
    }
    with patch("ollama.show", return_value=mock_details):
        # This currently fails or should fail if we haven't fixed it
        # depending on strictness, but we want to assert True eventually.
        # For reproduction, we expect this might return False typically
        # if the pattern isn't matched.
        assert check_model_tool_support("ministral-3") == True

    # Let's test specifically the failing case
    mock_ministral_details = {
        "modelfile": 'TEMPLATE """{{- if $.Tools }}...{{ end }}"""'
    }
    with patch("ollama.show", return_value=mock_ministral_details):
        assert check_model_tool_support("ministral-3") == True
        
def test_check_model_tool_support_various_patterns():
    """Test various tool support patterns."""
    patterns = [
        ('TEMPLATE """{{ .Tools }}"""', True),
        ('TEMPLATE """{{.Tools}}"""', True),
        ('PARAMETER tool_use true', True),
        ('TEMPLATE """{{ $.Tools }}"""', True), # The missing one
        ('TEMPLATE """{{- if .Tools }}"""', True),
        ('TEMPLATE """{{- if $.Tools }}"""', True),
        ('No tools here', False),
    ]

    for modelfile_content, expected in patterns:
        with patch("ollama.show", return_value={"modelfile": modelfile_content}):
            assert check_model_tool_support("test-model") == expected, f"Failed for pattern: {modelfile_content}"

def test_check_model_tool_support_error():
    """Test error handling."""
    with patch("ollama.show", side_effect=ResponseError("Not found")):
        assert check_model_tool_support("invalid-model") == False
