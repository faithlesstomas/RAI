import os
import pytest
from unittest.mock import MagicMock, patch
import gitlab

from rai.tools.gitlab import GitlabTools

@pytest.fixture(autouse=True)
def mock_env_vars():
    with patch.dict(os.environ, {
        "GITLAB_ACCESS_TOKEN": "test_token",
        "GITLAB_BASE_URL": "https://gitlab.com",
    }):
        yield

@pytest.fixture
def mock_gitlab_instance():
    with patch('gitlab.Gitlab') as mock_gitlab_class:
        mock_gl = MagicMock()
        mock_gitlab_class.return_value = mock_gl
        yield mock_gl

@pytest.fixture
def gitlab_tools(mock_gitlab_instance):
    return GitlabTools()

def test_authentication_success(gitlab_tools, mock_gitlab_instance):
    mock_gitlab_instance.auth.assert_called_once()

def test_authentication_failure_no_token():
    with patch.dict(os.environ, {"GITLAB_ACCESS_TOKEN": ""}):
        with pytest.raises(ValueError, match="GITLAB_ACCESS_TOKEN environment variable is not set"):
            GitlabTools()

def test_list_projects(gitlab_tools, mock_gitlab_instance):
    mock_project = MagicMock()
    mock_project.id = 1
    mock_project.name = "Test Project"
    mock_project.path_with_namespace = "test-group/test-project"
    mock_project.web_url = "https://gitlab.com/test-group/test-project"
    mock_project.description = "A test project"
    mock_gitlab_instance.projects.list.return_value = [mock_project]

    projects = gitlab_tools.list_projects(search="Test")
    mock_gitlab_instance.projects.list.assert_called_with(search="Test", owned=False, iterator=True)
    assert len(projects) == 1
    assert projects[0]["name"] == "Test Project"

def test_get_project(gitlab_tools, mock_gitlab_instance):
    mock_project = MagicMock()
    mock_project.id = 1
    mock_project.name = "Test Project"
    mock_project.path_with_namespace = "test-group/test-project"
    mock_project.web_url = "https://gitlab.com/test-group/test-project"
    mock_project.description = "A test project"
    mock_project.default_branch = "main"
    mock_project.created_at = "2023-01-01T00:00:00Z"
    mock_project.last_activity_at = "2023-01-01T00:00:00Z"
    mock_gitlab_instance.projects.get.return_value = mock_project

    project = gitlab_tools.get_project("test-group/test-project")
    mock_gitlab_instance.projects.get.assert_called_with("test-group/test-project")
    assert project["name"] == "Test Project"

def test_list_merge_requests(gitlab_tools, mock_gitlab_instance):
    mock_project = MagicMock()
    mock_mr = MagicMock()
    mock_mr.id = 101
    mock_mr.iid = 1
    mock_mr.title = "Test MR"
    mock_mr.state = "opened"
    mock_mr.web_url = "https://gitlab.com/test-group/test-project/-/merge_requests/1"
    mock_mr.author = {"username": "test_user"}
    mock_mr.created_at = "2023-01-01T00:00:00Z"
    mock_mr.updated_at = "2023-01-01T00:00:00Z"
    mock_project.mergerequests.list.return_value = [mock_mr]
    mock_gitlab_instance.projects.get.return_value = mock_project

    mrs = gitlab_tools.list_merge_requests("test-group/test-project")
    mock_gitlab_instance.projects.get.assert_called_with("test-group/test-project")
    mock_project.mergerequests.list.assert_called_with(state="opened", all=True)
    assert len(mrs) == 1
    assert mrs[0]["title"] == "Test MR"

def test_get_merge_request(gitlab_tools, mock_gitlab_instance):
    mock_project = MagicMock()
    mock_mr = MagicMock()
    mock_mr.id = 101
    mock_mr.iid = 1
    mock_mr.title = "Test MR"
    mock_mr.description = "Description of test MR"
    mock_mr.state = "opened"
    mock_mr.web_url = "https://gitlab.com/test-group/test-project/-/merge_requests/1"
    mock_mr.author = {"username": "test_user"}
    mock_mr.created_at = "2023-01-01T00:00:00Z"
    mock_mr.updated_at = "2023-01-01T00:00:00Z"
    mock_mr.source_branch = "feature"
    mock_mr.target_branch = "main"
    mock_project.mergerequests.get.return_value = mock_mr
    mock_gitlab_instance.projects.get.return_value = mock_project

    mr = gitlab_tools.get_merge_request("test-group/test-project", 1)
    mock_gitlab_instance.projects.get.assert_called_with("test-group/test-project")
    mock_project.mergerequests.get.assert_called_with(1)
    assert mr["title"] == "Test MR"

def test_list_issues(gitlab_tools, mock_gitlab_instance):
    mock_project = MagicMock()
    mock_issue = MagicMock()
    mock_issue.id = 201
    mock_issue.iid = 10
    mock_issue.title = "Test Issue"
    mock_issue.state = "opened"
    mock_issue.web_url = "https://gitlab.com/test-group/test-project/-/issues/10"
    mock_issue.author = {"username": "test_user"}
    mock_issue.created_at = "2023-01-01T00:00:00Z"
    mock_issue.updated_at = "2023-01-01T00:00:00Z"
    mock_project.issues.list.return_value = [mock_issue]
    mock_gitlab_instance.projects.get.return_value = mock_project

    issues = gitlab_tools.list_issues("test-group/test-project")
    mock_gitlab_instance.projects.get.assert_called_with("test-group/test-project")
    mock_project.issues.list.assert_called_with(state="opened", all=True)
    assert len(issues) == 1
    assert issues[0]["title"] == "Test Issue"

def test_get_issue(gitlab_tools, mock_gitlab_instance):
    mock_project = MagicMock()
    mock_issue = MagicMock()
    mock_issue.id = 201
    mock_issue.iid = 10
    mock_issue.title = "Test Issue"
    mock_issue.description = "Description of test issue"
    mock_issue.state = "opened"
    mock_issue.web_url = "https://gitlab.com/test-group/test-project/-/issues/10"
    mock_issue.author = {"username": "test_user"}
    mock_issue.created_at = "2023-01-01T00:00:00Z"
    mock_issue.updated_at = "2023-01-01T00:00:00Z"
    mock_project.issues.get.return_value = mock_issue
    mock_gitlab_instance.projects.get.return_value = mock_project

    issue = gitlab_tools.get_issue("test-group/test-project", 10)
    mock_gitlab_instance.projects.get.assert_called_with("test-group/test-project")
    mock_project.issues.get.assert_called_with(10)
    assert issue["title"] == "Test Issue"

def test_get_file_content(gitlab_tools, mock_gitlab_instance):
    mock_project = MagicMock()
    mock_file = MagicMock()
    mock_file.decode.return_value = b"file content"
    mock_project.files.get.return_value = mock_file
    mock_gitlab_instance.projects.get.return_value = mock_project

    content = gitlab_tools.get_file_content("test-group/test-project", "README.md")
    mock_gitlab_instance.projects.get.assert_called_with("test-group/test-project")
    mock_project.files.get.assert_called_with(file_path="README.md", ref="main")
    assert content == "file content"

def test_create_issue(gitlab_tools, mock_gitlab_instance):
    mock_project = MagicMock()
    mock_issue = MagicMock()
    mock_issue.id = 301
    mock_issue.iid = 20
    mock_issue.title = "New Issue"
    mock_issue.description = "New issue description"
    mock_issue.web_url = "https://gitlab.com/test-group/test-project/-/issues/20"
    mock_issue.author = {"username": "test_user"}
    mock_project.issues.create.return_value = mock_issue
    mock_gitlab_instance.projects.get.return_value = mock_project

    issue = gitlab_tools.create_issue("test-group/test-project", "New Issue", "New issue description")
    mock_gitlab_instance.projects.get.assert_called_with("test-group/test-project")
    mock_project.issues.create.assert_called_with({'title': "New Issue", 'description': "New issue description"})
    assert issue["title"] == "New Issue"

def test_create_merge_request(gitlab_tools, mock_gitlab_instance):
    mock_project = MagicMock()
    mock_mr = MagicMock()
    mock_mr.id = 401
    mock_mr.iid = 30
    mock_mr.title = "New MR"
    mock_mr.description = "New MR description"
    mock_mr.web_url = "https://gitlab.com/test-group/test-project/-/merge_requests/30"
    mock_mr.author = {"username": "test_user"}
    mock_project.mergerequests.create.return_value = mock_mr
    mock_gitlab_instance.projects.get.return_value = mock_project

    mr = gitlab_tools.create_merge_request(
        "test-group/test-project", "feature-branch", "main", "New MR", "New MR description"
    )
    mock_gitlab_instance.projects.get.assert_called_with("test-group/test-project")
    mock_project.mergerequests.create.assert_called_with({
        'source_branch': "feature-branch",
        'target_branch': "main",
        'title': "New MR",
        'description': "New MR description",
    })
    assert mr["title"] == "New MR"

# Test error handling for all methods
def test_list_projects_error(gitlab_tools, mock_gitlab_instance):
    mock_gitlab_instance.projects.list.side_effect = gitlab.exceptions.GitlabError("API Error")
    projects = gitlab_tools.list_projects()
    assert projects == []

def test_get_project_error(gitlab_tools, mock_gitlab_instance):
    mock_gitlab_instance.projects.get.side_effect = gitlab.exceptions.GitlabError("API Error")
    project = gitlab_tools.get_project("non-existent/project")
    assert project is None

def test_list_merge_requests_error(gitlab_tools, mock_gitlab_instance):
    mock_project = MagicMock()
    mock_project.mergerequests.list.side_effect = gitlab.exceptions.GitlabError("API Error")
    mock_gitlab_instance.projects.get.return_value = mock_project
    mrs = gitlab_tools.list_merge_requests("test-group/test-project")
    assert mrs == []

def test_get_merge_request_error(gitlab_tools, mock_gitlab_instance):
    mock_project = MagicMock()
    mock_project.mergerequests.get.side_effect = gitlab.exceptions.GitlabError("API Error")
    mock_gitlab_instance.projects.get.return_value = mock_project
    mr = gitlab_tools.get_merge_request("test-group/test-project", 999)
    assert mr is None

def test_list_issues_error(gitlab_tools, mock_gitlab_instance):
    mock_project = MagicMock()
    mock_project.issues.list.side_effect = gitlab.exceptions.GitlabError("API Error")
    mock_gitlab_instance.projects.get.return_value = mock_project
    issues = gitlab_tools.list_issues("test-group/test-project")
    assert issues == []

def test_get_issue_error(gitlab_tools, mock_gitlab_instance):
    mock_project = MagicMock()
    mock_project.issues.get.side_effect = gitlab.exceptions.GitlabError("API Error")
    mock_gitlab_instance.projects.get.return_value = mock_project
    issue = gitlab_tools.get_issue("test-group/test-project", 999)
    assert issue is None

def test_get_file_content_error(gitlab_tools, mock_gitlab_instance):
    mock_project = MagicMock()
    mock_project.files.get.side_effect = gitlab.exceptions.GitlabError("API Error")
    mock_gitlab_instance.projects.get.return_value = mock_project
    content = gitlab_tools.get_file_content("test-group/test-project", "non-existent.md")
    assert content is None

def test_create_issue_error(gitlab_tools, mock_gitlab_instance):
    mock_project = MagicMock()
    mock_project.issues.create.side_effect = gitlab.exceptions.GitlabError("API Error")
    mock_gitlab_instance.projects.get.return_value = mock_project
    issue = gitlab_tools.create_issue("test-group/test-project", "Failing Issue")
    assert issue is None

def test_create_merge_request_error(gitlab_tools, mock_gitlab_instance):
    mock_project = MagicMock()
    mock_project.mergerequests.create.side_effect = gitlab.exceptions.GitlabError("API Error")
    mock_gitlab_instance.projects.get.return_value = mock_project
    mr = gitlab_tools.create_merge_request("test-group/test-project", "bug-fix", "main", "Failing MR")
    assert mr is None