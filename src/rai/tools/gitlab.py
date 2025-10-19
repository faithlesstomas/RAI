"""
GitLab Authentication Setup Guide

1. Getting Personal Access Token (PAT):
   a. Navigate to GitLab Settings:
      - Log into GitLab
      - Click your avatar (top-right)
      - Select "Preferences"
      - Go to "Access Tokens" under "User Settings"

   b. Generate New Token:
      - Enter a descriptive name for the token
      - Set an expiration date (optional, but recommended)
      - Select scopes (minimum 'api' access for most operations, 'read_repository'
        for read-only, 'write_repository' for write operations)
      - Click "Create personal access token"
      - IMPORTANT: Save token immediately - only shown once!

2. Setting Environment Variables:

   # For GitLab.com
   export GITLAB_ACCESS_TOKEN="your_token_here"
   export GITLAB_BASE_URL="https://gitlab.com"

   # For Self-Managed GitLab
   export GITLAB_BASE_URL="https://YOUR-GITLAB-HOSTNAME"
   export GITLAB_ACCESS_TOKEN="your_token_here"
"""

import os
from typing import Any, Dict, List, Optional

import gitlab
from agno.tools import Toolkit
from agno.utils.log import logger
from requests import exceptions as requests_exceptions


class GitlabTools(Toolkit):
    """
    A collection of tools for interacting with the GitLab API.
    Requires GITLAB_ACCESS_TOKEN and GITLAB_BASE_URL environment variables to be set.
    """

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(
            name="gitlab_tools",
            tools=[
                self.list_projects,
                self.get_project,
                self.list_merge_requests,
                self.get_merge_request,
                self.list_issues,
                self.get_issue,
                self.get_file_content,
                self.create_issue,
                self.create_merge_request,
            ],
            **kwargs,
        )
        self.gitlab_token = os.getenv("GITLAB_ACCESS_TOKEN")
        gitlab_url = os.getenv("GITLAB_BASE_URL", "https://gitlab.com")
        self.gitlab_url = gitlab_url.strip('"\'').rstrip("/")
        self.gl: Optional[gitlab.Gitlab] = None
        self._disabled = False
        self._get_gitlab_client()  # Initial attempt

    def _get_gitlab_client(self) -> Optional[gitlab.Gitlab]:
        if self._disabled:
            return None
        if self.gl:
            return self.gl

        if not self.gitlab_token:
            raise ValueError("GITLAB_ACCESS_TOKEN environment variable is not set")

        try:
            logger.debug(
                f"Attempting to initialize GitLab client with URL: {self.gitlab_url}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            self.gl = gitlab.Gitlab(self.gitlab_url, private_token=self.gitlab_token)
            self.gl.auth()
            logger.debug("GitLab client successfully initialized and authenticated.")
            return self.gl
        except (gitlab.exceptions.GitlabError, requests_exceptions.ConnectionError) as e:
            logger.warning(
                f"Disabling GitlabTools. Failed to connect or authenticate with GitLab: {e}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            self._disabled = True
            self.gl = None
            return None

    def list_projects(
        self,
        search: Optional[str] = None,
        owned: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Lists GitLab projects.

        Args:
            search: A string to search for in project names and paths.
            owned: If True, only list projects owned by the authenticated user.

        Returns:
            A list of dictionaries, each representing a project.
        """
        gl_client = self._get_gitlab_client()
        if not gl_client:
            return []
        logger.debug(
            f"Listing projects with search='{search}' and owned={owned}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
        )
        try:
            projects_iterator = gl_client.projects.list(
                search=search, owned=owned, iterator=True
            )
            projects = []
            for p in projects_iterator:
                projects.append(
                    {
                        "id": p.id,
                        "name": p.name,
                        "path_with_namespace": p.path_with_namespace,
                        "web_url": p.web_url,
                        "description": p.description,
                    }
                )
            logger.debug(
                f"Found {len(projects)} projects."  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            return projects
        except gitlab.exceptions.GitlabError as e:
            logger.error(
                f"Failed to list projects: {e}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            return []

    def get_project(self, project_id_or_path: str) -> Optional[Dict[str, Any]]:
        """
        Gets details of a specific GitLab project.

        Args:
            project_id_or_path: The ID or path with namespace of the project
            (e.g., "my-group/my-project").

        Returns:
            A dictionary representing the project, or None if not found.
        """
        gl_client = self._get_gitlab_client()
        if not gl_client:
            return None
        logger.debug(
            f"Getting project with project_id_or_path='{project_id_or_path}'"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
        )
        try:
            project = gl_client.projects.get(project_id_or_path)
            project_details = {
                "id": project.id,
                "name": project.name,
                "path_with_namespace": project.path_with_namespace,
                "web_url": project.web_url,
                "description": project.description,
                "default_branch": project.default_branch,
                "created_at": project.created_at,
                "last_activity_at": project.last_activity_at,
            }
            logger.debug(
                f"Found project: {project_details}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            return project_details
        except gitlab.exceptions.GitlabError as e:
            logger.error(
                f"Failed to get project {project_id_or_path}: {e}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            return None

    def list_merge_requests(
        self,
        project_id_or_path: str,
        state: str = "opened",
    ) -> List[Dict[str, Any]]:
        """
        Lists merge requests for a given project.

        Args:
            project_id_or_path: The ID or path with namespace of the project.
            state: The state of the merge requests to list (e.g., 'opened', 'closed',
            'merged', 'all').

        Returns:
            A list of dictionaries, each representing a merge request.
        """
        gl_client = self._get_gitlab_client()
        if not gl_client:
            return []
        logger.debug(
            f"Listing merge requests for project '{project_id_or_path}' with state='{state}'"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
        )
        try:
            project = gl_client.projects.get(project_id_or_path)
            mrs = project.mergerequests.list(state=state, all=True)
            result = [
                {
                    "id": mr.id,
                    "iid": mr.iid,
                    "title": mr.title,
                    "state": mr.state,
                    "web_url": mr.web_url,
                    "author": mr.author["username"],
                    "created_at": mr.created_at,
                    "updated_at": mr.updated_at,
                }
                for mr in mrs
            ]
            logger.debug(
                f"Found {len(result)} merge requests."  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            return result
        except gitlab.exceptions.GitlabError as e:
            logger.error(
                f"Failed to list merge requests for {project_id_or_path}: {e}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            return []

    def get_merge_request(
        self,
        project_id_or_path: str,
        mr_iid: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Gets details of a specific merge request for a given project.

        Args:
            project_id_or_path: The ID or path with namespace of the project.
            mr_iid: The internal ID (IID) of the merge request.

        Returns:
            A dictionary representing the merge request, or None if not found.
        """
        gl_client = self._get_gitlab_client()
        if not gl_client:
            return None
        logger.debug(
            f"Getting merge request iid={mr_iid} for project '{project_id_or_path}'"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
        )
        try:
            project = gl_client.projects.get(project_id_or_path)
            mr = project.mergerequests.get(mr_iid)
            mr_details = {
                "id": mr.id,
                "iid": mr.iid,
                "title": mr.title,
                "description": mr.description,
                "state": mr.state,
                "web_url": mr.web_url,
                "author": mr.author["username"],
                "created_at": mr.created_at,
                "updated_at": mr.updated_at,
                "source_branch": mr.source_branch,
                "target_branch": mr.target_branch,
            }
            logger.debug(
                f"Found merge request: {mr_details}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            return mr_details
        except gitlab.exceptions.GitlabError as e:
            logger.error(
                f"Failed to get merge request {mr_iid} for {project_id_or_path}: {e}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            return None

    def list_issues(
        self,
        project_id_or_path: str,
        state: str = "opened",
    ) -> List[Dict[str, Any]]:
        """
        Lists issues for a given project.

        Args:
            project_id_or_path: The ID or path with namespace of the project.
            state: The state of the issues to list (e.g., 'opened', 'closed', 'all').

        Returns:
            A list of dictionaries, each representing an issue.
        """
        gl_client = self._get_gitlab_client()
        if not gl_client:
            return []
        logger.debug(
            f"Listing issues for project '{project_id_or_path}' with state='{state}'"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
        )
        try:
            project = gl_client.projects.get(project_id_or_path)
            issues = project.issues.list(state=state, all=True)
            result = [
                {
                    "id": issue.id,
                    "iid": issue.iid,
                    "title": issue.title,
                    "state": issue.state,
                    "web_url": issue.web_url,
                    "author": issue.author["username"],
                    "created_at": issue.created_at,
                    "updated_at": issue.updated_at,
                }
                for issue in issues
            ]
            logger.debug(
                f"Found {len(result)} issues."  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            return result
        except gitlab.exceptions.GitlabError as e:
            logger.error(
                f"Failed to list issues for {project_id_or_path}: {e}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            return []

    def get_issue(
        self,
        project_id_or_path: str,
        issue_iid: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Gets details of a specific issue for a given project.

        Args:
            project_id_or_path: The ID or path with namespace of the project.
            issue_iid: The internal ID (IID) of the issue.

        Returns:
            A dictionary representing the issue, or None if not found.
        """
        gl_client = self._get_gitlab_client()
        if not gl_client:
            return None
        logger.debug(
            f"Getting issue iid={issue_iid} for project '{project_id_or_path}'"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
        )
        try:
            project = gl_client.projects.get(project_id_or_path)
            issue = project.issues.get(issue_iid)
            issue_details = {
                "id": issue.id,
                "iid": issue.iid,
                "title": issue.title,
                "description": issue.description,
                "state": issue.state,
                "web_url": issue.web_url,
                "author": issue.author["username"],
                "created_at": issue.created_at,
                "updated_at": issue.updated_at,
            }
            logger.debug(
                f"Found issue: {issue_details}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            return issue_details
        except gitlab.exceptions.GitlabError as e:
            logger.error(
                f"Failed to get issue {issue_iid} for {project_id_or_path}: {e}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            return None

    def get_file_content(
        self,
        project_id_or_path: str,
        file_path: str,
        ref: str = "main",
    ) -> Optional[str]:
        """
        Gets the content of a file from a GitLab project.

        Args:
            project_id_or_path: The ID or path with namespace of the project.
            file_path: The path to the file within the repository.
            ref: The branch, tag, or commit SHA to retrieve the file from.
            Defaults to 'main'.

        Returns:
            The content of the file as a string, or None if not found.
        """
        gl_client = self._get_gitlab_client()
        if not gl_client:
            return None
        logger.debug(
            f"Getting file content for '{file_path}' in project '{project_id_or_path}' at ref '{ref}'"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
        )
        try:
            project = gl_client.projects.get(project_id_or_path)
            file = project.files.get(file_path=file_path, ref=ref)
            content = file.decode().decode("utf-8")
            logger.debug(
                f"Got file content of length {len(content)}."  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            return content
        except gitlab.exceptions.GitlabError as e:
            logger.error(
                f"Failed to get file content for {file_path} in {project_id_or_path} (ref: {ref}): {e}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            return None

    def create_issue(
        self,
        project_id_or_path: str,
        title: str,
        description: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Creates a new issue in a GitLab project.

        Args:
            project_id_or_path: The ID or path with namespace of the project.
            title: The title of the new issue.
            description: The description of the new issue.

        Returns:
            A dictionary representing the created issue, or None if creation failed.
        """
        gl_client = self._get_gitlab_client()
        if not gl_client:
            return None
        logger.debug(
            f"Creating issue with title '{title}' in project '{project_id_or_path}'"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
        )
        try:
            project = gl_client.projects.get(project_id_or_path)
            issue = project.issues.create({"title": title, "description": description})
            issue_details = {
                "id": issue.id,
                "iid": issue.iid,
                "title": issue.title,
                "description": issue.description,
                "web_url": issue.web_url,
                "author": issue.author["username"],
            }
            logger.debug(
                f"Created issue: {issue_details}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            return issue_details
        except gitlab.exceptions.GitlabError as e:
            logger.error(
                f"Failed to create issue in {project_id_or_path}: {e}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            return None

    def create_merge_request(
        self,
        project_id_or_path: str,
        source_branch: str,
        target_branch: str,
        title: str,
        description: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Creates a new merge request in a GitLab project.

        Args:
            project_id_or_path: The ID or path with namespace of the project.
            source_branch: The source branch of the merge request.
            target_branch: The target branch of the merge request.
            title: The title of the new merge request.
            description: The description of the new merge request.

        Returns:
            A dictionary representing the created merge request, or None if creation
            failed.
        """
        gl_client = self._get_gitlab_client()
        if not gl_client:
            return None
        logger.debug(
            f"Creating merge request with title '{title}' in project '{project_id_or_path}'"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
        )
        try:
            project = gl_client.projects.get(project_id_or_path)
            mr = project.mergerequests.create(
                {
                    "source_branch": source_branch,
                    "target_branch": target_branch,
                    "title": title,
                    "description": description,
                }
            )
            mr_details = {
                "id": mr.id,
                "iid": mr.iid,
                "title": mr.title,
                "description": mr.description,
                "web_url": mr.web_url,
                "author": mr.author["username"],
            }
            logger.debug(
                f"Created merge request: {mr_details}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            return mr_details
        except gitlab.exceptions.GitlabError as e:
            logger.error(
                f"Failed to create merge request in {project_id_or_path}: {e}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )
            return None