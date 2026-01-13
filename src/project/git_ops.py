"""Git operations for experiment tracking and versioning."""

import asyncio
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional


class GitOps:
    """Git operations for research project versioning."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
    
    def _run_git(self, *args, check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command in the repo directory."""
        return subprocess.run(
            ["git"] + list(args),
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=check,
        )
    
    def is_initialized(self) -> bool:
        """Check if the repo is already initialized."""
        return (self.repo_path / ".git").exists()
    
    def init(self) -> bool:
        """Initialize a git repository."""
        if self.is_initialized():
            return True
        
        try:
            self._run_git("init")
            self._run_git("config", "user.email", "research-mcp@local")
            self._run_git("config", "user.name", "Research MCP")
            return True
        except subprocess.CalledProcessError:
            return False
    
    def add(self, path: str = ".") -> bool:
        """Stage files for commit."""
        try:
            self._run_git("add", path)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def commit(self, message: str) -> Optional[str]:
        """Create a commit with the given message."""
        try:
            # Check if there are changes to commit
            status = self._run_git("status", "--porcelain", check=False)
            if not status.stdout.strip():
                return None  # Nothing to commit
            
            self._run_git("add", ".")
            self._run_git("commit", "-m", message)
            
            # Get the commit hash
            result = self._run_git("rev-parse", "HEAD")
            return result.stdout.strip()[:8]
        except subprocess.CalledProcessError:
            return None
    
    def tag(self, tag_name: str, message: Optional[str] = None) -> bool:
        """Create a tag."""
        try:
            if message:
                self._run_git("tag", "-a", tag_name, "-m", message)
            else:
                self._run_git("tag", tag_name)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def get_tags(self) -> list[str]:
        """Get all tags."""
        try:
            result = self._run_git("tag", "-l")
            return [t.strip() for t in result.stdout.split("\n") if t.strip()]
        except subprocess.CalledProcessError:
            return []
    
    def get_log(self, max_entries: int = 10) -> list[dict]:
        """Get recent commit log."""
        try:
            result = self._run_git(
                "log",
                f"-{max_entries}",
                "--pretty=format:%H|%s|%ai",
                check=False,
            )
            
            commits = []
            for line in result.stdout.split("\n"):
                if line.strip():
                    parts = line.split("|", 2)
                    if len(parts) == 3:
                        commits.append({
                            "hash": parts[0][:8],
                            "message": parts[1],
                            "date": parts[2],
                        })
            return commits
        except subprocess.CalledProcessError:
            return []
    
    def get_status(self) -> dict:
        """Get repository status."""
        try:
            status_result = self._run_git("status", "--porcelain", check=False)
            
            modified = []
            added = []
            deleted = []
            untracked = []
            
            for line in status_result.stdout.split("\n"):
                if not line.strip():
                    continue
                status_code = line[:2]
                file_path = line[3:]
                
                if "M" in status_code:
                    modified.append(file_path)
                elif "A" in status_code:
                    added.append(file_path)
                elif "D" in status_code:
                    deleted.append(file_path)
                elif "?" in status_code:
                    untracked.append(file_path)
            
            return {
                "initialized": self.is_initialized(),
                "modified": modified,
                "added": added,
                "deleted": deleted,
                "untracked": untracked,
                "clean": not (modified or added or deleted or untracked),
            }
        except subprocess.CalledProcessError:
            return {"initialized": False, "error": "Git not available"}
    
    def create_branch(self, branch_name: str) -> bool:
        """Create a new branch."""
        try:
            self._run_git("checkout", "-b", branch_name)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def checkout(self, ref: str) -> bool:
        """Checkout a branch or commit."""
        try:
            self._run_git("checkout", ref)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def get_current_branch(self) -> Optional[str]:
        """Get the current branch name."""
        try:
            result = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    def diff(self, path: Optional[str] = None) -> str:
        """Get diff of changes."""
        try:
            if path:
                result = self._run_git("diff", path, check=False)
            else:
                result = self._run_git("diff", check=False)
            return result.stdout
        except subprocess.CalledProcessError:
            return ""
    
    # Async methods for use with MCP tools
    
    async def commit(self, message: str) -> Optional[str]:
        """Async commit with the given message."""
        return await asyncio.to_thread(self._sync_commit, message)
    
    def _sync_commit(self, message: str) -> Optional[str]:
        """Sync version of commit."""
        try:
            status = self._run_git("status", "--porcelain", check=False)
            if not status.stdout.strip():
                return None
            
            self._run_git("add", ".")
            self._run_git("commit", "-m", message)
            
            result = self._run_git("rev-parse", "HEAD")
            return result.stdout.strip()[:8]
        except subprocess.CalledProcessError:
            return None
    
    async def push(self, remote: str = "origin", branch: Optional[str] = None) -> bool:
        """Push commits to remote."""
        return await asyncio.to_thread(self._sync_push, remote, branch)
    
    def _sync_push(self, remote: str, branch: Optional[str]) -> bool:
        """Sync version of push."""
        try:
            if branch:
                self._run_git("push", remote, branch)
            else:
                self._run_git("push", remote)
            return True
        except subprocess.CalledProcessError:
            return False
    
    async def auto_commit_stage(self, stage: str) -> Optional[str]:
        """Commit all changes with a stage message and push."""
        commit_hash = await self.commit(f"Stage: {stage}")
        if commit_hash:
            await self.push()
        return commit_hash
    
    async def create_github_repo(
        self,
        name: str,
        private: bool = True,
        description: str = "",
    ) -> Optional[str]:
        """Create a GitHub repo using gh CLI."""
        return await asyncio.to_thread(
            self._sync_create_github_repo, name, private, description
        )
    
    def _sync_create_github_repo(
        self,
        name: str,
        private: bool,
        description: str,
    ) -> Optional[str]:
        """Sync version of create_github_repo."""
        try:
            if not self.is_initialized():
                self.init()
            
            self._sync_commit("Initial commit")
            
            cmd = ["gh", "repo", "create", name]
            if private:
                cmd.append("--private")
            else:
                cmd.append("--public")
            
            if description:
                cmd.extend(["--description", description])
            
            cmd.extend(["--source", ".", "--push"])
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            
            if result.returncode == 0:
                return self.get_repo_url()
            
            return None
        except Exception:
            return None
    
    def get_repo_url(self) -> Optional[str]:
        """Get the GitHub repo URL."""
        try:
            result = self._run_git("remote", "get-url", "origin", check=False)
            url = result.stdout.strip()
            
            if url.startswith("git@github.com:"):
                url = url.replace("git@github.com:", "https://github.com/")
                url = url.replace(".git", "")
            elif url.endswith(".git"):
                url = url[:-4]
            
            return url if url else None
        except subprocess.CalledProcessError:
            return None
    
    async def set_remote(self, url: str, remote_name: str = "origin") -> bool:
        """Set or update remote URL."""
        return await asyncio.to_thread(self._sync_set_remote, url, remote_name)
    
    def _sync_set_remote(self, url: str, remote_name: str) -> bool:
        """Sync version of set_remote."""
        try:
            try:
                self._run_git("remote", "add", remote_name, url)
            except subprocess.CalledProcessError:
                self._run_git("remote", "set-url", remote_name, url)
            return True
        except subprocess.CalledProcessError:
            return False


def init_project_repo(project_path: Path) -> GitOps:
    """Initialize a git repo for a project and return GitOps instance."""
    git = GitOps(project_path)
    git.init()
    
    # Initial commit
    git.commit("Initial project setup")
    git.tag("project_created", "Project initialized by research-mcp")
    
    return git


def commit_experiment_start(git: GitOps, experiment_name: str, config: dict) -> Optional[str]:
    """Commit when starting an experiment."""
    message = f"Start experiment: {experiment_name}"
    commit_hash = git.commit(message)
    if commit_hash:
        git.tag(f"exp_{experiment_name}_start", f"Started {experiment_name}")
    return commit_hash


def commit_experiment_complete(git: GitOps, experiment_name: str, results_summary: str) -> Optional[str]:
    """Commit when an experiment completes."""
    message = f"Complete experiment: {experiment_name}\n\n{results_summary}"
    commit_hash = git.commit(message)
    if commit_hash:
        git.tag(f"exp_{experiment_name}_complete", f"Completed {experiment_name}")
    return commit_hash


def commit_idea_approved(git: GitOps, idea_title: str) -> Optional[str]:
    """Commit when an idea is approved."""
    message = f"Idea approved: {idea_title}"
    commit_hash = git.commit(message)
    if commit_hash:
        git.tag("idea_approved", f"Approved: {idea_title}")
    return commit_hash


def commit_paper_draft(git: GitOps, version: str) -> Optional[str]:
    """Commit a paper draft."""
    message = f"Paper draft {version}"
    commit_hash = git.commit(message)
    if commit_hash:
        git.tag(f"draft_{version}", f"Paper draft {version}")
    return commit_hash
