#!/usr/bin/env python3
"""Release script for version bumping and GitHub releases."""

import re
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def get_repo_root() -> Path:
    """Get the repository root directory."""
    result = run_cmd(["git", "rev-parse", "--show-toplevel"])
    return Path(result.stdout.strip())


def check_branch() -> bool:
    """Check if current branch is main."""
    result = run_cmd(["git", "branch", "--show-current"])
    branch = result.stdout.strip()
    if branch != "main":
        print(f"Error: Must be on 'main' branch, currently on '{branch}'")
        return False
    return True


def check_clean() -> bool:
    """Check if working directory is clean."""
    result = run_cmd(["git", "status", "--porcelain"])
    if result.stdout.strip():
        print("Error: Working directory has uncommitted changes:")
        print(result.stdout)
        return False
    return True


def check_pushed() -> bool:
    """Check if latest commit is pushed to remote."""
    # Fetch latest from remote
    run_cmd(["git", "fetch", "origin", "main"], check=False)

    # Compare local and remote
    result = run_cmd(["git", "rev-parse", "HEAD"])
    local_head = result.stdout.strip()

    result = run_cmd(["git", "rev-parse", "origin/main"], check=False)
    if result.returncode != 0:
        print("Warning: Could not find origin/main, skipping push check")
        return True

    remote_head = result.stdout.strip()

    if local_head != remote_head:
        # Check if local is ahead or behind
        result = run_cmd(["git", "log", "--oneline", f"origin/main..HEAD"])
        if result.stdout.strip():
            print("Error: Local branch has unpushed commits:")
            print(result.stdout)
            return False

        result = run_cmd(["git", "log", "--oneline", f"HEAD..origin/main"])
        if result.stdout.strip():
            print("Error: Local branch is behind origin/main:")
            print(result.stdout)
            return False

    return True


def get_current_version(pyproject_path: Path) -> str:
    """Read current version from pyproject.toml."""
    content = pyproject_path.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")
    return match.group(1)


def bump_patch(version: str) -> str:
    """Bump the patch version (0.1.2 -> 0.1.3)."""
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version}")
    parts[2] = str(int(parts[2]) + 1)
    return ".".join(parts)


def validate_version(version: str) -> bool:
    """Validate version is valid semver."""
    pattern = r"^\d+\.\d+\.\d+$"
    return bool(re.match(pattern, version))


def update_version(pyproject_path: Path, new_version: str) -> None:
    """Update version in pyproject.toml."""
    content = pyproject_path.read_text()
    new_content = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        f'\\1"{new_version}"',
        content,
        flags=re.MULTILINE
    )
    pyproject_path.write_text(new_content)


def create_release(version: str) -> None:
    """Create git commit, tag, and GitHub release."""
    # Update lock file
    print("Updating lock file...")
    run_cmd(["uv", "lock"], capture=False)

    # Commit the version change
    run_cmd(["git", "add", "pyproject.toml", "uv.lock"])
    run_cmd(["git", "commit", "-m", f"chore: bump version to {version}"])

    # Create tag
    run_cmd(["git", "tag", version])

    # Push commit and tag
    print("Pushing to origin...")
    run_cmd(["git", "push", "origin", "main"])
    run_cmd(["git", "push", "origin", version])

    # Create GitHub release
    print("Creating GitHub release...")
    run_cmd(["gh", "release", "create", version, "--generate-notes"], capture=False)


def main():
    """Main entry point."""
    print("=== Release Script ===\n")

    # Pre-flight checks
    print("Running pre-flight checks...")
    if not check_branch():
        sys.exit(1)
    if not check_clean():
        sys.exit(1)
    if not check_pushed():
        sys.exit(1)
    print("All checks passed!\n")

    # Get current version
    repo_root = get_repo_root()
    pyproject_path = repo_root / "pyproject.toml"
    current_version = get_current_version(pyproject_path)
    proposed_version = bump_patch(current_version)

    # Prompt for version
    print(f"Current version: {current_version}")
    print(f"Proposed version: {proposed_version}")
    user_input = input(f"\nPress Enter to accept '{proposed_version}' or type a new version: ").strip()

    if user_input:
        new_version = user_input
        if not validate_version(new_version):
            print(f"Error: Invalid version format '{new_version}'. Expected: X.Y.Z")
            sys.exit(1)
    else:
        new_version = proposed_version

    # Confirm
    print(f"\nWill release version: {new_version}")
    confirm = input("Proceed? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    # Update version and release
    print(f"\nUpdating version to {new_version}...")
    update_version(pyproject_path, new_version)

    print("Creating release...")
    create_release(new_version)

    print(f"\n=== Released {new_version} successfully! ===")


if __name__ == "__main__":
    main()
