"""
Fails a pull request that adds a broken claim to the agent guidance files.

For the intel-ci.yml GitHub Actions workflow's "Agent guidance" job.

Guidance files are AGENTS.md and CLAUDE.md at any depth,
.github/copilot-instructions.md, .github/instructions/, SKILL.md,
.claude/skills/ and skills/.

Usage:
  python ci/guidance.py [--event pull_request|push] [--base REV] [--head REV]
                        [--work DIR]

On a pull_request, with no guidance file changed between the merge base of the
pull request and its head, the job passes. Otherwise it installs dokimi with
hashes, scans the merge base and the head with `dokimi repo scan` (no network,
no model), and fails on a claim that fails at the head, in a changed guidance
file, and did not fail in that file at the merge base. A claim is its file,
kind and text, never its line. Claims that already fail do not fail it, and
neither do untouched files. It also fails when the merge base had guidance and
the head has none. A repeated copy of a failing claim counts as new. On a
push it scans the head, reports, and always passes. A summary goes
to $GITHUB_STEP_SUMMARY. The event, --base and --head default to the
workflow's. Needs only the Python standard library, and Python 3.11 or newer
for dokimi.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import venv
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = Path(__file__).resolve().parent / "guidance-requirements.txt"
SCHEMA = "repo_readiness.v1"
# dokimi's default globs, plus skills
GLOBS = [
    "**/AGENTS.md",
    "**/CLAUDE.md",
    ".github/copilot-instructions.md",
    ".github/instructions/**/*.md",
    "**/SKILL.md",
    ".claude/skills/**/*.md",
    "skills/**/*.md",
]
MAX_TEXT = 200


class GuidanceError(Exception):
    """Raised when the files or the scan cannot be trusted."""


def is_guidance(path: str) -> bool:
    """Whether a repository path is a guidance file."""
    name = PurePosixPath(path).name
    return (
        name in ("AGENTS.md", "CLAUDE.md", "SKILL.md")
        or path == ".github/copilot-instructions.md"
        or (
            path.endswith(".md")
            and path.startswith(
                (".github/instructions/", ".claude/skills/", "skills/")
            )
        )
    )


def git(*args: str) -> str:
    """Output of git run in this repository."""
    done = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if done.returncode:
        raise GuidanceError(
            f"git {' '.join(args)} failed: {done.stderr.strip()}; "
            "the checkout needs fetch-depth: 0"
        )
    return done.stdout


def has_file(rev: str, name: str) -> bool:
    """Whether the tree at rev holds a file."""
    return (
        subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "-e", f"{rev}:{name}"],
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def touched_files(base: str, head: str) -> list[str]:
    """Guidance files that differ between the merge base and the head."""
    names = git("diff", "--name-only", "--no-renames", "-z", base, head)
    return sorted(path for path in names.split("\0") if is_guidance(path))


def install_dokimi(work: Path) -> Path:
    """Installs the pinned dokimi into a new environment; its executable."""
    env = work / "venv"
    venv.create(env, with_pip=True)
    bindir = env / ("Scripts" if os.name == "nt" else "bin")
    command = [
        str(bindir / "python"),
        "-m",
        "pip",
        "install",
        "--quiet",
        "--require-hashes",
        "--no-deps",
        "--only-binary",
        ":all:",
        "--requirement",
        str(REQUIREMENTS),
    ]
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)
    return bindir / "dokimi"


def write_manifest(work: Path, base: str) -> Path:
    """The guidance-only manifest for both scans, from the base tree."""
    guidance = {"globs": GLOBS}
    if has_file(base, "AGENTS.md"):
        guidance["also_scan"] = ["AGENTS.md"]
    manifest = {
        "schema": "repo_manifest.v1",
        "id": "guidance",
        "guidance": guidance,
    }
    path = work / "manifest.yaml"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def scan(
    dokimi: Path, work: Path, manifest: Path, label: str, rev: str
) -> dict:
    """The dokimi report of the repository at one revision."""
    out = work / f"{label}.json"
    command = [
        str(dokimi),
        "repo",
        "scan",
        f"{ROOT}@{rev}",
        "--manifest-file",
        str(manifest),
        "--cache-dir",
        str(work / "cache"),
        "--out",
        str(out),
    ]
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)
    report = json.loads(out.read_text(encoding="utf-8"))
    if report.get("schema_version") != SCHEMA:
        raise GuidanceError(
            f"dokimi wrote {report.get('schema_version')!r}, not {SCHEMA}"
        )
    return report


def failing(report: dict) -> list[dict]:
    """The claims that fail in a report."""
    return [c for c in report["claims"] if c["status"] == "fail"]


def key(claim: dict) -> tuple[str, str, str]:
    """What identifies a claim across revisions."""
    return claim["file"], claim["kind"], claim["claim"]


def new_failures(base: dict, head: dict, touched: list[str]) -> list[dict]:
    """Failing head claims in touched files that did not fail at the base."""
    old = Counter(key(claim) for claim in failing(base))
    found = []
    for claim in failing(head):
        if claim["file"] not in touched:
            continue
        if old[key(claim)] > 0:
            old[key(claim)] -= 1
        else:
            found.append(claim)
    return found


def code(text: object) -> str:
    """Untrusted text as one Markdown code span: no links, no table breaks."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) > MAX_TEXT:
        text = text[: MAX_TEXT - 3] + "..."
    text = text.replace("`", "'").replace("|", "\\|")
    return f"`{text}`"


def plain(text: object) -> str:
    """Untrusted text for the log, which cannot start a workflow command."""
    text = re.sub(r"[^\x20-\x7e\n]", "?", str(text))
    return text.replace("::", ": :")


def claim_counts(report: dict) -> str:
    """One line of a report's counts."""
    counts = report["counts"]
    return (
        f"Claims: {counts['total']} "
        f"(ok {counts['ok']}, unresolved {counts['unresolved']}, "
        f"fail {counts['fail']})."
    )


def summarize(
    touched: list[str], head: dict, found: list[dict], problem: str
) -> str:
    """The Markdown job summary of a scanned pull request."""
    lines = [
        "## Agent guidance",
        "",
        f"Guidance files changed: {len(touched)}.",
        claim_counts(head),
        f"New failing claims: {len(found)}.",
    ]
    if problem:
        lines.extend(["", problem])
    if found:
        lines.extend(
            ["", "| Location | Kind | Claim | Why |", "|---|---|---|---|"]
        )
        for claim in found:
            location = f"{claim['file']}:{claim['line']}"
            lines.append(
                f"| {code(location)} | {code(claim['kind'])} "
                f"| {code(claim['claim'])} | {code(claim['why'])} |"
            )
    return "\n".join([*lines, ""])


def pull_request_range(args: argparse.Namespace) -> tuple[str, str]:
    """The pull request's base and head revisions."""
    base, head = args.base, args.head
    if not (base and head):
        with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as file:
            event = json.load(file)["pull_request"]
        base = base or event["base"]["sha"]
        head = head or event["head"]["sha"]
    return git("merge-base", base, head).strip(), head


def check(args: argparse.Namespace, work: Path) -> tuple[str, int]:
    """The summary and exit status for one event."""
    if args.event == "push":
        head = args.head or os.environ["GITHUB_SHA"]
        try:
            dokimi = install_dokimi(work)
            manifest = write_manifest(work, head)
            report = scan(dokimi, work, manifest, "head", head)
        except (GuidanceError, subprocess.CalledProcessError) as error:
            text = f"Report only, the scan failed: {plain(error)}"
            return f"## Agent guidance\n\n{text}\n", 0
        text = f"## Agent guidance\n\nReport only.\n{claim_counts(report)}\n"
        return text, 0
    base, head = pull_request_range(args)
    touched = touched_files(base, head)
    if not touched:
        return "## Agent guidance\n\nNo guidance files changed.\n", 0
    dokimi = install_dokimi(work)
    manifest = write_manifest(work, base)
    old = scan(dokimi, work, manifest, "base", base)
    new = scan(dokimi, work, manifest, "head", head)
    found = new_failures(old, new, touched)
    problem = ""
    if new["incomplete"] and not old["incomplete"]:
        problem = "The head has no guidance left to scan, the base had some."
    return summarize(touched, new, found, problem), int(bool(found or problem))


def main(argv: list[str] | None = None) -> int:
    """Runs the gate."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--event",
        default=os.environ.get("GITHUB_EVENT_NAME", "pull_request"),
        choices=("pull_request", "push"),
    )
    parser.add_argument("--base", help="the pull request's base revision")
    parser.add_argument("--head", help="the pull request's head revision")
    parser.add_argument(
        "--work", type=Path, help="scratch directory; default is a new one"
    )
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as scratch:
        summary, status = check(args, args.work or Path(scratch))
    print(plain(summary))
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as file:
            file.write(summary)
    return status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (GuidanceError, subprocess.CalledProcessError) as error:
        print(f"guidance: {plain(error)}", file=sys.stderr)
        sys.exit(1)
