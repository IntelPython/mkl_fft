"""
Runs ci/test.py against one built artifact, the way Intel Python CI does.

For the GitHub Actions test job.

Each Python version's artifact is installed into a fresh conda environment from
a local channel plus conda-forge, pinned to its exact build, with the test
requirements ci/config.yml lists, and the test runs with that environment's
Python from an empty working directory, on Linux or Windows. The interpreter
spec and the requirements are the ones Jenkins uses. It gets the same variables,
and nothing else from the runner, as it does in Jenkins, so a test that passes
here passes there. Jenkins runs ci/test.py again on its own machines, GPUs
included; the scans are Jenkins' own and never run here.

Usage, from any Python 3.10 or newer; it runs itself again with the conda base
install's Python, from $CONDA, which setup-miniconda sets, or the conda on PATH:
  python ci/hooks.py --python 3.14 --artifacts DIR --work DIR

Needs conda, conda-package-handling and ruamel.yaml, which every conda base has;
installs conda-index into base when it is missing.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matrix

# Every conda base has these, as dependencies of conda itself. ci/config.yml is
# read the way Jenkins reads it: with PyYAML when it is there, else ruamel.yaml.
try:
    from conda_package_handling import api as CPH_API
except ModuleNotFoundError:
    CPH_API = None
try:
    import yaml as PYYAML
except ModuleNotFoundError:
    PYYAML = None
try:
    from ruamel.yaml import YAML as RUAMEL_YAML
except ModuleNotFoundError:
    RUAMEL_YAML = None

ROOT = matrix.ROOT
IS_WINDOWS = os.name == "nt"
PLATFORM = "windows" if IS_WINDOWS else "linux"
CONDA_SUFFIXES = (".conda", ".tar.bz2")
# What the test may see of the runner's environment; tokens and everything else
# are dropped. The Windows names are what any Windows process needs to start and
# find its temp directory.
KEEP = (
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "TERM",
    "TMPDIR",
    "TZ",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CONDA_PKGS_DIRS",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
)


class HookError(Exception):
    """Raised when the artifact cannot be installed or ci/test.py is missing."""


def run(command: list[str], **options: object) -> None:
    """Echoes and runs a command, never through a shell."""
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True, **options)


def conda(*args: str) -> None:
    """Runs conda from the base environment running this script."""
    run([sys.executable, "-m", "conda", *args])


def stage_channel(packages: list[Path], channel: Path) -> None:
    """
    Lays the packages out as an indexed local channel, by the subdir each one
    declares.
    """
    if CPH_API is None:
        raise HookError(
            "run this with a conda base Python, "
            "which has conda-package-handling"
        )
    for package in packages:
        with tempfile.TemporaryDirectory() as scratch:
            CPH_API.extract(str(package), scratch, components="info")
            index = json.loads(
                (Path(scratch) / "info" / "index.json").read_text(
                    encoding="utf-8"
                )
            )
        subdir = channel / str(index.get("subdir") or "noarch")
        subdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package, subdir / package.name)
    (channel / "noarch").mkdir(parents=True, exist_ok=True)
    if subprocess.run(
        [sys.executable, "-c", "import conda_index"], check=False
    ).returncode:
        conda(
            "install",
            "-y",
            "-q",
            "-n",
            "base",
            "--override-channels",
            "-c",
            "conda-forge",
            "conda-index",
        )
    run([sys.executable, "-m", "conda_index", str(channel)])


def test_python(version: str) -> str:
    """The interpreter spec for the test environment, as Jenkins spells it."""
    minor, threaded = matrix.version_key(version)
    # The .* matters: conda reads a bare 3.11 as exactly 3.11.0, not the newest
    # 3.11.
    if minor < matrix.FREE_THREADED_MINOR:
        return f"3.{minor}.*"
    # From 3.13 on conda-forge marks the ABI, so 3.14 cannot resolve to the
    # free-threaded build.
    return f'3.{minor}.* *_cp3{minor}{"t" if threaded else ""}'


def test_requirements() -> tuple[list[str], list[str]]:
    """
    The conda-forge and pip packages ci/config.yml installs next to the
    artifact.
    """
    text = (ROOT / "ci" / "config.yml").read_text(encoding="utf-8")
    if PYYAML is not None:
        config = PYYAML.safe_load(text)
    elif RUAMEL_YAML is not None:
        config = RUAMEL_YAML(typ="safe", pure=True).load(text)
    else:
        raise HookError("reading ci/config.yml needs PyYAML or ruamel.yaml")
    test = (config or {}).get("test") or {}
    lists = []
    for key in ("requires", "pip_requires"):
        values = test.get(key) or []
        if not isinstance(values, list) or not all(
            isinstance(v, str) for v in values
        ):
            raise HookError(
                f"ci/config.yml: test.{key} must be a list of package specs"
            )
        lists.append(values)
    return lists[0], lists[1]


def exact_spec(package: Path) -> str:
    """name=version=build from a conda package file name."""
    stem = package.name
    for suffix in CONDA_SUFFIXES:
        stem = stem.removesuffix(suffix)
    name, version, build = stem.rsplit("-", 2)
    return f"{name}={version}={build}"


def env_python(env_dir: Path) -> Path:
    """The interpreter of a conda environment."""
    return env_dir / "python.exe" if IS_WINDOWS else env_dir / "bin" / "python"


def env_path(env_dir: Path) -> list[Path]:
    """
    The directories conda's activation would put first on PATH for an
    environment.
    """
    if not IS_WINDOWS:
        return [env_dir / "bin"]
    library = env_dir / "Library"
    return [
        env_dir,
        library / "mingw-w64" / "bin",
        library / "usr" / "bin",
        library / "bin",
        env_dir / "Scripts",
        env_dir / "bin",
    ]


def test_env(
    env_dir: Path, artifacts: Path, results: Path, args: argparse.Namespace
) -> dict:
    """The variables Jenkins gives ci/test.py, and no runner tokens."""
    env = {key: value for key, value in os.environ.items() if key in KEEP}
    env["PATH"] = os.pathsep.join(
        [*map(str, env_path(env_dir)), env.get("PATH", "")]
    )
    env.update(
        {
            "CONDA_PREFIX": str(env_dir),
            "GH_CI_ARTIFACTS": str(artifacts),
            "GH_CI_SOURCE": str(ROOT),
            "GH_CI_RESULTS": str(results),
            "GH_CI_KIND": "conda",
            "GH_CI_PYTHON": args.python,
            "GH_CI_PLATFORM": PLATFORM,
            "GH_CI_EVENT": args.event,
            "GH_CI_COMMIT": args.commit,
        }
    )
    return env


def summarize(lines: list[str]) -> None:
    """Adds the results to the job summary page, in GitHub Actions."""
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as page:
            page.write("\n".join(lines) + "\n")


def install(args: argparse.Namespace) -> Path:
    """
    Creates the test environment with the artifact, pinned to its exact build;
    returns it.
    """
    system, selector = matrix.detect(ROOT)
    declared = {
        python.version: python for python in matrix.declared(system, selector)
    }
    if args.python not in declared:
        raise HookError(
            f"Python {args.python} is not declared; "
            f'declared: {", ".join(declared)}'
        )
    packages = sorted(
        path
        for path in args.artifacts.rglob("*")
        if path.name.endswith(CONDA_SUFFIXES)
    )
    if not packages:
        raise HookError(f"{args.artifacts} holds no conda package")
    requires, pip_requires = test_requirements()
    channel, env_dir = args.work / "channel", args.work / "env"
    stage_channel(packages, channel)
    conda(
        "create",
        "-y",
        "-q",
        "-p",
        str(env_dir),
        "--override-channels",
        "-c",
        channel.as_uri(),
        "-c",
        "conda-forge",
        f"python {test_python(args.python)}",
        *requires,
        *map(exact_spec, packages),
    )
    if pip_requires:
        run(
            [
                str(env_python(env_dir)),
                "-m",
                "pip",
                "install",
                "--no-input",
                *pip_requires,
            ]
        )
    return env_dir


def run_test(env_dir: Path, args: argparse.Namespace) -> int:
    """Runs ci/test.py with the environment's Python; returns its status."""
    script = ROOT / "ci" / "test.py"
    if not script.is_file():
        raise HookError("ci/test.py is required and missing")
    results, scratch = (
        args.work / "results" / "test",
        args.work / "scratch" / "test",
    )
    for directory in (results, scratch):
        directory.mkdir(parents=True, exist_ok=True)
    command = [str(env_python(env_dir)), str(script)]
    print(f"+ {shlex.join(command)}", flush=True)
    status = subprocess.run(
        command,
        cwd=scratch,
        check=False,
        env=test_env(env_dir, args.artifacts, results, args),
    ).returncode
    result = "passed" if status == 0 else f"failed, exit {status}"
    summarize(
        [
            f"### {PLATFORM.capitalize()} Python {args.python}",
            "",
            "| Test | Result |",
            "|---|---|",
            f"| ci/test.py | {result} |",
        ]
    )
    return status


def conda_base_python() -> Path:
    """The conda base install's Python: from $CONDA, else the conda on PATH."""
    root = os.environ.get("CONDA", "")
    if not root:
        found = shutil.which("conda")
        if not found:
            raise HookError(
                "no conda base install: set CONDA or put conda on PATH"
            )
        # conda lives in bin, Scripts or condabin, one level below the install.
        root = str(Path(found).resolve().parent.parent)
    python = env_python(Path(root))
    if not python.is_file():
        raise HookError(
            f"{root} is not a conda base install; it has no {python.name}"
        )
    return python


def main(argv: list[str] | None = None) -> int:
    """Installs the artifact, then runs ci/test.py against it."""
    if CPH_API is None:
        # conda and conda-package-handling come with the base install, so run
        # there instead.
        command = [
            str(conda_base_python()),
            str(Path(__file__).resolve()),
            *(sys.argv[1:] if argv is None else argv),
        ]
        return subprocess.run(command, check=False).returncode
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--python", required=True, help="a declared version, such as 3.14t"
    )
    parser.add_argument(
        "--artifacts", required=True, type=Path, help="the downloaded artifact"
    )
    parser.add_argument(
        "--work", required=True, type=Path, help="an empty scratch directory"
    )
    parser.add_argument(
        "--event", default="push", choices=("pr", "tag", "push")
    )
    parser.add_argument("--commit", default="")
    args = parser.parse_args(argv)
    args.artifacts, args.work = args.artifacts.resolve(), args.work.resolve()
    return run_test(install(args), args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        HookError,
        matrix.DeclarationError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"hooks: {error}", file=sys.stderr)
        sys.exit(1)
