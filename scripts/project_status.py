#!/usr/bin/env python3
"""Live project-status probe for NUANCE MCP.

Used by the ``/goals`` Claude Code slash command, but also runnable on its
own::

    python scripts/project_status.py            # human-readable report
    python scripts/project_status.py --json      # machine-readable
    python scripts/project_status.py --no-tests  # skip the pytest run

The probe is deliberately defensive — every section is wrapped so a missing
dependency (e.g. pytest, or the PyJEM driver) degrades to a warning rather
than crashing the whole report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Individual probes
# --------------------------------------------------------------------------


def _version() -> str:
    init = REPO / "src" / "nuance_mcp" / "__init__.py"
    try:
        m = re.search(r'__version__\s*=\s*"([^"]+)"', init.read_text())
        return m.group(1) if m else "unknown"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def _git() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10
        ).stdout.strip()

    try:
        branch = run("branch", "--show-current")
        commits = run("rev-list", "--count", "HEAD")
        porcelain = run("status", "--porcelain")
        dirty = [ln for ln in porcelain.splitlines() if ln.strip()]
        last = run("log", "-1", "--pretty=%h %s")
        return {
            "branch": branch or "(detached)",
            "commits": int(commits) if commits.isdigit() else commits,
            "uncommitted_files": len(dirty),
            "uncommitted": [ln.strip() for ln in dirty[:12]],
            "last_commit": last,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _server_inventory() -> dict:
    """Import the server in offline mode and count tools/prompts."""

    os.environ.setdefault("JEOL_MCP_MODE", "offline")
    # Keep the probe output clean — suppress the server's INFO banner so the
    # /goals slash command shows only the status report.
    os.environ.setdefault("JEOL_MCP_LOG_LEVEL", "WARNING")
    sys.path.insert(0, str(REPO / "src"))
    try:
        import asyncio

        from jeol_mcp.server import build_server

        async def _gather() -> dict:
            mcp = build_server()
            tools = await mcp.list_tools()
            prompts = await mcp.list_prompts()
            read_only = 0
            for t in tools:
                ann = getattr(t, "annotations", None)
                if ann is not None and getattr(ann, "readOnlyHint", False):
                    read_only += 1
            # Group by domain prefix (jeol_<group>_...)
            groups: dict[str, int] = {}
            for t in tools:
                parts = t.name.split("_")
                key = parts[1] if len(parts) > 1 and parts[0] == "jeol" else "other"
                groups[key] = groups.get(key, 0) + 1
            return {
                "tools": len(tools),
                "prompts": len(prompts),
                "read_only": read_only,
                "state_changing": len(tools) - read_only,
                "groups": dict(sorted(groups.items())),
            }

        return asyncio.run(_gather())
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _tests() -> dict:
    """Run pytest quietly and parse the summary line."""

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, "JEOL_MCP_MODE": "offline"},
        )
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        passed = re.search(r"(\d+) passed", tail)
        failed = re.search(r"(\d+) failed", tail)
        errors = re.search(r"(\d+) error", tail)
        return {
            "passed": int(passed.group(1)) if passed else 0,
            "failed": int(failed.group(1)) if failed else 0,
            "errors": int(errors.group(1)) if errors else 0,
            "summary": tail,
            "exit_code": proc.returncode,
        }
    except FileNotFoundError:
        return {"error": "pytest not installed"}
    except subprocess.TimeoutExpired:
        return {"error": "pytest timed out"}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _assets() -> dict:
    def count(glob: str) -> int:
        return len(list(REPO.glob(glob)))

    src_lines = 0
    for p in (REPO / "src").rglob("*.py"):
        try:
            src_lines += len(p.read_text(errors="ignore").splitlines())
        except Exception:  # noqa: BLE001
            pass
    return {
        "example_scripts": count("examples/*.py"),
        "doc_pages": count("docs/*.md"),
        "tool_modules": count("src/jeol_mcp/tools/*.py"),
        "test_files": count("tests/test_*.py"),
        "skill_present": (REPO / "skills" / "jeol" / "SKILL.md").exists(),
        "src_lines": src_lines,
    }


# --------------------------------------------------------------------------
# Roadmap — the milestone definitions. ``done`` is recomputed live where the
# project state makes that possible; otherwise it is a curated judgement.
# --------------------------------------------------------------------------


def _roadmap(inv: dict, tests: dict, assets: dict) -> list[dict]:
    tools_ok = isinstance(inv.get("tools"), int) and inv["tools"] >= 100
    tests_green = tests.get("failed", 1) == 0 and tests.get("passed", 0) > 0
    return [
        {
            "id": "M1",
            "title": "Core MCP server — full PyJEM TEM3 + detector + EDS coverage",
            "status": "done" if tools_ok else "in_progress",
            "evidence": f"{inv.get('tools', '?')} tools, {inv.get('prompts', '?')} prompts, "
            "FastMCP multi-transport (stdio/http/sse)",
        },
        {
            "id": "M2",
            "title": "Safety layer — two-key gating, soft envelopes, Pydantic validation",
            "status": "done",
            "evidence": f"{inv.get('state_changing', '?')} state-changing tools behind "
            "confirm-gate; HT/tilt/stage/DAC clamps",
        },
        {
            "id": "M3",
            "title": "API fidelity — signatures cross-checked vs pyJEM 1.3.9.3617",
            "status": "done",
            "evidence": "snapshot arg order, exposure µs, EDS param-dict, GetMagValue "
            "tuple, scan-mode/gain ranges — see CHANGELOG",
        },
        {
            "id": "M4",
            "title": "Offline-simulator validation against the real PyJEM.offline",
            "status": "pending",
            "evidence": f"{tests.get('passed', '?')} tests pass against an in-repo fake; "
            "real PyJEM.offline (JEOL-only) not yet exercised",
        },
        {
            "id": "M5",
            "title": "Hardware validation on a live JEOL column",
            "status": "pending",
            "evidence": "never run against a physical TEM — requires a beam-time session",
        },
        {
            "id": "M6",
            "title": "Evaluation suite (mcp-builder Phase 4 — 10 eval questions)",
            "status": "pending",
            "evidence": "evaluation XML not yet authored",
        },
        {
            "id": "M7",
            "title": "Packaging & PyPI release",
            "status": "in_progress",
            "evidence": "pyproject.toml + console-script ready; not yet published",
        },
        {
            "id": "M8",
            "title": "Continuous integration (GitHub Actions)",
            "status": "pending"
            if not (REPO / ".github" / "workflows").exists()
            else "done",
            "evidence": "no .github/workflows/ present"
            if not (REPO / ".github" / "workflows").exists()
            else "workflow present",
        },
        {
            "id": "M9",
            "title": "Quality gates green — test suite passing",
            "status": "done" if tests_green else "in_progress",
            "evidence": tests.get("summary", "tests not run"),
        },
    ]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

_ICON = {"done": "[x]", "in_progress": "[~]", "pending": "[ ]"}


def collect(run_tests: bool = True) -> dict:
    inv = _server_inventory()
    tests = _tests() if run_tests else {"skipped": True}
    assets = _assets()
    return {
        "version": _version(),
        "git": _git(),
        "inventory": inv,
        "tests": tests,
        "assets": assets,
        "roadmap": _roadmap(inv, tests, assets),
    }


def render(data: dict) -> str:
    L: list[str] = []
    L.append("=" * 66)
    L.append(f"  NUANCE MCP — project status   (v{data['version']})")
    L.append("=" * 66)

    g = data["git"]
    if "error" not in g:
        L.append(
            f"  git        : branch {g['branch']}, {g['commits']} commits, "
            f"{g['uncommitted_files']} uncommitted file(s)"
        )
        L.append(f"  last commit: {g['last_commit']}")
    else:
        L.append(f"  git        : (unavailable — {g['error']})")

    inv = data["inventory"]
    if "error" not in inv:
        L.append(
            f"  surface    : {inv['tools']} tools "
            f"({inv['read_only']} read-only / {inv['state_changing']} state-changing), "
            f"{inv['prompts']} workflow prompts"
        )
    else:
        L.append(f"  surface    : (server import failed — {inv['error']})")

    t = data["tests"]
    if t.get("skipped"):
        L.append("  tests      : (skipped)")
    elif "error" in t:
        L.append(f"  tests      : (unavailable — {t['error']})")
    else:
        verdict = "GREEN" if t["failed"] == 0 and t["passed"] > 0 else "RED"
        L.append(
            f"  tests      : {verdict} — {t['passed']} passed, "
            f"{t['failed']} failed, {t['errors']} error(s)"
        )

    a = data["assets"]
    L.append(
        f"  assets     : {a['tool_modules']} tool modules, {a['test_files']} test files, "
        f"{a['example_scripts']} examples, {a['doc_pages']} docs, "
        f"{'SKILL.md OK' if a['skill_present'] else 'SKILL.md MISSING'}, "
        f"{a['src_lines']} src LOC"
    )

    if inv.get("groups"):
        L.append("")
        L.append("  tool groups:")
        row = "    "
        for name, n in inv["groups"].items():
            chunk = f"{name}={n}  "
            if len(row) + len(chunk) > 64:
                L.append(row)
                row = "    "
            row += chunk
        if row.strip():
            L.append(row)

    L.append("")
    L.append("-" * 66)
    L.append("  ROADMAP")
    L.append("-" * 66)
    done = sum(1 for m in data["roadmap"] if m["status"] == "done")
    total = len(data["roadmap"])
    for m in data["roadmap"]:
        L.append(f"  {_ICON[m['status']]} {m['id']}  {m['title']}")
        L.append(f"        └ {m['evidence']}")
    L.append("")
    L.append(
        f"  Progress: {done}/{total} milestones complete "
        f"({round(100 * done / total)}%)."
    )
    L.append("=" * 66)
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="NUANCE MCP status probe.")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p.add_argument("--no-tests", action="store_true", help="skip the pytest run")
    args = p.parse_args(argv)

    data = collect(run_tests=not args.no_tests)
    if args.json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(render(data))
    # Exit non-zero if tests are red, so CI / the slash command can react.
    t = data["tests"]
    if not t.get("skipped") and "error" not in t and t.get("failed", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
