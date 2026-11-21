"""Rebuild the explicitly scoped, local-only Graphify navigation artifacts.

Run with the separate Graphify environment; no project modules are imported.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ["GRAPHIFY_BACKEND"] = "none"
os.environ["GRAPHIFY_QUERY_LOG_DISABLE"] = "1"

from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.detect import count_words, save_manifest
from graphify.diagnostics import diagnose_extraction, format_diagnostic_report
from graphify.export import to_json
from graphify.exporters.html import to_html
from graphify.extract import extract
from graphify.report import generate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "graphify-out"
SOURCE_DIRS = (
    "ws/src/cube_task/cube_task",
    "ws/src/piper_vision/piper_vision",
)
LABELS = {
    "server": "Web Service Routes",
    "physics": "Interactive Physics Session",
    "skill_orchestrator": "Composable Skill Orchestration",
    "skill_execution": "Validated Skill Execution",
    "llm": "Language Model Planning",
    "stacking": "Stack Placement Geometry",
    "batch": "Sequential Batch Planning",
    "conversation": "Dialogue Context Management",
    "simulation": "Preview Simulation Session",
    "active_observation": "Active View Planning",
    "placement_observation": "Placement Region Observation",
    "stack_perception": "Stack Scene Observation",
    "scene": "MuJoCo Scene Construction",
    "engine": "MuJoCo Physics Engine",
    "arm": "MuJoCo Arm Control",
    "world": "Simulation World State",
    "runner": "Legacy Task Runner",
    "skills": "Legacy Motion Skills",
    "l1": "Standalone Cube Detection",
    "handeye": "Camera Handeye Calibration",
    "camera": "Camera Configuration Geometry",
    "transforms": "Coordinate Transform Utilities",
    "app": "Browser Task Controls",
    "editor": "Browser Scene Editor",
}


def write_json(name: str, value: object) -> None:
    (OUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    os.chdir(ROOT)
    # A positive extension and directory list prevents discovery from ever
    # opening .env, conversations, output records, or third-party code.
    files = sorted(
        path
        for relative in SOURCE_DIRS
        for path in (ROOT / relative).rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix in {".py", ".js"}
        and "__pycache__" not in path.parts
    )
    if not files or len(files) > 500:
        raise SystemExit(f"Scope needs review: {len(files)} source files")
    total_words = sum(count_words(path) for path in files)
    if total_words > 2_000_000:
        raise SystemExit(f"Scope needs review: {total_words} words")
    relative_files = [path.relative_to(ROOT).as_posix() for path in files]
    detection = {
        "scan_root": str(ROOT),
        "files": {"code": [str(path) for path in files]},
        "total_files": len(files),
        "total_words": total_words,
        "warning": "Scoped local AST only; repository documentation, tests, configs, assets and third-party packages are outside this graph.",
    }
    print(f"Corpus: {len(files)} code files, approximately {total_words:,} words")
    (OUT / ".graphify_python").write_text(sys.executable, encoding="utf-8")
    (OUT / ".graphify_root").write_text(str(ROOT), encoding="utf-8")
    extraction = extract(files, cache_root=ROOT, root=ROOT)
    graph = build_from_json(extraction, root=ROOT, directed=True)
    if not graph.number_of_nodes():
        raise SystemExit("Empty extraction; existing graph preserved")
    communities = cluster(graph)
    cohesion = score_all(graph, communities)
    labels = {}
    community_summary = []
    for cid, members in communities.items():
        counts = Counter(
            Path(graph.nodes[node].get("source_file", "unknown")).stem
            for node in members
        )
        dominant = counts.most_common(1)[0][0]
        labels[cid] = LABELS.get(dominant, dominant.replace("_", " ").title() + " Components")
        community_summary.append({
            "id": cid, "label": labels[cid], "nodes": len(members),
            "cohesion": cohesion[cid], "top_files": counts.most_common(5),
        })
    diagnostics = diagnose_extraction(extraction, directed=True, root=ROOT)
    # Keep the upstream diagnostic source portable across local installations.
    producer = diagnostics.get("producer_suppression", {})
    if producer.get("path"):
        producer["path"] = "graphify/extract.py (installed package)"
    print(format_diagnostic_report(diagnostics))
    if not to_json(graph, communities, str(OUT / "graph.json"), community_labels=labels):
        raise SystemExit("Graphify refused a smaller graph; inspect scope before replacing existing outputs")
    gods = god_nodes(graph)
    surprises = surprising_connections(graph, communities)
    questions = suggest_questions(graph, communities, labels)
    report = generate(
        graph, communities, cohesion, labels, gods, surprises, detection,
        {"input": 0, "output": 0}, str(ROOT), suggested_questions=questions,
    )
    report += "\n## Scope and interpretation\n\n"
    report += "See [scope.json](scope.json) for the exact source list and [the project guide](../docs/architecture/graphify-guide.md) for verified execution boundaries.\n\n"
    report += "Only Graphify AST extraction was used: no semantic model calls, uploaded project files, or API keys. The zero token count refers to Graphify model calls only. Community names use dominant source files; clustering does not establish runtime boundaries.\n\n"
    report += "Graph health diagnostics are recorded in [health.json](health.json). Static resolution can have dangling/merged edges and cannot establish runtime reachability, browser HTTP wiring, physical grasp success, or ROS hardware readiness.\n\n"
    report += (
        f"Health warning: {diagnostics['dangling_endpoint_edges']} raw AST edges have unresolved endpoints; "
        f"{diagnostics['self_loop_edges']} self-loops; "
        f"{diagnostics['directed_same_endpoint_collapsed_edges']} same-endpoint edges merge in the simple directed graph. "
        "These are navigation limits, not confirmed application defects.\n\n"
    )
    report += "graphify-out/ is written into the working tree; review which generated outputs to share before committing.\n"
    (OUT / "GRAPH_REPORT.md").write_text(report, encoding="utf-8")
    write_json(".graphify_labels.json", {str(key): value for key, value in labels.items()})
    write_json("communities.json", community_summary)
    write_json("health.json", diagnostics)
    write_json("scope.json", {
        "graphify_version": version("graphifyy"),
        "skill_source": "https://github.com/Graphify-Labs/graphify",
        "skill_ref": "b582ad401aaef4378935ec6620e3b165ae8e03c1",
        "mode": "local deterministic AST; GRAPHIFY_BACKEND=none",
        "directed": True,
        "clustering": "NetworkX Louvain fallback (Leiden extra not installed)",
        "source_directories": SOURCE_DIRS,
        "extensions": [".py", ".js"],
        "files": relative_files,
        "excluded": [
            ".env and credentials", ".git", "docs and conversations", "tests",
            "configs", "HTML and CSS", "models and media", "recorded runs",
            "third_party", "other ROS packages", "runtime environments",
        ],
    })
    write_json("cost.json", {
        "runs": [{"date": datetime.now(timezone.utc).isoformat(),
                  "input_tokens": 0, "output_tokens": 0, "files": len(files)}],
        "total_input_tokens": 0, "total_output_tokens": 0,
        "note": "Graphify external-model calls only; host conversation usage is not counted",
    })
    save_manifest(detection["files"], root=ROOT, scan_corpus=set(detection["files"]["code"]))
    if graph.number_of_nodes() > 5000:
        print("More than 5,000 nodes: HTML will show aggregated communities")
    to_html(graph, communities, str(OUT / "graph.html"), community_labels=labels, node_limit=5000)
    to_html(graph, communities, str(OUT / "graph-overview.html"), community_labels=labels, node_limit=1000)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, {len(communities)} communities")
    print("Graphify model tokens: 0 input, 0 output")


if __name__ == "__main__":
    main()
