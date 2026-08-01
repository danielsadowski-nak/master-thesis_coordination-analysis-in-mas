"""Generate a static HTML review form for Phase C manual annotation."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.mast_classifier import MASTFailureMode, MAST_FAILURE_MODE_DEFINITIONS


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _read_trace_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"[Trace konnte nicht gelesen werden: {exc}]"


def _load_items(reviewer_csv: Path, worklist_csv: Path, reviewer_tag: str) -> tuple[list[str], list[dict[str, Any]]]:
    reviewer_df = pd.read_csv(reviewer_csv)
    worklist_df = pd.read_csv(worklist_csv)
    reviewer_worklist = worklist_df[worklist_df["reviewer"].astype(str) == reviewer_tag].copy()
    if reviewer_worklist.empty:
        raise ValueError(f"Keine Worklist-Einträge für {reviewer_tag} in {worklist_csv}")

    reviewer_df = reviewer_df.copy()
    reviewer_df["annotation_item_id"] = reviewer_df["annotation_item_id"].astype(str)
    reviewer_worklist["annotation_item_id"] = reviewer_worklist["annotation_item_id"].astype(str)

    merged = reviewer_worklist.merge(reviewer_df, on="annotation_item_id", how="left", suffixes=("_worklist", ""), validate="one_to_one")
    if merged.isna().any().any() and merged["run_id"].isna().any():
        missing = merged.loc[merged["run_id"].isna(), "annotation_item_id"].tolist()
        raise ValueError(f"Worklist-Einträge nicht im Reviewer-Sheet gefunden: {missing}")

    columns = reviewer_df.columns.tolist()
    items: list[dict[str, Any]] = []
    for _, row in merged.sort_values("sequence").iterrows():
        raw_log_path = Path(_safe_text(row.get("raw_log_path_worklist") or row.get("raw_log_path")))
        resolved_log_path = _resolve_repo_path(raw_log_path)
        items.append(
            {
                "sequence": int(row["sequence"]),
                "reviewer": reviewer_tag,
                "annotation_item_id": _safe_text(row.get("annotation_item_id")),
                "framework": _safe_text(row.get("framework")),
                "benchmark": _safe_text(row.get("benchmark")),
                "run_index": _safe_text(row.get("run_index")),
                "run_id": _safe_text(row.get("run_id")),
                "raw_log_path": str(resolved_log_path),
                "success": _safe_text(row.get("success")),
                "final_output": _safe_text(row.get("final_output")),
                "manual_task_successful": _safe_text(row.get("manual_task_successful")),
                "manual_primary_failure_modes": _safe_text(row.get("manual_primary_failure_modes")),
                "manual_summary": _safe_text(row.get("manual_summary")),
                "reviewer_id": _safe_text(row.get("reviewer_id")),
                "adjudicated_task_successful": _safe_text(row.get("adjudicated_task_successful")),
                "adjudicated_primary_failure_modes": _safe_text(row.get("adjudicated_primary_failure_modes")),
                "adjudicated_summary": _safe_text(row.get("adjudicated_summary")),
                "notes": _safe_text(row.get("notes")),
                "missing_fields": _safe_text(row.get("missing_fields")),
                "trace_text": _read_trace_text(resolved_log_path),
            }
        )

    return columns, items


def _build_modes_payload() -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for mode in MASTFailureMode:
        payload.append(
            {
                "label": mode.value,
                "definition": MAST_FAILURE_MODE_DEFINITIONS.get(mode, ""),
            }
        )
    return payload


def _render_html(*, reviewer_tag: str, source_csv: Path, columns: list[str], items: list[dict[str, Any]]) -> str:
    payload = {
        "reviewerTag": reviewer_tag,
        "sourceCsv": str(source_csv),
        "columns": columns,
        "items": items,
        "modes": _build_modes_payload(),
    }
    json_payload = json.dumps(payload, ensure_ascii=False)
    title = html.escape(f"Phase C Review Form - {reviewer_tag}")
    return f"""<!doctype html>
<html lang=\"de\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: #fffdf8;
      --ink: #1f2933;
      --muted: #52606d;
      --line: #d9cdbd;
      --accent: #9f3a2d;
      --accent-soft: #f3dfd8;
      --ok: #1f7a4f;
      --warn: #8d5b16;
      --shadow: 0 18px 40px rgba(64, 46, 23, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Georgia, 'Times New Roman', serif; color: var(--ink); background: linear-gradient(180deg, #efe5d8 0%, var(--bg) 100%); }}
    .shell {{ display: grid; grid-template-columns: 320px 1fr; min-height: 100vh; }}
    .sidebar {{ border-right: 1px solid var(--line); background: rgba(255,255,255,0.5); padding: 20px; position: sticky; top: 0; height: 100vh; overflow: auto; }}
    .main {{ padding: 24px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow); padding: 18px; margin-bottom: 18px; }}
    h1, h2, h3 {{ margin: 0 0 12px; line-height: 1.2; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 22px; }}
    h3 {{ font-size: 16px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }}
    p, li, label, summary, button, input, textarea {{ font-size: 15px; line-height: 1.45; }}
    .meta {{ color: var(--muted); font-size: 14px; }}
    .stats {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin: 12px 0 18px; }}
    .stat {{ background: #fff7ee; border: 1px solid var(--line); border-radius: 12px; padding: 12px; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0; }}
    button {{ border: 1px solid var(--ink); border-radius: 999px; background: white; padding: 10px 14px; cursor: pointer; }}
    button.primary {{ background: var(--accent); border-color: var(--accent); color: white; }}
    button.secondary {{ background: var(--accent-soft); border-color: var(--line); }}
    .item-list {{ display: grid; gap: 10px; }}
    .item-button {{ width: 100%; text-align: left; border-radius: 14px; padding: 12px; border: 1px solid var(--line); background: #fff; }}
    .item-button.active {{ border-color: var(--accent); background: #fff8f6; }}
    .item-button.complete {{ border-color: #b7d7c8; background: #f1fbf5; }}
    .item-button .small {{ display: block; font-size: 12px; color: var(--muted); margin-top: 4px; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .field {{ margin-bottom: 16px; }}
    .field label.title {{ display: block; font-weight: 700; margin-bottom: 8px; }}
    .choices {{ display: grid; gap: 8px; }}
    .choice {{ display: flex; gap: 10px; align-items: start; padding: 10px; border: 1px solid var(--line); border-radius: 12px; background: #fff; }}
    textarea, input[type=text] {{ width: 100%; border: 1px solid var(--line); border-radius: 12px; padding: 12px; background: white; color: var(--ink); }}
    textarea {{ min-height: 110px; resize: vertical; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f9f6ef; border: 1px solid var(--line); border-radius: 12px; padding: 14px; max-height: 380px; overflow: auto; }}
    details {{ border: 1px solid var(--line); border-radius: 12px; padding: 12px; background: #fff; }}
    .hint {{ color: var(--muted); font-size: 13px; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 999px; background: var(--accent-soft); color: var(--accent); font-size: 12px; margin-right: 6px; }}
    .progress-bar {{ height: 10px; border-radius: 999px; background: #eadfce; overflow: hidden; }}
    .progress-fill {{ height: 100%; background: linear-gradient(90deg, #c25b3f 0%, #d98a4e 100%); width: 0%; }}
    .status-line {{ margin-top: 8px; font-size: 14px; color: var(--muted); }}
    .mode-def {{ font-size: 13px; color: var(--muted); margin-left: 27px; margin-top: 2px; }}
    @media (max-width: 980px) {{
      .shell {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: static; height: auto; border-right: none; border-bottom: 1px solid var(--line); }}
      .grid-2, .stats {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class=\"shell\">
    <aside class=\"sidebar\">
      <h1>Review Form</h1>
      <p class=\"meta\">{html.escape(reviewer_tag)} · lokale HTML-Datei · CSV-Export kompatibel mit der Pipeline</p>
      <div class=\"panel\">
        <h3>Ablauf</h3>
        <ol>
          <li>Reviewer-ID einmal eintragen.</li>
          <li>Für jedes Item Erfolg, Failure Modes und Kurzbegründung setzen.</li>
          <li>Zwischenspeicherung läuft automatisch im Browser.</li>
          <li>Am Ende CSV herunterladen und als Reviewer-Datei zurückgeben.</li>
        </ol>
      </div>
      <div class=\"panel\">
        <div class=\"field\">
          <label class=\"title\" for=\"global-reviewer-id\">Reviewer-ID</label>
          <input id=\"global-reviewer-id\" type=\"text\" placeholder=\"z. B. r1, ds, reviewer_a\">
        </div>
        <div class=\"progress-bar\"><div id=\"progress-fill\" class=\"progress-fill\"></div></div>
        <div id=\"progress-text\" class=\"status-line\"></div>
        <div class=\"toolbar\">
          <button id=\"download-csv\" class=\"primary\">CSV herunterladen</button>
          <button id=\"download-json\" class=\"secondary\">Backup JSON</button>
          <button id=\"clear-local\">Lokalen Speicher leeren</button>
        </div>
      </div>
      <div id=\"item-list\" class=\"item-list\"></div>
    </aside>
    <main class=\"main\">
      <div class=\"panel\">
        <h2 id=\"item-title\"></h2>
        <div id=\"item-meta\" class=\"meta\"></div>
        <div class=\"stats\">
          <div class=\"stat\"><strong>Framework</strong><div id=\"framework\"></div></div>
          <div class=\"stat\"><strong>Benchmark</strong><div id=\"benchmark\"></div></div>
          <div class=\"stat\"><strong>Run-ID</strong><div id=\"run-id\"></div></div>
          <div class=\"stat\"><strong>Trace-Pfad</strong><div id=\"trace-path\" class=\"hint\"></div></div>
        </div>
      </div>
      <div class=\"panel\">
        <h3>Automatisch übernommener Output</h3>
        <pre id=\"final-output\"></pre>
      </div>
      <div class=\"panel\">
        <details>
          <summary>Vollständigen Trace einblenden</summary>
          <pre id=\"trace-text\"></pre>
        </details>
      </div>
      <div class=\"panel\">
        <div class=\"field\">
          <label class=\"title\">1. Wurde die Aufgabe erfolgreich gelöst?</label>
          <div class=\"choices\">
            <label class=\"choice\"><input type=\"radio\" name=\"manual_task_successful\" value=\"true\"> <span><strong>Ja</strong><br><span class=\"hint\">Nur wenn die Aufgabe tatsächlich erfüllt ist, nicht nur behauptet wird.</span></span></label>
            <label class=\"choice\"><input type=\"radio\" name=\"manual_task_successful\" value=\"false\"> <span><strong>Nein</strong><br><span class=\"hint\">Wenn die Lösung unvollständig, falsch oder nicht ausreichend verifiziert ist.</span></span></label>
          </div>
        </div>
        <div class=\"field\">
          <label class=\"title\">2. Welche primären MAST-Fehlermodi sind direkt belegt?</label>
          <div id=\"mode-choices\" class=\"choices\"></div>
        </div>
        <div class=\"field\">
          <label class=\"title\" for=\"manual-summary\">3. Kurze Evidenz-Zusammenfassung</label>
          <textarea id=\"manual-summary\" placeholder=\"1-3 Sätze: Welche Textstellen im Output oder Trace begründen Ihre Entscheidung?\"></textarea>
        </div>
        <div class=\"field\">
          <label class=\"title\" for=\"notes\">4. Optionale Notizen</label>
          <textarea id=\"notes\" placeholder=\"Optional: Unsicherheiten, offene Punkte, spätere Adjudication-Hinweise\"></textarea>
        </div>
      </div>
      <div class=\"panel\">
        <div class=\"toolbar\">
          <button id=\"prev-item\">Vorheriges Item</button>
          <button id=\"next-item\" class=\"primary\">Nächstes Item</button>
        </div>
        <p class=\"hint\">Pflicht für vollständige Zeile: Erfolg gesetzt, mindestens ein MAST-Mode gewählt, Summary nicht leer.</p>
      </div>
    </main>
  </div>
  <script>
    const payload = {json_payload};
    const storageKey = `phase-c-review-form::${{payload.reviewerTag}}::${{payload.sourceCsv}}`;
    let currentIndex = 0;
    let state = loadState();

    function defaultState() {{
      const items = {{}};
      for (const item of payload.items) {{
        items[item.annotation_item_id] = {{
          manual_task_successful: item.manual_task_successful || "",
          manual_primary_failure_modes: item.manual_primary_failure_modes ? item.manual_primary_failure_modes.split(';').map(v => v.trim()).filter(Boolean) : [],
          manual_summary: item.manual_summary || "",
          reviewer_id: item.reviewer_id || "",
          notes: item.notes || "",
        }};
      }}
      return {{ reviewer_id: "", items }};
    }}

    function loadState() {{
      try {{
        const raw = localStorage.getItem(storageKey);
        if (!raw) return defaultState();
        const parsed = JSON.parse(raw);
        return mergeState(defaultState(), parsed);
      }} catch (error) {{
        console.warn(error);
        return defaultState();
      }}
    }}

    function mergeState(base, incoming) {{
      const merged = structuredClone(base);
      if (incoming && typeof incoming.reviewer_id === 'string') merged.reviewer_id = incoming.reviewer_id;
      if (incoming && incoming.items) {{
        for (const [key, value] of Object.entries(incoming.items)) {{
          if (!merged.items[key]) continue;
          merged.items[key] = {{ ...merged.items[key], ...value }};
          if (!Array.isArray(merged.items[key].manual_primary_failure_modes)) {{
            merged.items[key].manual_primary_failure_modes = [];
          }}
        }}
      }}
      return merged;
    }}

    function saveState() {{
      localStorage.setItem(storageKey, JSON.stringify(state));
      updateProgress();
      renderItemList();
    }}

    function currentItem() {{
      return payload.items[currentIndex];
    }}

    function currentEntry() {{
      return state.items[currentItem().annotation_item_id];
    }}

    function isComplete(entry) {{
      return entry.manual_task_successful && entry.manual_primary_failure_modes.length > 0 && entry.manual_summary.trim();
    }}

    function updateProgress() {{
      const complete = Object.values(state.items).filter(isComplete).length;
      const total = payload.items.length;
      const pct = total ? (complete / total) * 100 : 0;
      document.getElementById('progress-fill').style.width = `${{pct}}%`;
      document.getElementById('progress-text').textContent = `${{complete}} / ${{total}} vollständig annotiert`;
    }}

    function renderItemList() {{
      const root = document.getElementById('item-list');
      root.innerHTML = '';
      payload.items.forEach((item, index) => {{
        const button = document.createElement('button');
        const entry = state.items[item.annotation_item_id];
        button.className = 'item-button' + (index === currentIndex ? ' active' : '') + (isComplete(entry) ? ' complete' : '');
        button.innerHTML = `<strong>#${{item.sequence}} · ${{item.annotation_item_id}}</strong><span class=\"small\">${{item.framework}} · ${{item.benchmark.split('/').pop()}}</span>`;
        button.addEventListener('click', () => {{ currentIndex = index; render(); }});
        root.appendChild(button);
      }});
    }}

    function renderModes() {{
      const root = document.getElementById('mode-choices');
      root.innerHTML = '';
      const entry = currentEntry();
      for (const mode of payload.modes) {{
        const wrapper = document.createElement('label');
        wrapper.className = 'choice';
        const checked = entry.manual_primary_failure_modes.includes(mode.label) ? 'checked' : '';
        wrapper.innerHTML = `<input type=\"checkbox\" value=\"${{escapeHtml(mode.label)}}\" ${{checked}}><span><strong>${{escapeHtml(mode.label)}}</strong><div class=\"mode-def\">${{escapeHtml(mode.definition)}}</div></span>`;
        wrapper.querySelector('input').addEventListener('change', (event) => {{
          const set = new Set(currentEntry().manual_primary_failure_modes);
          if (event.target.checked) set.add(mode.label); else set.delete(mode.label);
          currentEntry().manual_primary_failure_modes = Array.from(set);
          saveState();
        }});
        root.appendChild(wrapper);
      }}
    }}

    function render() {{
      const item = currentItem();
      const entry = currentEntry();
      document.getElementById('item-title').textContent = `Item #${{item.sequence}} · ${{item.annotation_item_id}}`;
      document.getElementById('item-meta').innerHTML = `<span class=\"badge\">${{escapeHtml(item.framework)}}</span><span class=\"badge\">${{escapeHtml(item.benchmark.split('/').pop())}}</span><span class=\"badge\">fehlend: ${{escapeHtml(item.missing_fields)}}</span>`;
      document.getElementById('framework').textContent = item.framework;
      document.getElementById('benchmark').textContent = item.benchmark;
      document.getElementById('run-id').textContent = item.run_id;
      document.getElementById('trace-path').textContent = item.raw_log_path;
      document.getElementById('final-output').textContent = item.final_output || '[leer]';
      document.getElementById('trace-text').textContent = item.trace_text || '[leer]';
      document.getElementById('manual-summary').value = entry.manual_summary;
      document.getElementById('notes').value = entry.notes;
      document.getElementById('global-reviewer-id').value = state.reviewer_id || entry.reviewer_id || '';
      document.querySelectorAll('input[name="manual_task_successful"]').forEach((node) => {{
        node.checked = node.value === entry.manual_task_successful;
      }});
      renderModes();
      renderItemList();
      updateProgress();
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}

    function buildExportRows() {{
      return payload.items.map((item) => {{
        const entry = state.items[item.annotation_item_id];
        return {{
          annotation_item_id: item.annotation_item_id,
          framework: item.framework,
          benchmark: item.benchmark,
          run_index: item.run_index,
          run_id: item.run_id,
          raw_log_path: item.raw_log_path,
          success: item.success,
          final_output: item.final_output,
          manual_task_successful: entry.manual_task_successful || '',
          manual_primary_failure_modes: entry.manual_primary_failure_modes.join('; '),
          manual_summary: entry.manual_summary || '',
          reviewer_id: state.reviewer_id || entry.reviewer_id || '',
          adjudicated_task_successful: item.adjudicated_task_successful || '',
          adjudicated_primary_failure_modes: item.adjudicated_primary_failure_modes || '',
          adjudicated_summary: item.adjudicated_summary || '',
          notes: entry.notes || '',
        }};
      }});
    }}

    function csvEscape(value) {{
      const text = value == null ? '' : String(value);
      return '"' + text.replaceAll('"', '""') + '"';
    }}

    function download(filename, content, type) {{
      const blob = new Blob([content], {{ type }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    }}

    document.getElementById('manual-summary').addEventListener('input', (event) => {{
      currentEntry().manual_summary = event.target.value;
      saveState();
    }});
    document.getElementById('notes').addEventListener('input', (event) => {{
      currentEntry().notes = event.target.value;
      saveState();
    }});
    document.getElementById('global-reviewer-id').addEventListener('input', (event) => {{
      state.reviewer_id = event.target.value;
      saveState();
    }});
    document.querySelectorAll('input[name="manual_task_successful"]').forEach((node) => {{
      node.addEventListener('change', (event) => {{
        currentEntry().manual_task_successful = event.target.value;
        saveState();
      }});
    }});
    document.getElementById('prev-item').addEventListener('click', () => {{
      currentIndex = Math.max(0, currentIndex - 1);
      render();
    }});
    document.getElementById('next-item').addEventListener('click', () => {{
      currentIndex = Math.min(payload.items.length - 1, currentIndex + 1);
      render();
    }});
    document.getElementById('download-csv').addEventListener('click', () => {{
      const rows = buildExportRows();
      const header = payload.columns;
      const body = rows.map((row) => header.map((column) => csvEscape(row[column] ?? '')).join(','));
      download(`phase_c_${{payload.reviewerTag}}_annotations.csv`, [header.join(','), ...body].join('\n'), 'text/csv;charset=utf-8');
    }});
    document.getElementById('download-json').addEventListener('click', () => {{
      download(`phase_c_${{payload.reviewerTag}}_backup.json`, JSON.stringify(state, null, 2), 'application/json;charset=utf-8');
    }});
    document.getElementById('clear-local').addEventListener('click', () => {{
      if (!confirm('Lokalen Zwischenstand wirklich löschen?')) return;
      localStorage.removeItem(storageKey);
      state = defaultState();
      render();
    }});

    render();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a static HTML review form for independent Phase C reviewers.")
    parser.add_argument("--reviewer-csv", type=Path, required=True, help="Reviewer CSV to annotate.")
    parser.add_argument("--worklist-csv", type=Path, required=True, help="Worklist CSV with annotation order.")
    parser.add_argument("--reviewer-tag", type=str, required=True, help="Reviewer key used in the worklist, e.g. reviewer1.")
    parser.add_argument("--output-html", type=Path, required=True, help="Target HTML file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reviewer_csv = _resolve_repo_path(args.reviewer_csv)
    worklist_csv = _resolve_repo_path(args.worklist_csv)
    output_html = _resolve_repo_path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    columns, items = _load_items(reviewer_csv, worklist_csv, args.reviewer_tag)
    output_html.write_text(
        _render_html(
            reviewer_tag=args.reviewer_tag,
            source_csv=reviewer_csv,
            columns=columns,
            items=items,
        ),
        encoding="utf-8",
    )
    print(f"Review form written to: {output_html}")
    print(f"Items embedded: {len(items)}")


if __name__ == "__main__":
    main()
