#!/usr/bin/env python3
"""Lightweight browser for Melder output databases.

Zero external dependencies — uses only the Python standard library.
Point it at a results.db and browse views in your web browser.

Usage:
    python3 scripts/serve-views.py results.db
    python3 scripts/serve-views.py results.db --port 9090

Then open http://localhost:8787 in your browser.
"""

import argparse
import html
import os
import sqlite3
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_PORT = 8787
PAGE_SIZE = 100

# ---------------------------------------------------------------------------
# HTML templates (inline, no external files)
# ---------------------------------------------------------------------------

STYLE = """
body { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
       margin: 0; padding: 0; background: #faf7f2; color: #2c2a28; }
header { background: #2c2a28; color: #faf7f2; padding: 16px 24px; }
header h1 { margin: 0; font-size: 20px; font-weight: 600; }
header .db-path { color: #9e9790; font-size: 13px; margin-top: 4px; }
nav { background: #f5f0e8; border-bottom: 1px solid #e5dfd6; padding: 12px 24px;
      display: flex; flex-wrap: wrap; gap: 8px; }
nav a { text-decoration: none; color: #2c2a28; background: #fff;
        border: 1px solid #e5dfd6; border-radius: 6px; padding: 6px 14px;
        font-size: 13px; transition: all 0.15s; }
nav a:hover { border-color: #d94f30; color: #d94f30; }
nav a.active { background: #d94f30; color: #fff; border-color: #d94f30; }
main { padding: 24px; max-width: 100%; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px;
        background: #fff; border-radius: 8px; overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
th { background: #f5f0e8; text-align: left; padding: 10px 14px;
     font-weight: 600; border-bottom: 2px solid #e5dfd6;
     position: sticky; top: 0; }
td { padding: 8px 14px; border-bottom: 1px solid #eeebe5; }
tr:hover td { background: #fdf9f3; }
.info { padding: 12px 24px; color: #6b6560; font-size: 13px;
        display: flex; justify-content: space-between; align-items: center; }
.info a { color: #d94f30; text-decoration: none; margin-left: 12px; }
.info a:hover { text-decoration: underline; }
.badge { display: inline-block; background: #e5dfd6; border-radius: 10px;
         padding: 2px 8px; font-size: 11px; color: #6b6560; margin-left: 6px; }
.empty { color: #9e9790; font-style: italic; padding: 40px; text-align: center; }
.score { font-variant-numeric: tabular-nums; }
.type-match { color: #2d8b55; font-weight: 500; }
.type-review { color: #d4a843; font-weight: 500; }
.type-candidate { color: #2a7b9b; font-weight: 500; }
.type-broken { color: #c93b3b; font-weight: 500; }
"""


def page_index(db_path, views, tables, meta):
    meta_html = ""
    if meta:
        rows = "".join(
            f"<tr><td style='font-weight:500'>{html.escape(k)}</td>"
            f"<td>{html.escape(v)}</td></tr>"
            for k, v in meta
        )
        meta_html = f"""
        <div style="margin-bottom:24px">
          <h3 style="margin:0 0 8px 0;font-size:15px">Run metadata</h3>
          <table style="width:auto"><tbody>{rows}</tbody></table>
        </div>"""

    view_links = "".join(
        f'<a href="/view/{html.escape(v[0])}">{html.escape(v[0])}'
        f'<span class="badge">{v[1]}</span></a>'
        for v in views
    )
    table_links = "".join(
        f'<a href="/table/{html.escape(t[0])}">{html.escape(t[0])}'
        f'<span class="badge">{t[1]}</span></a>'
        for t in tables
    )

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Melder — {html.escape(os.path.basename(db_path))}</title>
<style>{STYLE}</style></head><body>
<header><h1>Melder Output Browser</h1>
<div class="db-path">{html.escape(db_path)}</div></header>
<main>
{meta_html}
<h3 style="margin:0 0 8px 0;font-size:15px">Views</h3>
<nav>{view_links}</nav>
<h3 style="margin:16px 0 8px 0;font-size:15px">Tables</h3>
<nav>{table_links}</nav>
</main></body></html>"""


def page_data(db_path, name, kind, columns, rows, offset, total):
    col_headers = "".join(f"<th>{html.escape(c)}</th>" for c in columns)

    def fmt_cell(col, val):
        if val is None:
            return '<td style="color:#ccc">NULL</td>'
        s = html.escape(str(val))
        if col == "relationship_type":
            cls = f"type-{val}" if val in ("match", "review", "candidate", "broken") else ""
            return f'<td class="{cls}">{s}</td>'
        if col in ("score", "composite_score", "field_score", "avg_score",
                    "min_score", "max_score"):
            return f'<td class="score">{s}</td>'
        return f"<td>{s}</td>"

    body = "".join(
        "<tr>" + "".join(fmt_cell(columns[i], v) for i, v in enumerate(row)) + "</tr>"
        for row in rows
    )

    if not rows:
        body = f'<tr><td colspan="{len(columns)}" class="empty">No rows</td></tr>'

    prev_link = ""
    next_link = ""
    if offset > 0:
        prev_off = max(0, offset - PAGE_SIZE)
        prev_link = f'<a href="/{kind}/{name}?offset={prev_off}">&larr; Previous</a>'
    if offset + PAGE_SIZE < total:
        next_off = offset + PAGE_SIZE
        next_link = f'<a href="/{kind}/{name}?offset={next_off}">Next &rarr;</a>'

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{html.escape(name)} — Melder</title>
<style>{STYLE}</style></head><body>
<header><h1>Melder Output Browser</h1>
<div class="db-path">{html.escape(db_path)}</div></header>
<nav><a href="/">&larr; Index</a>
<a class="active">{html.escape(name)}</a></nav>
<div class="info">
<span>Showing {offset+1}–{min(offset+PAGE_SIZE, total)} of {total} rows</span>
<span>{prev_link}{next_link}</span>
</div>
<main><table><thead><tr>{col_headers}</tr></thead>
<tbody>{body}</tbody></table></main></body></html>"""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    db_path = None

    def log_message(self, format, *args):
        pass  # Silence per-request logging

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA query_only = ON")

            if path == "" or path == "/":
                self._serve_index(conn)
            elif path.startswith("/view/"):
                name = path[6:]
                self._serve_data(conn, name, "view", params)
            elif path.startswith("/table/"):
                name = path[7:]
                self._serve_data(conn, name, "table", params)
            else:
                self._send(404, "Not found")

            conn.close()
        except Exception as e:
            self._send(500, f"Error: {e}")

    def _serve_index(self, conn):
        views = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name"
        ).fetchall()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

        view_counts = []
        for (name,) in views:
            try:
                cnt = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            except Exception:
                cnt = "?"
            view_counts.append((name, cnt))

        table_counts = []
        for (name,) in tables:
            try:
                cnt = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            except Exception:
                cnt = "?"
            table_counts.append((name, cnt))

        meta = []
        try:
            meta = conn.execute("SELECT key, value FROM metadata ORDER BY rowid").fetchall()
        except Exception:
            pass

        body = page_index(self.db_path, view_counts, table_counts, meta)
        self._send(200, body)

    def _serve_data(self, conn, name, kind, params):
        check = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name=? AND type=?",
            (name, kind)
        ).fetchone()
        if not check:
            self._send(404, f"{kind} '{name}' not found")
            return

        offset = int(params.get("offset", [0])[0])
        total = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]

        cursor = conn.execute(
            f'SELECT * FROM "{name}" LIMIT {PAGE_SIZE} OFFSET {offset}'
        )
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        body = page_data(self.db_path, name, kind, columns, rows, offset, total)
        self._send(200, body)

    def _send(self, code, body):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(
        description="Browse a Melder output database in your web browser."
    )
    parser.add_argument("db", help="Path to the SQLite output database (results.db)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Port to listen on (default {DEFAULT_PORT})")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        sys.exit(1)

    Handler.db_path = os.path.abspath(args.db)

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Serving {args.db} at {url}")
    print("Press Ctrl-C to stop.\n")

    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
