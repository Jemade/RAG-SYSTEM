import json
import os
import secrets
import sqlite3
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse


def create_app(root="var"):
    app = FastAPI(title="RAG System · Trace Inspector")
    db_path = Path(root) / "traces.sqlite3"
    token = os.getenv("RAG_INSPECTOR_TOKEN")

    def authorize(authorization: str | None = Header(default=None)):
        if token is None:
            return
        scheme, _, supplied = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(supplied, token):
            raise HTTPException(status_code=401, detail="Invalid inspector token")

    def read(sql, parameters=()):
        if not db_path.exists():
            return []
        with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True) as db:
            return db.execute(sql, parameters).fetchall()

    @app.get("/api/traces", dependencies=[Depends(authorize)])
    def traces():
        return [
            {"id": row[0], "created": row[1], **json.loads(row[2])}
            for row in read(
                "SELECT id,created,payload FROM traces ORDER BY created DESC,rowid DESC LIMIT 100"
            )
        ]

    @app.get("/api/traces/{trace_id}", dependencies=[Depends(authorize)])
    def trace(trace_id: str):
        rows = read("SELECT payload FROM traces WHERE id=?", (trace_id,))
        if not rows:
            raise HTTPException(404, "Trace not found")
        return json.loads(rows[0][0])

    @app.get("/", response_class=HTMLResponse)
    def home():
        return HTML

    return app


HTML = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RAG System — Trace Inspector</title>
<style>body{margin:0;background:#f5f6f8;color:#17202b;font:15px system-ui}header{background:#17202b;color:white;padding:24px 5%}header p{color:#bfc9d5}main{max-width:1200px;margin:32px auto;padding:0 24px}h1{font-size:24px}button{background:white;border:1px solid #ccd3dd;padding:12px;text-align:left;cursor:pointer;border-radius:4px;color:#17202b}#list{display:grid;gap:8px}section{background:white;border:1px solid #dce0e6;padding:24px;margin-bottom:20px;border-radius:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px monospace}label{display:block;margin-bottom:10px}input{width:95%;padding:12px;border:1px solid #b5bdc9}#detail{display:none}.meta{color:#586474;font-size:13px}</style>
<header><strong>RAG SYSTEM</strong><h1>Trace Inspector</h1><p>Queries, retrieval scores and cited evidence in one place.</p></header><main><section><label for="search">Filter queries</label><input id="search" placeholder="Search the latest 100 traces"><p id="summary" class="meta"></p><div id="list"></div></section><section id="detail"><h2>Trace details</h2><p>Scores use different scales. Citation integrity checks evidence strings; it does not prove causal use.</p><pre id="payload"></pre></section></main>
<script>let traces=[];const list=document.getElementById('list');function render(){list.replaceChildren();const q=document.getElementById('search').value.toLowerCase();for(const t of traces.filter(t=>t.query.toLowerCase().includes(q))){const b=document.createElement('button');b.textContent=t.status.toUpperCase()+' · '+t.query+' · '+Math.round(t.timings_ms.total)+' ms';b.onclick=()=>{document.getElementById('detail').style.display='block';document.getElementById('payload').textContent=JSON.stringify(t,null,2)};list.append(b)}}fetch('/api/traces').then(r=>r.json()).then(v=>{traces=v;document.getElementById('summary').textContent=v.length+' recent traces · '+v.filter(t=>t.status==='error').length+' errors';render()}).catch(()=>{document.getElementById('summary').textContent='Could not load traces.'});document.getElementById('search').oninput=render;</script></html>"""
