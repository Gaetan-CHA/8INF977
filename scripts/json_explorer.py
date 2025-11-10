#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
json_explorer.py — Lecture JSON/JSONL + rapport multipage HTML/Markdown

Fonctions clés
- Lecture d'un .jsonl (une entrée par ligne) ou .json (liste/dict)
- Normalisation/aplatissement des champs imbriqués
- Conversion timestamp `ts` (epoch secondes) -> timezone locale
- Exports CSV (brut + synthèse)
- Graphiques matplotlib (sans thème personnalisé)
- Rapport MULTI-PAGE HTML:
    - index.html (navigation uniquement)
    - pages/overview.html
    - pages/models.html
    - pages/attacks.html
    - pages/defenses.html
    - pages/timeline.html
    - pages/raw.html
- Rapport Markdown (report.md) en bonus
- Dossier de sortie horodaté pour éviter les collisions

Usage (PowerShell) :
python "D:\\UQAC\\8INF977 - Sujet spécial en cybersécurité\\scripts\\json_explorer.py" \
  -i "D:\\UQAC\\8INF977 - Sujet spécial en cybersécurité\\runs\\ATTACKS_DEF_1762201128.jsonl" \
  -o "D:\\UQAC\\8INF977 - Sujet spécial en cybersécurité\\rapport" \
  --tz "America/Toronto"
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Union

import pandas as pd
import matplotlib.pyplot as plt

try:
    import pytz  # type: ignore
except Exception:
    pytz = None

# -------------------------------
# Helpers
# -------------------------------

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _now_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_lines(path: str) -> List[Any]:
    """Charge JSONL (une ligne = un objet) ou JSON standard."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        # Heuristique simple : si plusieurs lignes JSON -> JSONL
        if "\n" in head and head.strip().startswith("{") and not head.strip().endswith("}"):
            # Probable JSONL
            data = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    # Dernière chance: certaines lignes peuvent contenir des virgules finales etc.
                    try:
                        data.append(json.loads(line.rstrip(",")))
                    except Exception as e:
                        print(f"[WARN] Ligne ignorée (JSON invalide): {line[:120]}... — {e}")
            return data
        else:
            obj = json.load(f)
            if isinstance(obj, list):
                return obj
            elif isinstance(obj, dict):
                return [obj]
            else:
                raise ValueError("JSON inconnu: attendu dict ou list pour le .json")


def _to_tz(dt_utc: datetime, tz_name: str | None) -> datetime:
    if tz_name and pytz is not None:
        tz = pytz.timezone(tz_name)
        return dt_utc.astimezone(tz)
    return dt_utc.astimezone()


def _epoch_to_dt(epoch_sec: Union[int, float, str], tz_name: str | None) -> str:
    try:
        val = float(epoch_sec)
    except Exception:
        return ""
    dt = datetime.fromtimestamp(val, tz=timezone.utc)
    dt_local = _to_tz(dt, tz_name)
    return dt_local.strftime("%Y-%m-%d %H:%M:%S %Z")


def _flatten(obj: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Aplatissement récursif des dicts imbriqués; listes indexées."""
    items = []
    for k, v in obj.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            if all(isinstance(x, (str, int, float, bool, type(None))) for x in v):
                items.append((new_key, ", ".join(map(lambda x: "" if x is None else str(x), v))))
            else:
                # Liste d'objets: indexation
                for i, el in enumerate(v):
                    if isinstance(el, dict):
                        items.extend(_flatten(el, f"{new_key}[{i}]", sep=sep).items())
                    else:
                        items.append((f"{new_key}[{i}]", json.dumps(el, ensure_ascii=False)))
        else:
            items.append((new_key, v))
    return dict(items)


# -------------------------------
# Rapport HTML
# -------------------------------

BASE_CSS = """
:root { --bg:#0f1115; --card:#151823; --ink:#eaeef6; --muted:#9aa4b2; --acc:#7c94ff; --ok:#30c48d; --warn:#f4a261; --bad:#ef476f; }
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,Segoe UI,Roboto,Ubuntu}
.wrapper{max-width:none;width:100%;margin:0 auto;padding:16px}
.table{width:100%;border-collapse:collapse;table-layout:auto;word-break:break-word}
.card{background:var(--card);border:1px solid #1f2433;border-radius:14px;
      padding:18px;margin:14px 0;box-shadow:0 8px 24px rgba(0,0,0,.25);
      width:100%;overflow-x:auto}
.h1{font-weight:700;font-size:22px;margin:0 0 6px}
.h2{font-weight:600;font-size:16px;margin:0 0 6px;color:var(--muted)}
.btns{display:flex;flex-wrap:wrap;gap:12px}
.btn{appearance:none;border:1px solid #2a3350;background:#1a1f2f;color:var(--ink);padding:10px 14px;border-radius:12px;text-decoration:none}
.btn:hover{border-color:var(--acc)}
.badge{display:inline-block;padding:4px 8px;border-radius:999px;background:#1a1f2f;border:1px solid #2a3350;color:var(--muted);font-size:12px}
.table{width:100%;border-collapse:collapse}
.table th,.table td{border-bottom:1px solid #262b3d;padding:8px 10px;text-align:left}
.meta{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.footer{color:var(--muted);margin-top:18px;font-size:12px}
"""

INDEX_HTML = """
<!doctype html><html lang="fr"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Rapport LLM — Navigation</title>
<style>{css}</style></head>
<body><div class="wrapper">
  <div class="card">
    <div class="h1">Rapport d'évaluation LLM — Navigation</div>
    <div class="meta">Ce rapport est découpé en pages thématiques. Choisis une section :</div>
  </div>
  <div class="card">
    <div class="btns">
      <a class="btn" href="pages/overview.html">Vue d'ensemble</a>
      <a class="btn" href="pages/models.html">Modèles</a>
      <a class="btn" href="pages/attacks.html">Attaques</a>
      <a class="btn" href="pages/defenses.html">Défenses</a>
      <a class="btn" href="pages/timeline.html">Chronologie</a>
      <a class="btn" href="pages/raw.html">Données brutes</a>
    </div>
  </div>
  <div class="footer">Généré par json_explorer.py</div>
</div></body></html>
"""

PAGE_TPL = """
<!doctype html><html lang=fr><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<style>{css}</style></head>
<body><div class="wrapper">
  <div class="card"><a class="btn" href="../index.html">⟵ Retour navigation</a></div>
  <div class="card">
    <div class="h1">{title}</div>
    <div class="meta">{subtitle}</div>
  </div>
  {content}
  <div class="footer">Généré {now}</div>
</div></body></html>
"""

# -------------------------------
# Génération graphiques & tables
# -------------------------------

def _save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=144, bbox_inches="tight")
    plt.close()


def _counts(series: pd.Series) -> pd.DataFrame:
    c = series.fillna("(NA)").value_counts().reset_index()
    c.columns = [series.name or "valeur", "count"]
    return c


# -------------------------------
# Pipeline
# -------------------------------

def build_report(input_path: str, out_root: str, tz_name: str | None = None) -> str:
    # 1) Charger
    records = _load_lines(input_path)
    if not records:
        raise SystemExit("Aucune ligne valide dans le fichier JSON/JSONL")

    # 2) Aplatir & DataFrame
    flat = []
    for r in records:
        d = _flatten(r)
        # Normalisation des champs standards si présents
        if "ts" in r:
            d["ts_local"] = _epoch_to_dt(r["ts"], tz_name)
        if "model" in r:
            d["model"] = r.get("model")
        if "attack" in r:
            d["attack"] = r.get("attack")
        if "defense" in r:
            d["defense"] = r.get("defense")
        if "event" in r:
            d["event"] = r.get("event")
        flat.append(d)

    df = pd.DataFrame(flat)

    # 3) Dossier horodaté
    run_dir = _ensure_dir(os.path.join(out_root, f"report_{_now_slug()}"))
    pages_dir = _ensure_dir(os.path.join(run_dir, "pages"))
    assets = _ensure_dir(os.path.join(run_dir, "assets"))

    # 4) Exports CSV
    df.to_csv(os.path.join(run_dir, "data_flat.csv"), index=False, encoding="utf-8-sig")
    # Synthèses
    synth = {
        "by_event": _counts(df.get("event", pd.Series(dtype=str))),
        "by_model": _counts(df.get("model", pd.Series(dtype=str))),
        "by_attack": _counts(df.get("attack", pd.Series(dtype=str))),
        "by_defense": _counts(df.get("defense", pd.Series(dtype=str))),
    }
    for name, dfx in synth.items():
        dfx.to_csv(os.path.join(run_dir, f"synth_{name}.csv"), index=False, encoding="utf-8-sig")

    # 5) Graphiques
    figs = {}
    if "model" in df.columns:
        _counts(df["model"]).plot(kind="bar", x=df["model"].name, y="count", legend=False, title="Répartition des modèles")
        figs["models"] = os.path.join(assets, "models.png")
        _save_fig(figs["models"]) 
    if "attack" in df.columns:
        _counts(df["attack"]).plot(kind="bar", x=df["attack"].name, y="count", legend=False, title="Répartition des attaques")
        figs["attacks"] = os.path.join(assets, "attacks.png")
        _save_fig(figs["attacks"]) 
    if "defense" in df.columns:
        _counts(df["defense"]).plot(kind="bar", x=df["defense"].name, y="count", legend=False, title="Répartition des défenses")
        figs["defenses"] = os.path.join(assets, "defenses.png")
        _save_fig(figs["defenses"]) 

    # Timeline si ts_local
    if "ts_local" in df.columns:
        ts_counts = df.groupby("ts_local").size().reset_index(name="count")
        ts_counts["idx"] = range(len(ts_counts))
        ts_counts.plot(kind="line", x="idx", y="count", legend=False, title="Chronologie (densité par enregistrement)")
        figs["timeline"] = os.path.join(assets, "timeline.png")
        _save_fig(figs["timeline"]) 

    # 6) Pages HTML
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # index.html — uniquement la nav
    with open(os.path.join(run_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(INDEX_HTML.format(css=BASE_CSS))

    # overview
    ov_parts = []
    ov_parts.append("<div class=card><div class=h2>Compteurs</div><div class=grid>")
    def _badge(label: str, val: int) -> str:
        return f"<div class=card><div class=h2>{label}</div><div class=h1>{val}</div></div>"
    ov_parts.append(_badge("Lignes", len(df)))
    ov_parts.append(_badge("Modèles uniques", df.get("model", pd.Series()).nunique()))
    ov_parts.append(_badge("Attaques uniques", df.get("attack", pd.Series()).nunique()))
    ov_parts.append(_badge("Défenses uniques", df.get("defense", pd.Series()).nunique()))
    ov_parts.append("</div></div>")

    if "models" in figs:
        ov_parts.append(f"<div class=card><div class=h2>Répartition modèles</div><img src='../assets/models.png' alt='models' /></div>")
    if "attacks" in figs:
        ov_parts.append(f"<div class=card><div class=h2>Répartition attaques</div><img src='../assets/attacks.png' alt='attacks' /></div>")
    if "defenses" in figs:
        ov_parts.append(f"<div class=card><div class=h2>Répartition défenses</div><img src='../assets/defenses.png' alt='defenses' /></div>")

    overview_html = PAGE_TPL.format(
        title="Vue d'ensemble",
        subtitle="Synthèse rapide des éléments évalués",
        css=BASE_CSS,
        content="\n".join(ov_parts),
        now=now,
    )
    with open(os.path.join(pages_dir, "overview.html"), "w", encoding="utf-8") as f:
        f.write(overview_html)

    # models page
    models_html_parts = []
    if "model" in df.columns:
        tbl = _counts(df["model"]).to_html(index=False, classes="table")
        models_html_parts.append(f"<div class=card><div class=h2>Comptes par modèle</div>{tbl}</div>")
        if "models" in figs:
            models_html_parts.append(f"<div class=card><img src='../assets/models.png' alt='models' /></div>")
    else:
        models_html_parts.append("<div class=card>Colonne 'model' absente.</div>")

    with open(os.path.join(pages_dir, "models.html"), "w", encoding="utf-8") as f:
        f.write(PAGE_TPL.format(title="Modèles", subtitle="Répartition et détails par modèle", css=BASE_CSS, content="\n".join(models_html_parts), now=now))

    # attacks page
    attacks_html_parts = []
    if "attack" in df.columns:
        tbl = _counts(df["attack"]).to_html(index=False, classes="table")
        attacks_html_parts.append(f"<div class=card><div class=h2>Comptes par attaque</div>{tbl}</div>")
        if "attacks" in figs:
            attacks_html_parts.append(f"<div class=card><img src='../assets/attacks.png' alt='attacks' /></div>")
    else:
        attacks_html_parts.append("<div class=card>Colonne 'attack' absente.</div>")

    with open(os.path.join(pages_dir, "attacks.html"), "w", encoding="utf-8") as f:
        f.write(PAGE_TPL.format(title="Attaques", subtitle="Répartition et détails par attaque", css=BASE_CSS, content="\n".join(attacks_html_parts), now=now))

    # defenses page
    defenses_html_parts = []
    if "defense" in df.columns:
        tbl = _counts(df["defense"]).to_html(index=False, classes="table")
        defenses_html_parts.append(f"<div class=card><div class=h2>Comptes par défense</div>{tbl}</div>")
        if "defenses" in figs:
            defenses_html_parts.append(f"<div class=card><img src='../assets/defenses.png' alt='defenses' /></div>")
    else:
        defenses_html_parts.append("<div class=card>Colonne 'defense' absente.</div>")

    with open(os.path.join(pages_dir, "defenses.html"), "w", encoding="utf-8") as f:
        f.write(PAGE_TPL.format(title="Défenses", subtitle="Répartition et détails par défense", css=BASE_CSS, content="\n".join(defenses_html_parts), now=now))

    # timeline page
    timeline_parts = []
    if "timeline" in figs:
        timeline_parts.append(f"<div class=card><div class=h2>Chronologie</div><img src='../assets/timeline.png' alt='timeline' /></div>")
    if "ts_local" in df.columns:
        sample = df[["ts_local", "model", "attack", "defense", "event"]].head(200)
        timeline_parts.append("<div class=card><div class=h2>Échantillon (200 premières lignes)</div>" + sample.to_html(index=False, classes="table") + "</div>")
    else:
        timeline_parts.append("<div class=card>Aucun champ temporel ('ts' ou 'ts_local') détecté.</div>")

    with open(os.path.join(pages_dir, "timeline.html"), "w", encoding="utf-8") as f:
        f.write(PAGE_TPL.format(title="Chronologie", subtitle="Densité et échantillon temporel", css=BASE_CSS, content="\n".join(timeline_parts), now=now))

    # raw page
    raw_cols = [c for c in df.columns if True]
    raw_html = df.head(500).to_html(index=False, classes="table")  # limiter l'affichage
    with open(os.path.join(pages_dir, "raw.html"), "w", encoding="utf-8") as f:
        f.write(PAGE_TPL.format(title="Données brutes", subtitle=f"Aperçu des 500 premières lignes, {len(df.columns)} colonnes", css=BASE_CSS, content=f"<div class=card>{raw_html}</div>", now=now))

    # 7) Bonus: Markdown synthétique
    md_lines = [
        "# Rapport LLM — Synthèse",
        "",
        f"- Lignes: **{len(df)}**",
        f"- Modèles uniques: **{df.get('model', pd.Series()).nunique()}**",
        f"- Attaques uniques: **{df.get('attack', pd.Series()).nunique()}**",
        f"- Défenses uniques: **{df.get('defense', pd.Series()).nunique()}**",
        "",
        "## Fichiers",
        "- `data_flat.csv` — données aplaties",
        "- `synth_*` — comptages par dimension",
        "- `pages/` — pages HTML thématiques",
    ]
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return run_dir


# -------------------------------
# CLI
# -------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JSON/JSONL → Rapport multipage + CSV + PNG")
    parser.add_argument("-i", "--input", required=True, help="Chemin du .jsonl ou .json")
    parser.add_argument("-o", "--output", required=True, help="Dossier racine de sortie (un sous-dossier horodaté sera créé)")
    parser.add_argument("--tz", default=None, help="Nom de timezone IANA (ex: America/Toronto). Si absent, timezone locale.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = os.path.abspath(args.input)
    out_root = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        raise SystemExit(f"Fichier introuvable: {input_path}")

    _ensure_dir(out_root)
    run_dir = build_report(input_path, out_root, tz_name=args.tz)

    print("\nRapport généré :")
    print(run_dir)
    print("Ouvre index.html pour naviguer entre les pages.")


if __name__ == "__main__":
    main()
