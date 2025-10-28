#!/usr/bin/env python3
"""
Runner multi-modèles A→B (API-only, Hugging Face)
- Pairing: pairs | random | self
- Prompts: auto | manual-bank | manual-input
- Dossier: RUN_YYYYMMDD_HHMM_n°XXX
- Choix au lancement: Config par fichier (runner_config.yaml) OU Config manuel.

Dépendances:
- src/llm_io/hf_client_api.py : HFClientAPI(model_name, token, max_new_tokens, temperature)
- configs/models_api.yaml     : modèles + hf_token
- configs/prompts.yaml        : banque de prompts (avec id, text)
- configs/runner_config.yaml  : paramètres globaux
"""
from __future__ import annotations

import os
import sys
import yaml
import json
import time
import re
import random
import argparse
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Iterator, List, Dict, Any

from src.llm_io.hf_client_api import HFClientAPI  # API only

# ====== Chemins ======
ROOT = Path(__file__).resolve().parents[2]
CFG_API = ROOT / "configs" / "models_api.yaml"
CFG_PROMPTS = ROOT / "configs" / "prompts.yaml"
CFG_RUNNER = ROOT / "configs" / "runner_config.yaml"
RUNS_DIR = ROOT / "runs"

# ====== Debug global ======
DEBUG = False  # toggle via --debug ou RUNNER_DEBUG=1

# ====== Utils ======

def _is_payment_error(err: Exception) -> bool:
    s = str(err).lower()
    return ("402" in s) or ("payment required" in s)


def _dprint(title: str, payload=None) -> None:
    if not DEBUG:
        return
    print(f"[DEBUG] {title}")
    if payload is None:
        return
    try:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception:
        print(str(payload))


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_HUB_TOKEN")

def _get_token(cfg: dict) -> str:
    for k in TOKEN_ENV_VARS:
        tok = os.getenv(k)
        if tok:
            _dprint("Token from env", {"var": k, "value": tok[:6] + "…"})
            return tok
    try:
        import keyring
        tok = keyring.get_password("hf", "HF_TOKEN")
        if tok:
            _dprint("Token from keyring", {"value": tok[:6] + "…"})
            return tok
    except Exception:
        pass
    tok = cfg.get("hf_token", "")
    if tok:
        _dprint("Token from config", {"value": tok[:6] + "…"})
    return tok


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "", str(name).split("/")[-1])

# ====== YAML → modèles ======

def _extract_models(cfg: dict) -> list[dict]:
    """Accepte:
      - model_X: { model_name|id: ..., max_new_tokens, temperature }
      - models: [ { id|model_name: ..., ... }, ... ]
    Normalise vers dicts: id, max_new_tokens, temperature
    """
    out = []
    # 1) model_1, model_2, ...
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and k.startswith("model_") and ("model_name" in v or "id" in v):
            try:
                n = int(k.split("_", 1)[1])
            except Exception:
                n = 10**9
            out.append((n, v))
    if out:
        out.sort(key=lambda x: x[0])
        seq = [v for _, v in out]
    else:
        seq = cfg.get("models", []) if isinstance(cfg, dict) else []
        if not isinstance(seq, list):
            seq = []

    norm = []
    for m in seq:
        if not isinstance(m, dict):
            continue
        mid = m.get("id") or m.get("model_name")
        if not mid:
            continue
        mx = int(m.get("max_new_tokens", 64))
        tp = float(m.get("temperature", 0.7))
        norm.append({
            "id": mid,
            "max_new_tokens": mx,
            "temperature": tp,
        })
    return norm

# ====== Config runner ======

def _load_runner_defaults() -> dict:
    cfg = _load_yaml(CFG_RUNNER)
    d = (cfg.get("defaults") or {}) if isinstance(cfg, dict) else {}
    # valeurs par défaut si absentes
    d.setdefault("interactive", True)
    d.setdefault("mode", "api")
    d.setdefault("pairing", "pairs")
    d.setdefault("prompt_mode", "auto")
    d.setdefault("random_k", 10)
    d.setdefault("max_new_tokens", 200)
    d.setdefault("temperature", 0.7)
    d.setdefault("runs_dir", str(RUNS_DIR))
    d.setdefault("prompts_file", str(CFG_PROMPTS))
    d.setdefault("quiet", True)   # réduit par défaut
    d.setdefault("seed", 0)
    d.setdefault("debug", False)  # debug désactivé par défaut
    d.setdefault("prompt_ids", [])
    return d

# ====== Prompts ======

def load_prompt_bank(path: Path) -> List[Dict[str, Any]]:
    """Retourne une liste de {id: int, text: str}.
    Si le YAML ne contient pas d'id, en génère séquentiellement.
    """
    data = _load_yaml(path) or {}
    items = []
    raw = []
    if isinstance(data, dict) and isinstance(data.get("prompts"), list):
        raw = data["prompts"]
    elif isinstance(data, list):
        raw = data
    if not raw:
        raw = [
            {"id": 1, "text": "Présente-toi en une phrase."},
            {"id": 2, "text": "Explique brièvement comment fonctionne une blockchain."},
            {"id": 3, "text": "Donne 3 conseils pour sécuriser un mot de passe."},
        ]
    # normalisation
    for i, p in enumerate(raw, 1):
        if isinstance(p, dict):
            pid = int(p.get("id", i))
            txt = str(p.get("text", "")).strip()
        else:
            pid = i
            txt = str(p).strip()
        if not txt:
            continue
        items.append({"id": pid, "text": txt})
    return items


def select_prompts(prompt_bank: List[Dict[str, Any]], ids: List[int] | None) -> List[str]:
    if not ids:
        return [p["text"] for p in prompt_bank]
    keep = {int(x) for x in ids}
    return [p["text"] for p in prompt_bank if int(p["id"]) in keep]

# ====== Pairings ======

def pairing_generator(model_ids: list[str], pairing: str, k_random: int | None = None) -> Iterator[tuple[str, str]]:
    if pairing == "pairs":
        for a, b in permutations(model_ids, 2):
            yield (a, b)
    elif pairing == "self":
        for a in model_ids:
            yield (a, a)
    elif pairing == "random":
        if not k_random:
            k_random = max(1, len(model_ids))
        for _ in range(k_random):
            a = random.choice(model_ids)
            b = random.choice(model_ids)
            yield (a, b)
    else:
        raise ValueError(f"pairing inconnu: {pairing}")

# ====== Dialogue ======

def _make_run_dir(base_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    idx = 1
    while True:
        d = base_dir / f"RUN_{ts}_n°{idx:03d}"
        try:
            d.mkdir(parents=True, exist_ok=False)
            return d
        except FileExistsError:
            idx += 1


def dialogue(prompts: list[str], A: HFClientAPI, B: HFClientAPI, run_dir: Path, tag: str) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_a = _sanitize(getattr(A, "model_name", "A"))
    name_b = _sanitize(getattr(B, "model_name", "B"))
    out_file = run_dir / f"{tag}_{ts}_{name_a}_{name_b}.jsonl"

    meta = {
        "timestamp": ts, "mode": "api",
        "a_model": getattr(A, "model_name", "A"),
        "b_model": getattr(B, "model_name", "B"),
        "prompts_count": len(prompts), "tag": tag,
    }
    with open(out_file, "w", encoding="utf-8") as w:
        w.write(json.dumps({"event": "start", **meta}, ensure_ascii=False) + "\n")
        for i, user in enumerate(prompts, 1):
            _dprint("Turn input", {"turn": i, "user": user[:120]})
            try:
                t0 = time.time(); a = A.generate(user); ta = (time.time() - t0) * 1000.0
                t0 = time.time(); b = B.generate(a);    tb = (time.time() - t0) * 1000.0
            except Exception as e:
                if _is_payment_error(e):
                    print(f"[STOP] Crédit insuffisant pour ce modèle. Paire ignorée: {name_a} → {name_b}")
                    return  # stop uniquement cette paire
                print(f"[ERR] turn {i} ({name_a}→{name_b}): {e}")
                return  # stop propre sur autre erreur
            w.write(json.dumps({
                "turn": i, "user_prompt": user,
                "a_output": a, "b_output": b,
                "latency_ms": round(ta + tb, 1),
                "a_latency_ms": round(ta, 1),
                "b_latency_ms": round(tb, 1),
            }, ensure_ascii=False) + "\n"); w.flush()
            if not _load_runner_defaults().get("quiet", True):
                print(f"[turn {i}] {round(ta+tb,1)} ms — {name_a} → {name_b}")
    if not _load_runner_defaults().get("quiet", True):
        print(f"Run stored: {out_file}")


# ====== CLI ======

def parse_args():
    p = argparse.ArgumentParser(description="Runner multi-modèles A→B (API-only)")
    p.add_argument("mode", nargs="?", default="api", choices=["api"], help="Source des modèles")

    # Listing
    p.add_argument("--list", action="store_true", help="Lister tous les modèles et quitter")

    # Génération
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)

    # Prompts
    p.add_argument("--prompt-mode", type=str, default="auto",
                   choices=["auto", "manual-bank", "manual-input"],
                   help="auto | manuel via banque | manuel saisie")
    p.add_argument("--prompts-file", type=str, default=str(CFG_PROMPTS), help="YAML/TXT de banque de prompts")
    p.add_argument("--prompt-ids", type=str, default="", help="IDs de prompts à exécuter, ex: 1,3,5")

    # Pairing
    p.add_argument("--pairing", type=str, default="pairs",
                   choices=["pairs", "random", "self"],
                   help="paires A!=B | aléatoire | A vs A")
    p.add_argument("--random-k", type=int, default=0, help="Nb de paires si pairing=random")

    # Exécution
    p.add_argument("--runs-dir", type=str, default=str(RUNS_DIR),
                   help="Dossier racine pour RUN_YYYYMMDD_HHMM_n°XXX (def=runs/)")
    p.add_argument("--quiet", action="store_true")

    # Divers
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wizard", action="store_true", help="Assistant interactif pas-à-pas")
    p.add_argument("--debug", action="store_true", help="Activer la verbosité maximale")

    return p.parse_args()

# ====== Exécution AUTO via config (aucun argument) ======

def _auto_run_from_config():
    defaults = _load_runner_defaults()
    global DEBUG
    DEBUG = bool(defaults.get("debug", False) or os.getenv("RUNNER_DEBUG") == "1")

    # Charger modèles + token
    cfg = _load_yaml(CFG_API)
    models = _extract_models(cfg)
    if not models:
        raise SystemExit(f"{CFG_API.as_posix()}: aucun modèle valide trouvé.")
    token = _get_token(cfg)
    if not token:
        raise SystemExit("HF_TOKEN manquant pour l’API Hugging Face.")

    # Sélection: tous les modèles
    selected_ids = [m["id"] for m in models]

    # Clients
    clients: dict[str, HFClientAPI] = {}
    for mid in selected_ids:
        clients[mid] = HFClientAPI(mid, token, int(defaults.get("max_new_tokens", 200)), float(defaults.get("temperature", 0.7)))
        _dprint("Client built", {"id": mid})

    # Prompts + sélection IDs
    prompt_bank = load_prompt_bank(Path(defaults.get("prompts_file", str(CFG_PROMPTS))))
    ids = [int(x) for x in defaults.get("prompt_ids", [])] or None
    prompts = select_prompts(prompt_bank, ids)

    # Dossier
    out_root = Path(defaults.get("runs_dir", str(RUNS_DIR)))
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_dir(out_root)

    # Pairings
    pairing = str(defaults.get("pairing", "pairs"))
    random_k = int(defaults.get("random_k", 10)) if pairing == "random" else 0
    tag = f"api_{pairing}"
    for a_id, b_id in pairing_generator(list(clients.keys()), pairing, random_k if random_k > 0 else None):
        try:
            dialogue(prompts, clients[a_id], clients[b_id], run_dir, tag)
        except Exception as e:
            print(f"[WARN] Dialogue échoué pour {a_id} → {b_id}: {e}")
    if not defaults.get("quiet", True):
        print(f"Dossier: {run_dir}")


def _manual_wizard_and_run():
    # inventaire
    cfg_api = _load_yaml(CFG_API)
    models = _extract_models(cfg_api)
    if not models:
        raise SystemExit(f"{CFG_API.as_posix()}: aucun modèle valide trouvé.")

    # choix mode
    mode = (input("Mode [api/local] (def=api): ").strip().lower() or "api")
    if mode != "api":
        print("Avertissement: seul 'api' est implémenté ici.")
        mode = "api"

    # pairing
    pairing = (input("Pairing [pairs/random/self] (def=pairs): ").strip().lower() or "pairs")
    random_k = 0
    if pairing == "random":
        t = input("random_k (def=10): ").strip()
        random_k = int(t) if t.isdigit() else 10

    # affichage des modèles (noms uniquement)
    print("Modèles disponibles (sélection):")
    for i, m in enumerate(models):
        print(f"  [{i}] {m['id']}")
    raw_idx = input("Indices modèles séparés par des virgules (vide=tous): ").strip()

    # prompts
    prompt_mode = (input("Prompt mode [auto/manual-bank/manual-input] (def=auto): ").strip().lower() or "auto")
    prompts_file = input(f"Chemin prompts (def={CFG_PROMPTS}): ").strip() or str(CFG_PROMPTS)

    # paramètres génération
    t = input("max_new_tokens (def=200): ").strip()
    max_new_tokens = int(t) if t.isdigit() else 200
    t = input("temperature (def=0.7): ").strip()
    try:
        temperature = float(t) if t else 0.7
    except ValueError:
        temperature = 0.7

    # dossier + flags
    runs_dir = input(f"Dossier runs (def={RUNS_DIR}): ").strip() or str(RUNS_DIR)
    quiet = (input("Quiet [Y/n]: ").strip().lower() in {"", "y", "yes"})  # quiet activé par défaut
    t = input("Seed (def=0 = non fixé): ").strip()
    seed = int(t) if t.isdigit() else 0
    debug = (input("Debug [y/N]: ").strip().lower() == "y")

    # sélection des modèles
    ids_all = [m["id"] for m in models]
    if raw_idx:
        idxs = []
        for tok in raw_idx.split(','):
            tok = tok.strip()
            if not tok:
                continue
            try:
                idxs.append(int(tok))
            except ValueError:
                pass
        selected_ids = [ids_all[i] for i in idxs if 0 <= i < len(ids_all)] or ids_all
    else:
        selected_ids = ids_all
    if pairing == "pairs" and len(selected_ids) < 2:
        raise SystemExit("pairing=pairs requiert au moins 2 modèles.")

    # token
    token = _get_token(cfg_api)
    if not token:
        raise SystemExit("HF_TOKEN manquant pour l’API Hugging Face.")

    # clients
    clients = {mid: HFClientAPI(mid, token, max_new_tokens, temperature) for mid in selected_ids}

    # prompts effectifs
    bank = load_prompt_bank(Path(prompts_file))
    prompt_ids = []
    if prompt_mode == "manual-bank":
        print("Prompts disponibles:")
        for p in bank:
            print(f"  [{p['id']}] {p['text']}")
        raw = input("IDs de prompts à exécuter (ex: 1,3,5) vide=tous: ").strip()
        prompt_ids = [int(x) for x in raw.split(",") if x.strip().isdigit()] if raw else []
    prompts = select_prompts(bank, prompt_ids or None)
    if prompt_mode == "manual-input":
        prompts = collect_prompts(argparse.Namespace(prompt_mode="manual-input", prompts_file=""))

    # dossier run
    out_root = Path(runs_dir); out_root.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_dir(out_root)

    # exécution
    tag = f"{mode}_{pairing}"
    k_rand = random_k if pairing == "random" and random_k > 0 else None
    for a_id, b_id in pairing_generator(list(clients.keys()), pairing, k_rand):
        dialogue(prompts, clients[a_id], clients[b_id], run_dir, tag)

    # commande équivalente affichée
    parts = [
        "python -m src.protocol.runner_v1-3", mode,
        f"--pairing {pairing}",
        f"--max-new-tokens {max_new_tokens}",
        f"--temperature {temperature}",
        f"--runs-dir \"{runs_dir}\"",
        f"--prompts-file \"{prompts_file}\"",
    ]
    if quiet: parts.append("--quiet")
    if debug: parts.append("--debug")
    if seed: parts += ["--seed", str(seed)]
    if prompt_ids: parts += ["--prompt-ids", ",".join(map(str, prompt_ids))]
    if pairing == "random" and random_k: parts += ["--random-k", str(random_k)]
    if prompt_mode: parts += ["--prompt-mode", prompt_mode]
    print("\nCommande équivalente:\n" + " ".join(parts))
    print(f"\nDossier: {run_dir}")


# ====== Main ======

def main():
    # Choix immédiat si aucun argument
    if len(sys.argv) == 1:
        print("Choix du mode de configuration:")
        print("  [1] Config par fichier (configs/runner_config.yaml)")
        print("  [2] Config manuel (assistant pas-à-pas)")
        choice = (input("Sélection (1/2, def=1): ").strip() or "1")
        if choice == "2":
            _manual_wizard_and_run()
            return
        _auto_run_from_config()
        return

    # Mode CLI classique (arguments)
    args = parse_args()

    # Debug
    global DEBUG
    DEBUG = bool(args.debug or os.getenv("RUNNER_DEBUG") == "1")
    _dprint("CLI args", vars(args))

    if args.seed:
        random.seed(args.seed)
        _dprint("Seed set", {"seed": args.seed})

    # Charger config et modèles
    cfg_api = _load_yaml(CFG_API)
    models = _extract_models(cfg_api)
    if not models:
        raise SystemExit(f"{CFG_API.as_posix()}: aucun modèle valide trouvé.")

    # Token
    token = _get_token(cfg_api)
    if not token:
        raise SystemExit("HF_TOKEN manquant pour l’API Hugging Face.")

    # Inventaire
    if not args.quiet or args.list:
        print("Modèles disponibles (provider=huggingface):")
        for i, m in enumerate(models):
            print(f"  [{i}] {m['id']}")
        if args.list:
            return

    # Sélection: tous les modèles (filtrage simple via indices manuel si besoin)
    selected_ids = [m["id"] for m in models]

    if args.pairing == "pairs" and len(selected_ids) < 2:
        raise SystemExit("pairing=pairs requiert au moins 2 modèles.")

    # Clients
    clients: dict[str, HFClientAPI] = {}
    for mid in selected_ids:
        clients[mid] = HFClientAPI(mid, token, int(args.max_new_tokens), float(args.temperature))
        _dprint("Client built", {"id": mid})

    # Prompts + sélection via --prompt-ids
    bank = load_prompt_bank(Path(args.prompts_file))
    pid_list = [int(x) for x in args.prompt_ids.split(',')] if getattr(args, "prompt_ids", "").strip() else None
    prompts = select_prompts(bank, pid_list)

    # Préparer dossier de run (RUN_*)
    out_root = Path(args.runs_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_dir(out_root)

    # Pairings et exécution
    tag = f"api_{args.pairing}"
    k_rand = args.random_k if args.random_k > 0 else None
    for a_id, b_id in pairing_generator(list(clients.keys()), args.pairing, k_rand):
        try:
            dialogue(prompts, clients[a_id], clients[b_id], run_dir, tag)
        except Exception as e:
            print(f"[WARN] Dialogue échoué pour {a_id} → {b_id}: {e}")

    if not args.quiet:
        print(f"Dossier: {run_dir}")


if __name__ == "__main__":
    main()
