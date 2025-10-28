#!/usr/bin/env python3
"""
Runner multi-modèles A→B
- Modes: api | local
- Génère des dialogues et écrit les traces dans un sous-dossier normalisé sous /runs/ :
  Run_YYYYMMDD_HHMM_n°XXX/...

Dépendances attendues côté projet:
- src/llm_io/hf_client_local.py : classe HFClientLocal(model_name, max_new_tokens, temperature)
- src/llm_io/hf_client_api.py   : classe HFClientAPI(model_name, token, max_new_tokens, temperature, endpoint_url=None)

Usage rapide
  python -m src.protocol.runner api --wizard
  python -m src.protocol.runner local --pairing self --include meta-llama/Meta-Llama-3-8B-Instruct
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
from pathlib import Path
from itertools import permutations
from typing import Iterator

# Clients
from src.llm_io.hf_client_local import HFClientLocal
from src.llm_io.hf_client_api import HFClientAPI

# ===== Debug global =====
DEBUG = False  # bascule via --debug ou RUNNER_DEBUG=1

# ===== Chemins =====
ROOT = Path(__file__).resolve().parents[2]
CFG_API_EXT       = ROOT / "configs" / "models_api_extended.yaml"
CFG_API_FALLBACK  = ROOT / "configs" / "models_api.yaml"
CFG_LOCAL         = ROOT / "configs" / "models_local.yaml"
RUNS_DIR          = ROOT / "runs"
PROMPT_BANK_DEFAULT = ROOT / "configs" / "prompt_bank.yaml"

# ================= Utils =================

def _mask(tok: str | None) -> str:
    if not tok:
        return ""
    return (tok[:6] + "…") if len(tok) > 8 else "***"

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

def _load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _get_token(cfg_hint: dict | None = None) -> str:
    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if tok:
        _dprint("ENV token detected", {"value": _mask(tok)})
        return tok
    try:
        import keyring  # optional
        tok = keyring.get_password("hf", "HF_TOKEN") or keyring.get_password("huggingface", "token")
        if tok:
            _dprint("Keyring token detected", {"value": _mask(tok)})
            return tok
    except Exception:
        pass
    if cfg_hint and isinstance(cfg_hint, dict):
        t = cfg_hint.get("hf_token", "")
        if t:
            _dprint("CFG token detected", {"value": _mask(t)})
        return t
    return ""

def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "", name.split("/")[-1])

# ============== Dossier de sortie ==============

def _make_run_dir(base_dir: Path) -> Path:
    """Crée un sous-dossier : Run_YYYYMMDD_HHMM_n°XXX sous base_dir.
    Incrémente XXX si le nom existe déjà.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    idx = 1
    while True:
        candidate = base_dir / f"Run_{ts}_n°{idx:03d}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            idx += 1

# ============== Chargement modèles ==============

def load_available_models(source: str) -> list[dict]:
    models: list[dict] = []
    if source == "api":
        cfg_path = CFG_API_EXT if CFG_API_EXT.exists() else (CFG_API_FALLBACK if CFG_API_FALLBACK.exists() else None)
        if not cfg_path:
            raise SystemExit("Aucune config API trouvée." 
                f"Attendu: - {CFG_API_EXT} - {CFG_API_FALLBACK}"
            )
        cfg = _load_yaml(cfg_path) or {}
        models = cfg.get("models") or []
        _dprint("Loaded API config", {"path": str(cfg_path), "count": len(models)})
    else:
        if not CFG_LOCAL.exists():
            raise SystemExit(f"Config locale introuvable: {CFG_LOCAL}")
        cfg = _load_yaml(CFG_LOCAL) or {}
        models = cfg.get("models") or []
        _dprint("Loaded LOCAL config", {"path": str(CFG_LOCAL), "count": len(models)})

    if not isinstance(models, list):
        raise SystemExit("Config invalide: clé 'models' absente ou non-liste.")
    out = [m for m in models if isinstance(m, dict) and m.get("id")]
    _dprint("Available models list", {"count": len(out)})
    return out

# ============== Clients & génération ==============

def make_client(model_id: str, mode: str, gen_params: dict | None = None, token: str | None = None, endpoint_url: str | None = None):
    gen_params = gen_params or {}
    max_new = int(gen_params.get("max_new_tokens", 200))
    temp    = float(gen_params.get("temperature", 0.7))
    if mode == "api":
        if not token:
            token = _get_token({})
        if not token:
            raise RuntimeError("Token manquant: keyring(HF/HF_TOKEN) ou env HF_TOKEN/HUGGINGFACEHUB_API_TOKEN ou cfg.hf_token.")
        _dprint("Make API client", {"model": model_id, "endpoint": endpoint_url or "<router>", "max_new_tokens": max_new, "temperature": temp, "token": _mask(token)})
        return HFClientAPI(model_id, token, max_new, temp, endpoint_url=endpoint_url)
    else:
        _dprint("Make LOCAL client", {"model": model_id, "max_new_tokens": max_new, "temperature": temp})
        return HFClientLocal(model_id, max_new, temp)

# ============== Prompts ==============

def load_prompt_bank(path: Path) -> list[str]:
    if not path.exists():
        return [
            "Présente-toi en une phrase.",
            "Explique brièvement comment fonctionne une blockchain.",
            "Donne 3 conseils pour sécuriser un mot de passe.",
        ]
    data = _load_yaml(path)
    if isinstance(data, dict) and isinstance(data.get("prompts"), list):
        return [str(p) for p in data["prompts"]]
    if isinstance(data, list):
        return [str(p) for p in data]
    return []

def collect_prompts(args: argparse.Namespace) -> list[str]:
    mode = args.prompt_mode
    if mode == "auto":
        bank = load_prompt_bank(Path(args.prompts_file) if args.prompts_file else PROMPT_BANK_DEFAULT)
        _dprint("Prompts auto", {"count": len(bank)})
        return bank
    if mode == "manual-bank":
        bank = load_prompt_bank(Path(args.prompts_file) if args.prompts_file else PROMPT_BANK_DEFAULT)
        if not bank:
            print("Banque vide. Bascule sur saisie manuelle.")
            mode = "manual-input"
        else:
            print("Sélection manuelle depuis la banque. Indiquer les index séparés par des virgules.")
            for i, p in enumerate(bank):
                print(f"[{i}] {p}")
            _dprint("Prompt bank loaded", {"count": len(bank)})
            sel = input("Index à utiliser (ex: 0,2,5) : ").strip()
            idx: list[int] = []
            if sel:
                for t in sel.split(','):
                    t = t.strip()
                    if not t:
                        continue
                    try:
                        idx.append(int(t))
                    except ValueError:
                        pass
            prompts = [bank[i] for i in idx if 0 <= i < len(bank)]
            if prompts:
                _dprint("Prompts manual-bank selection", {"indices": idx, "count": len(prompts)})
                return prompts
            print("Aucune sélection valide. Bascule sur saisie manuelle.")
            mode = "manual-input"
    if mode == "manual-input":
        print("Entrer vos prompts, ligne par ligne. Ligne vide pour terminer.")
        lines: list[str] = []
        while True:
            s = input("> ").rstrip("")
            if not s:
                break
            lines.append(s)
        if not lines:
            print("Aucun prompt fourni. Utilisation de la banque par défaut.")
            bank = load_prompt_bank(PROMPT_BANK_DEFAULT)
            _dprint("Prompts fallback default", {"count": len(bank)})
            return bank
        _dprint("Prompts manual-input", {"count": len(lines)})
        return lines
    raise ValueError(f"prompt_mode inconnu: {mode}")

# ============== Sélection & clients ==============

def select_model_ids(all_models: list[dict], includes: list[str] | None = None, exclude: list[str] | None = None) -> list[str]:
    ids = [m["id"] for m in all_models]
    if includes:
        inc: list[str] = []
        for pat in includes:
            pat = pat.strip()
            inc += [mid for mid in ids if (mid == pat or pat in mid)]
        ids = sorted(set(inc))
    if exclude:
        ex = set(exclude)
        ids = [mid for mid in ids if mid not in ex]
    return ids

def build_clients(model_ids: list[str], mode: str, token_hint: str | None = None,
                  max_new_tokens: int = 200, temperature: float = 0.7,
                  per_model_meta: dict[str, dict] | None = None, endpoint_url_global: str = ""):
    token = token_hint or _get_token({}) if mode == "api" else None
    _dprint("Build clients config", {
        "mode": mode,
        "count": len(model_ids),
        "endpoint_url_global": endpoint_url_global or None,
        "gen": {"max_new_tokens": max_new_tokens, "temperature": temperature}
    })
    clients: dict[str, object] = {}
    for mid in model_ids:
        meta = (per_model_meta or {}).get(mid, {})
        ep = endpoint_url_global or meta.get("endpoint_url", "")
        _dprint("Client meta", {"model": mid, "endpoint": ep or "<router>", "meta_keys": list(meta.keys())})
        clients[mid] = make_client(
            mid, mode,
            {"max_new_tokens": max_new_tokens, "temperature": temperature},
            token=token, endpoint_url=(ep or None)
        )
    return clients

# ============== Pairings ==============

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

# ============== Dialogue run ==============

def dialogue(prompts: list[str], A_client, B_client, mode_label: str, out_base_dir: Path = RUNS_DIR):
    out_base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_dir(out_base_dir)
    ts_full = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_a = _sanitize(getattr(A_client, "model_name", "A"))
    name_b = _sanitize(getattr(B_client, "model_name", "B"))

    # Fichiers de sortie
    out_file = run_dir / f"{mode_label.capitalize()}_{ts_full}_{name_a}_{name_b}.jsonl"
    run_info = run_dir / f"run_info_{ts_full}.json"

    # Métadonnées de run
    meta = {
        "timestamp": ts_full,
        "mode": mode_label,
        "a_model": getattr(A_client, "model_name", "A"),
        "b_model": getattr(B_client, "model_name", "B"),
        "prompts_count": len(prompts),
    }
    with open(run_info, "w", encoding="utf-8") as rf:
        json.dump(meta, rf, ensure_ascii=False, indent=2)

    with open(out_file, "w", encoding="utf-8") as w:
        # En-tête pour éviter un fichier vide
        w.write(json.dumps({"event": "start", **meta}) + "")
        w.flush()
        for i, user in enumerate(prompts, 1):
            _dprint("Turn input", {"turn": i, "user_prompt_head": user[:120], "len": len(user)})
            try:
                t0 = time.time(); a = A_client.generate(user); ta = (time.time() - t0) * 1000
            except Exception as e:
                import traceback
                err = {
                    "turn": i,
                    "error": True,
                    "stage": "A.generate",
                    "exception": type(e).__name__,
                    "message": str(e),
                }
                w.write(json.dumps(err, ensure_ascii=False) + ""); w.flush()
                _dprint("Exception A.generate", {"type": type(e).__name__, "msg": str(e), "trace": traceback.format_exc()})
                print(f"[ERR] A.generate: {e}")
                continue
            try:
                t0 = time.time(); b = B_client.generate(a);    tb = (time.time() - t0) * 1000
            except Exception as e:
                import traceback
                err = {
                    "turn": i,
                    "error": True,
                    "stage": "B.generate",
                    "exception": type(e).__name__,
                    "message": str(e),
                    "a_head": str(a)[:120],
                }
                w.write(json.dumps(err, ensure_ascii=False) + ""); w.flush()
                _dprint("Exception B.generate", {"type": type(e).__name__, "msg": str(e), "trace": traceback.format_exc(), "a_head": str(a)[:120]})
                print(f"[ERR] B.generate: {e}")
                continue
            row = {
                "turn": i,
                "mode": mode_label,
                "timestamp": ts_full,
                "user_prompt": user,
                "a_model": getattr(A_client, "model_name", "A"),
                "b_model": getattr(B_client, "model_name", "B"),
                "a_output": a,
                "b_output": b,
                "latency_ms": round(ta + tb, 1),
                "a_latency_ms": round(ta, 1),
                "b_latency_ms": round(tb, 1)
            }
            w.write(json.dumps(row, ensure_ascii=False) + ""); w.flush()
            _dprint("Turn output", {"turn": i, "a_head": str(a)[:120], "b_head": str(b)[:120], "latency_ms": row["latency_ms"]})
            print(f"[turn {i}] {row['latency_ms']} ms — {name_a} → {name_b}")
    print(f"Run stored in {out_file}")
    print(f"Dossier: {run_dir}")

# ============== Wizard ==============

def wizard(all_models: list[dict]):
    print("=== Mode pas-à-pas ===")
    pairing = (input("Pairing [pairs/random/self] (def=pairs): ").strip().lower() or "pairs")
    krand = 0
    if pairing == "random":
        try:
            krand = int(input("Nombre de paires aléatoires (def=10): ").strip() or 10)
        except ValueError:
            krand = 10

    src = input("Source des modèles [api/local] (def=api): ").strip().lower() or "api"

    print("Modèles disponibles:")
    for i, m in enumerate(all_models):
        ident = m.get("id"); ent = m.get("enterprise", "?"); pb = m.get("params_b", "?"); ctx = m.get("context_length", "?")
        print(f"  [{i}] id={ident} | enterprise={ent} | params_b={pb} | context={ctx}")

    ids_all = [m["id"] for m in all_models]
    if pairing == "self":
        sel = input("Choisis 1 index (A vs A), ou appuie Entrée pour tout: ").strip().lower()
        if not sel:
            selected_ids = ids_all
        else:
            try:
                idx = int(sel)
                selected_ids = [ids_all[idx]] if 0 <= idx < len(ids_all) else ids_all
            except ValueError:
                selected_ids = ids_all
    elif pairing == "pairs":
        sel = input("Choisis 2+ indices (ex: 0,1) pour toutes les combinaisons A!=B, vide=all: ").strip().lower()
        if not sel:
            selected_ids = ids_all
        else:
            idxs: list[int] = []
            for t in sel.split(','):
                t = t.strip()
                if not t:
                    continue
                try:
                    idxs.append(int(t))
                except ValueError:
                    pass
            selected_ids = [ids_all[i] for i in idxs if 0 <= i < len(ids_all)]
            if len(selected_ids) < 2:
                print("Moins de 2 modèles sélectionnés. Ajout automatique pour garantir les paires.")
                selected_ids = (selected_ids + ids_all)[:2]
    else:  # random
        sel = input("Sélection indices (ex: 0,2,5) pour l'échantillon aléatoire, vide=all: ").strip().lower()
        if not sel:
            selected_ids = ids_all
        else:
            idxs: list[int] = []
            for t in sel.split(','):
                t = t.strip()
                if not t:
                    continue
                try:
                    idxs.append(int(t))
                except ValueError:
                    pass
            selected_ids = [ids_all[i] for i in idxs if 0 <= i < len(ids_all)] or ids_all

    pmode = (input("Prompt mode [auto/manual-bank/manual-input] (def=auto): ").strip().lower() or "auto")
    pfile = input("Chemin banque de prompts (vide=defaut): ").strip()

    try:
        max_new = int(input("max_new_tokens (def=200): ").strip() or 200)
    except ValueError:
        max_new = 200
    try:
        temperature = float(input("temperature (def=0.7): ").strip() or 0.7)
    except ValueError:
        temperature = 0.7

    return {
        "mode": src,
        "selected_ids": selected_ids,
        "prompt_mode": pmode,
        "prompts_file": pfile,
        "pairing": pairing,
        "random_k": krand,
        "max_new_tokens": max_new,
        "temperature": temperature,
    }

# ============== CLI ==============

def parse_args():
    p = argparse.ArgumentParser(description="Runner multi-modèles pour dialogues A→B")
    p.add_argument("mode", nargs="?", default="api", choices=["api", "local"], help="Source des modèles")

    # Listing / filtrage
    p.add_argument("--list", action="store_true", help="Lister tous les modèles et quitter")
    p.add_argument("--include", type=str, default="", help="Motifs/ids à inclure, séparés par ,")
    p.add_argument("--exclude", type=str, default="", help="Ids à exclure, séparés par ,")

    # Génération
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)

    # Prompts
    p.add_argument("--prompt-mode", type=str, default="auto",
                   choices=["auto", "manual-bank", "manual-input"], help="auto | manuel via banque | manuel saisie")
    p.add_argument("--prompts-file", type=str, default="", help="YAML/TXT de banque de prompts")

    # Pairing
    p.add_argument("--pairing", type=str, default="pairs", choices=["pairs", "random", "self"], help="paires A!=B | aléatoire | A vs A")
    p.add_argument("--random-k", type=int, default=0, help="Nb de paires si pairing=random")

    # Exécution
    p.add_argument("--runs-dir", type=str, default=str(RUNS_DIR), help="Dossier racine pour Run_YYYYMMDD_HHMM_n°XXX (def=runs/)")
    p.add_argument("--quiet", action="store_true")

    # Divers
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wizard", action="store_true", help="Assistant interactif pas-à-pas")
    p.add_argument("--debug", action="store_true", help="Activer la verbosité maximale")

    # Endpoint API dédié (override global)
    p.add_argument("--endpoint-url", type=str, default="", help="Endpoint HF dédié (bypass router) pour tous les modèles")

    return p.parse_args()

# ============== Main ==============

def main():
    # Aide universelle personnalisée
    if any(a.lower() in {"help", "--help", "-h"} for a in sys.argv[1:]):
        print("Utilisation : python -m src.protocol.runner [api|local] [options]")
        print("Options principales :")
        print("  --list                      Lister tous les modèles et quitter")
        print("  --include a,b               Filtrer les modèles à inclure (motifs ou ids)")
        print("  --exclude a,b               Exclure certains modèles")
        print("  --prompt-mode MODE          'auto' | 'manual-bank' | 'manual-input'")
        print("  --pairing MODE              'pairs' | 'random' | 'self'")
        print("  --random-k N                Nombre de paires aléatoires si random")
        print("  --prompts-file chemin.yaml  Banque de prompts personnalisée")
        print("  --max-new-tokens N          Nombre max de tokens générés")
        print("  --temperature T             Température de génération")
        print("  --runs-dir dossier          Dossier racine (runs/) pour Run_YYYYMMDD_HHMM_n°XXX")
        print("  --quiet                     Réduit la sortie console")
        print("  --seed N                    Graine aléatoire")
        print("  --wizard                    Assistant interactif pas-à-pas")
        print("  --endpoint-url URL          Endpoint HF dédié pour bypass le router")
        return

    args = parse_args()
    # Debug
    global DEBUG
    DEBUG = bool(args.debug or os.getenv("RUNNER_DEBUG") == "1")
    _dprint("CLI args", vars(args))

    if args.seed:
        random.seed(args.seed)
        _dprint("Seed set", {"seed": args.seed})

    # Modèles
    all_models = load_available_models(args.mode)
    if not all_models:
        raise SystemExit("Aucun modèle défini dans la configuration.")
    _dprint("Models loaded", {"count": len(all_models)})

    model_meta = {m["id"]: m for m in all_models}

    # Wizard
    if args.wizard:
        cfg = wizard(all_models)
        args.mode = cfg["mode"]
        args.prompt_mode = cfg["prompt_mode"]
        args.prompts_file = cfg["prompts_file"]
        args.pairing = cfg["pairing"]
        args.random_k = cfg["random_k"]
        args.max_new_tokens = cfg["max_new_tokens"]
        args.temperature = cfg["temperature"]
        selected_ids = cfg["selected_ids"]
    else:
        if not args.quiet or args.list:
            print("Modèles disponibles (provider=huggingface):")
            _dprint("Inventory print", {"quiet": args.quiet, "list": args.list})
            for i, m in enumerate(all_models):
                ident = m.get("id"); ent = m.get("enterprise", "?"); pb = m.get("params_b", "?"); ctx = m.get("context_length", "?")
                print(f"  [{i}] id={ident} | enterprise={ent} | params_b={pb} | context={ctx}")
            if args.list:
                return
        include = [s for s in args.include.split(',') if s.strip()] if args.include else None
        exclude = [s for s in args.exclude.split(',') if s.strip()] if args.exclude else None
        selected_ids = select_model_ids(all_models, include, exclude)
        _dprint("Selected ids", {"ids": selected_ids})
        if args.pairing == "pairs" and len(selected_ids) < 2:
            raise SystemExit("pairing=pairs requiert au moins 2 modèles. Utilise --include ou --wizard.")

    # Clients
    clients = build_clients(
        selected_ids,
        args.mode,
        token_hint=None,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        per_model_meta=model_meta,
        endpoint_url_global=(args.endpoint_url or "")
    )
    _dprint("Clients built", {"count": len(clients), "keys": list(clients.keys())})

    # Prompts
    prompts = collect_prompts(args)
    _dprint("Prompts ready", {"count": len(prompts)})

    # Sorties
    runs_dir = Path(args.runs_dir)

    # Pairings
    k_rand = args.random_k if args.random_k > 0 else None
    _dprint("Pairing plan", {"pairing": args.pairing, "k_random": k_rand})
    for a_id, b_id in pairing_generator(selected_ids, args.pairing, k_rand):
        _dprint("Pair", {"A": a_id, "B": b_id})
        A = clients[a_id]
        B = clients[b_id]
        mode_label = f"{args.mode}_{args.pairing}"
        try:
            dialogue(prompts, A, B, mode_label, runs_dir)
        except Exception as e:
            import traceback
            _dprint("Dialogue exception", {"A": a_id, "B": b_id, "type": type(e).__name__, "msg": str(e), "trace": traceback.format_exc()})
            print(f"[WARN] Dialogue échoué pour {a_id} → {b_id}: {e}")
            if args.mode == "api":
                print("Astuce: passe un --endpoint-url vers un Inference Endpoint que tu gères, ou sélectionne un modèle servi publiquement.")

if __name__ == "__main__":
        main()