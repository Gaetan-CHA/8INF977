#!/usr/bin/env python3
"""
Runner multi-modèles A→B (API-only, Azure OpenAI)
- Pairing: pairs | random | self
- Prompts: auto (via banque YAML)
- Dossier: RUN_YYYYMMDD_HHMM_n°XXX
- Configuration par fichier: configs/runner_config_azure.yaml

Dépendances (runtime):
- pip install openai pyyaml keyring
- configs/models_azure.yaml       : modèles (deployments) Azure
- configs/prompts.yaml            : banque de prompts (id, text)
- configs/runner_config_azure.yaml: paramètres globaux
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
import inspect
from datetime import datetime as _dt
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Iterator, List, Dict, Any
from openai import AzureOpenAI

# ====== Chemins ======
ROOT = Path(__file__).resolve().parents[2]
CFG_API = ROOT / "configs" / "models_azure.yaml"
CFG_PROMPTS = ROOT / "configs" / "prompts.yaml"
CFG_RUNNER = ROOT / "configs" / "runner_config_azure.yaml"
RUNS_DIR = ROOT / "runs"

# ====== Debug global ======
DEBUG = False  # toggle via --debug ou RUNNER_DEBUG=1

# ====== Utils ======

def _is_payment_error(err: Exception) -> bool:
    s = str(err).lower()
    return ("402" in s) or ("payment required" in s)

# --- DEBUG pretty (amélioré: timestamp + fichier:ligne + trimming) ---
def _trim_strings(obj, maxlen=300):
    if isinstance(obj, str):
        return obj if len(obj) <= maxlen else obj[:maxlen] + f"… ({len(obj)} chars)"
    if isinstance(obj, dict):
        return {k: _trim_strings(v, maxlen) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_trim_strings(v, maxlen) for v in obj)
    return obj

def _dprint(title: str, payload=None) -> None:
    if not DEBUG:
        return
    outer = inspect.getouterframes(inspect.currentframe(), 2)[1]
    fname = Path(outer.filename).name
    lineno = outer.lineno
    ts = _dt.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[DBG {ts} {fname}:{lineno}] {title}")
    if payload is not None:
        try:
            clean = _trim_strings(payload)
            print(json.dumps(clean, ensure_ascii=False, indent=2))
        except Exception:
            print(str(payload))

def _trace(msg: str):
    if not DEBUG:
        return
    outer = inspect.stack()[1]
    print(f"[DBG {_dt.now().strftime('%H:%M:%S.%f')[:-3]} {Path(outer.filename).name}:{outer.lineno}] {msg}")

def _load_yaml(path: Path) -> Any:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# ====== Token Azure ======
def _get_azure_key(deployment: str) -> str:
    # 1) keyring: service="aoai", username="<deployment>"
    try:
        import keyring  # type: ignore
        k = keyring.get_password("aoai", deployment)
        if k:
            _dprint("Azure key from keyring", {"deployment": deployment, "value": k[:6] + "…"})
            return k
    except Exception:
        pass
    # 2) env var spécifique: AZURE_OPENAI_KEY__<DEPLOYMENT_UPPER>
    env_name = f"AZURE_OPENAI_KEY__{deployment.upper().replace('-', '_')}"
    k = os.getenv(env_name)
    if k:
        _dprint("Azure key from env", {"var": env_name, "value": k[:6] + "…"})
        return k
    # 3) env var globale (ultime fallback)
    k = os.getenv("AZURE_OPENAI_KEY", "")
    if k:
        _dprint("Azure key from env", {"var": "AZURE_OPENAI_KEY", "value": k[:6] + "…"})
    return k

def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "", str(name).split("/")[-1])

# ====== YAML → modèles ======
def _extract_models(cfg: dict) -> list[dict]:
    """
    Attend un dict avec:
      azure: { endpoint, api_version, default_model }
      models: [ { model_name, deployment, ... } ]
    Retourne une liste de modèles normalisés.
    """
    models = []
    seq = (cfg or {}).get("models", []) if isinstance(cfg, dict) else []
    for m in seq:
        if not isinstance(m, dict):
            continue
        name = m.get("model_name") or m.get("id")
        dep  = m.get("deployment") or name
        if not name or not dep:
            continue
        models.append({
            "model_name": name,
            "deployment": dep,
            "temperature": float(m.get("temperature", 1.0)),
            "max_completion_tokens": int(m.get("max_completion_tokens", 1024)),
            "top_p": float(m.get("top_p", 1.0)),
            "frequency_penalty": float(m.get("frequency_penalty", 0.0)),
            "presence_penalty": float(m.get("presence_penalty", 0.0)),
            "api_version": m.get("api_version"),
            "endpoint": m.get("endpoint"),  # optionnel par modèle
        })
    return models

def _azure_globals(cfg: dict) -> dict:
    a = (cfg or {}).get("azure", {}) if isinstance(cfg, dict) else {}
    return {
        "endpoint": a.get("endpoint", ""),
        "api_version": a.get("api_version", "2024-12-01-preview"),
        "default_model": a.get("default_model", ""),
    }

# ====== Config runner ======
def _load_runner_defaults() -> dict:
    cfg = _load_yaml(CFG_RUNNER)
    d = (cfg.get("defaults") or {}) if isinstance(cfg, dict) else {}
    d.setdefault("interactive", False)
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
    """Retourne une liste de {id: int, text: str}. Si vide, fournit 3 prompts par défaut."""
    data = _load_yaml(path) or {}
    items: List[Dict[str, Any]] = []
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

# ====== Azure Client Wrapper ======
class AzureClientAPI:
    """
    Adaptateur simple pour garder A.generate(text) -> str
    Utilise chat.completions avec un seul message user.
    """
    def __init__(self, *, endpoint: str, api_key: str, api_version: str,
                 deployment: str, temperature: float = 1.0,
                 max_completion_tokens: int = 512, top_p: float = 1.0,
                 frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
                 model_name: str | None = None):
        self.model_name = model_name or deployment
        self.deployment = deployment
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        self._client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )

    def generate(self, user_text: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.deployment,  # IMPORTANT: nom du déploiement Azure
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_text},
            ],
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            return ""

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

def dialogue(prompts: list[str], A: AzureClientAPI, B: AzureClientAPI, run_dir: Path, tag: str) -> None:
    """
    Chaque prompt suit ce schéma :
      1. A répond au prompt utilisateur.
      2. B répond à la réponse de A (avec le contexte complet).
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_a = _sanitize(getattr(A, "model_name", "A"))
    name_b = _sanitize(getattr(B, "model_name", "B"))
    out_file = run_dir / f"{tag}_{ts}_{name_a}_{name_b}.jsonl"

    meta = {
        "timestamp": ts, "mode": "api",
        "a_model": name_a,
        "b_model": name_b,
        "prompts_count": len(prompts), "tag": tag,
    }

    with open(out_file, "w", encoding="utf-8") as w:
        w.write(json.dumps({"event": "start", **meta}, ensure_ascii=False) + "\n")

        for i, user in enumerate(prompts, 1):
            _dprint("Turn input", {"turn": i, "user": user[:120]})

            try:
                # 1) A répond au prompt initial
                t0 = time.time()
                a_reply = A.generate(user)
                ta = (time.time() - t0) * 1000.0

                # 2) B répond à la réponse de A, avec le contexte complet
                t0 = time.time()
                convo_context = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": a_reply},
                    {"role": "user", "content": "Réponds à ce message précédent."}
                ]
                resp_b = B._client.chat.completions.create(
                    model=B.deployment,
                    messages=convo_context,
                    max_completion_tokens=B.max_completion_tokens,
                    temperature=B.temperature,
                    top_p=B.top_p,
                    frequency_penalty=B.frequency_penalty,
                    presence_penalty=B.presence_penalty,
                )
                b_reply = resp_b.choices[0].message.content or ""
                tb = (time.time() - t0) * 1000.0

            except Exception as e:
                if _is_payment_error(e):
                    print(f"[STOP] Crédit insuffisant pour ce modèle. Paire ignorée: {name_a} → {name_b}")
                    return
                print(f"[ERR] turn {i} ({name_a}→{name_b}): {e}")
                return

            record = {
                "turn": i,
                "user_prompt": user,
                "a_output": a_reply,
                "b_input_context": [user, a_reply],
                "b_output": b_reply,
                "latency_ms": round(ta + tb, 1),
                "a_latency_ms": round(ta, 1),
                "b_latency_ms": round(tb, 1),
            }
            w.write(json.dumps(record, ensure_ascii=False) + "\n")
            w.flush()

            if not _load_runner_defaults().get("quiet", True):
                print(f"[turn {i}] {round(ta+tb,1)} ms — {name_a} → {name_b}")

    if not _load_runner_defaults().get("quiet", True):
        print(f"Run stored: {out_file}")

# ====== CLI ======
def parse_args():
    p = argparse.ArgumentParser(description="Runner multi-modèles A→B (API-only, Azure OpenAI)")
    p.add_argument("mode", nargs="?", default="api", choices=["api"], help="Source des modèles")

    # Listing
    p.add_argument("--list", action="store_true", help="Lister tous les modèles et quitter")

    # Génération
    p.add_argument("--max-new-tokens", type=int, default=200, help="Nombre de tokens de complétion (map vers max_completion_tokens)")
    p.add_argument("--temperature", type=float, default=0.7)

    # Prompts
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
    p.add_argument("--debug", action="store_true", help="Activer la verbosité maximale")

    return p.parse_args()

# ====== Exécution AUTO via config (aucun argument) ======
def _auto_run_from_config():
    defaults = _load_runner_defaults()
    global DEBUG
    DEBUG = bool(defaults.get("debug", False) or os.getenv("RUNNER_DEBUG") == "1")

    cfg = _load_yaml(CFG_API)
    models = _extract_models(cfg)
    if not models:
        raise SystemExit(f"{CFG_API.as_posix()}: aucun modèle valide trouvé.")

    az = _azure_globals(cfg)
    endpoint_global = az["endpoint"]
    api_version = az["api_version"]
    if not endpoint_global:
        raise SystemExit("Endpoint Azure manquant (azure.endpoint).")

    if int(defaults.get("seed", 0)) != 0:
        random.seed(int(defaults["seed"]))

    # --- Clients (clé PAR déploiement, dans la boucle) ---
    clients: dict[str, AzureClientAPI] = {}
    for m in models:
        endpoint_m = m.get("endpoint") or endpoint_global
        api_key = _get_azure_key(m["deployment"])
        if not api_key:
            raise SystemExit(
                f"Clé manquante pour le déploiement '{m['deployment']}'. "
                f"Ajoute-la via: python src/protocol/manage_aoai_keys.py ensure"
            )
        clients[m["deployment"]] = AzureClientAPI(
            endpoint=endpoint_m,
            api_key=api_key,
            api_version=m.get("api_version") or api_version,
            deployment=m["deployment"],
            temperature=float(m.get("temperature", 1.0)),
            max_completion_tokens=int(m.get("max_completion_tokens", 1024)),
            top_p=float(m.get("top_p", 1.0)),
            frequency_penalty=float(m.get("frequency_penalty", 0.0)),
            presence_penalty=float(m.get("presence_penalty", 0.0)),
            model_name=m["model_name"],
        )
        _dprint("Client built", {"deployment": m["deployment"], "model": m["model_name"]})

    # Prompts
    prompt_bank = load_prompt_bank(Path(defaults.get("prompts_file", str(CFG_PROMPTS))))
    ids = [int(x) for x in defaults.get("prompt_ids", [])] or None
    prompts = select_prompts(prompt_bank, ids)

    # Dossier
    out_root = Path(defaults.get("runs_dir", str(RUNS_DIR))); out_root.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_dir(out_root)

    # Pairings
    pairing = str(defaults.get("pairing", "pairs"))
    random_k = int(defaults.get("random_k", 10)) if pairing == "random" else 0
    tag = f"api_{pairing}"
    model_ids = list(clients.keys())
    for a_id, b_id in pairing_generator(model_ids, pairing, random_k if random_k > 0 else None):
        try:
            dialogue(prompts, clients[a_id], clients[b_id], run_dir, tag)
        except Exception as e:
            print(f"[WARN] Dialogue échoué pour {a_id} → {b_id}: {e}")

# ====== Main ======
def main():
    # Si aucun argument CLI -> run auto via fichiers de config
    if len(sys.argv) == 1:
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

    # Charger config et modèles (Azure)
    cfg_api = _load_yaml(CFG_API)
    models = _extract_models(cfg_api)
    if not models:
        raise SystemExit(f"{CFG_API.as_posix()}: aucun modèle valide trouvé.")

    az = _azure_globals(cfg_api)
    endpoint = az.get("endpoint") or ""
    api_version_global = az.get("api_version") or "2024-12-01-preview"
    if not endpoint:
        raise SystemExit("Endpoint Azure manquant dans models_azure.yaml (azure.endpoint).")

    # Inventaire
    if not args.quiet or args.list:
        print("Modèles disponibles (provider=azure):")
        for i, m in enumerate(models):
            print(f"  [{i}] {m['deployment']} ({m['model_name']})")
        if args.list:
            return

    if args.pairing == "pairs" and len(models) < 2:
        raise SystemExit("pairing=pairs requiert au moins 2 modèles.")

    # Clients Azure
    clients: dict[str, AzureClientAPI] = {}
    for m in models:
        api_key = _get_azure_key(m["deployment"])
        if not api_key:
            raise SystemExit("AZURE_OPENAI_KEY manquant (clé fournie par ton service externe).")
        clients[m["deployment"]] = AzureClientAPI(
            endpoint=endpoint,
            api_key=api_key,
            api_version=m.get("api_version") or api_version_global,
            deployment=m["deployment"],
            temperature=float(args.temperature),
            max_completion_tokens=int(args.max_new_tokens),
            top_p=float(m.get("top_p", 1.0)),
            frequency_penalty=float(m.get("frequency_penalty", 0.0)),
            presence_penalty=float(m.get("presence_penalty", 0.0)),
            model_name=m["model_name"],
        )
        _dprint("Client built", {"deployment": m["deployment"], "model": m["model_name"]})

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
    model_ids = list(clients.keys())

    for a_id, b_id in pairing_generator(model_ids, args.pairing, k_rand):
        try:
            dialogue(prompts, clients[a_id], clients[b_id], run_dir, tag)
        except Exception as e:
            print(f"[WARN] Dialogue échoué pour {a_id} → {b_id}: {e}")

    if not args.quiet:
        print(f"Dossier: {run_dir}")

if __name__ == "__main__":
    main()
