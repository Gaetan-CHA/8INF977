#!/usr/bin/env python3
"""
Test script for models_api.yaml on Hugging Face platform.

Features
- Lists locally cached HF models (installed) from ~/.cache/huggingface/hub.
- Prints short procedure to install dependencies and download models.
- Checks access to Hugging Face API token (env, keyring, token file).
- Loads models from models_api.yaml and verifies availability:
    * Installed locally (cache present)
    * Remote status via HfApi: available / gated / not_found / error / no_api
    * Repo type: public vs gated (nature du dépôt)
    * Access test: ok / no / unknown (capacité effective à lister/télécharger)
- Outputs a summary table to stdout and saves CSV + Markdown files.

Usage
  python test_models.py --yaml models_api.yaml --out out

Requires
  pip install pyyaml huggingface_hub tabulate
  # optional but useful
  pip install transformers safetensors accelerate
"""
from __future__ import annotations
import os
import sys
import csv
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple


# Optional keyring import
try:
    import keyring as _keyring  # type: ignore
except Exception:
    _keyring = None

# Optional hub imports (compatible 0.36.0 → >=1.0)
try:
    from huggingface_hub import HfApi, snapshot_download  # OK pour 0.36.0
    try:
        from huggingface_hub import HfHubHTTPError        # >=0.39
    except Exception:
        from huggingface_hub.utils import HfHubHTTPError  # <=0.38 (dont 0.36.0)
except Exception as e:
    print("IMPORT_ERROR huggingface_hub:", repr(e))
    HfApi = None
    snapshot_download = None
    class HfHubHTTPError(Exception):  # fallback local
        pass


if HfApi is None:
    print("ERROR: 'huggingface_hub' absent dans l’environnement courant.")
    print("Fix: python -m pip install -U huggingface_hub keyring tabulate pyyaml")
    sys.exit(1)


# Optional pretty table
try:
    from tabulate import tabulate  # type: ignore
except Exception:
    tabulate = None  # type: ignore

DEFAULT_YAML = "models_api.yaml"

@dataclass
class ModelSpec:
    enterprise: str
    model_id: str
    params_b: Optional[float]
    context_length: Optional[int]

@dataclass
class ModelCheck:
    installed: bool
    remote: str       # available | gated | not_found | error | no_api
    repo_type: str    # public | gated | unknown
    access: str       # ok | no | unknown
    note: str         # details/tags/reason


# --------------------------- token helpers ---------------------------

def get_token_local() -> str:
    """Return HF token from env, keyring (hf/HF_TOKEN or huggingface/token), or ~/.huggingface/token file."""
    # Environment variables
    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if tok:
        return tok.strip()

    # Keyring (custom service used in your project)
    if _keyring is not None:
        try:
            tok = _keyring.get_password("hf", "HF_TOKEN")
            if tok:
                return tok.strip()
        except Exception:
            pass
        # Keyring (official hub defaults)
        try:
            tok = _keyring.get_password("huggingface", "token")
            if tok:
                return tok.strip()
        except Exception:
            pass

    # Token file fallback
    token_file = Path.home() / ".huggingface" / "token"
    if token_file.exists():
        try:
            return token_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    return ""


# --------------------------- other helpers ---------------------------

def print_install_procedure():
    print("== Installation / setup quickstart ==")
    print("1) Dépendances Python :")
    print("   python -m pip install -U pyyaml huggingface_hub keyring tabulate transformers safetensors accelerate")
    print("2) Auth Hugging Face (recommandé pour les modèles gated) :")
    print("   # Option A: login CLI et stockage dans le keyring Windows")
    print("   python -m huggingface_hub.cli login")
    print("   python -m huggingface_hub.cli whoami")
    print("   # Option B: variable d’environnement pour la session courante")
    print('   $env:HUGGINGFACEHUB_API_TOKEN="hf_xxx"')
    print("   # Option C: fichier %USERPROFILE%/.huggingface/token contenant le jeton")
    print("3) Télécharger un modèle dans le cache local :")
    print('   python -c "from huggingface_hub import snapshot_download; snapshot_download(\'meta-llama/Meta-Llama-3-8B-Instruct\')"')


def print_no_token_help():
    print("\n== Aucun token détecté : procédure rapide ==")
    print("A) Installer/mettre à jour les libs :")
    print("   python -m pip install -U huggingface_hub keyring")
    print("B) Ajouter le token dans le keyring custom du projet → (hf, HF_TOKEN) :")
    print("   # PowerShell :")
    print("   python -c \"import keyring; keyring.set_password('hf','HF_TOKEN','hf_xxx')\"")
    print("C) Copier aussi le token vers le keyring officiel Hugging Face (facultatif mais utile) :")
    print('   python -c "import keyring; t=keyring.get_password(\'hf\',\'HF_TOKEN\'); '
          'keyring.set_password(\'huggingface\',\'token\',t) if t else print(\'no token\')"')
    print("D) Vérifier :")
    print('   python -c "import os,keyring; '
          'print(\'env:\', bool(os.getenv(\'HUGGINGFACEHUB_API_TOKEN\') or os.getenv(\'HF_TOKEN\'))); '
          'print(\'kr_hf:\', bool(keyring.get_password(\'hf\',\'HF_TOKEN\')) if keyring else False); '
          'print(\'kr_official:\', bool(keyring.get_password(\'huggingface\',\'token\')) if keyring else False)"')


def list_local_models_from_cache() -> List[str]:
    """Return repo ids inferred from HF cache structure ~/.cache/huggingface/hub/models--*"""
    cache = Path(os.environ.get("HF_HOME", Path.home()/".cache"/"huggingface"))/"hub"
    found: List[str] = []
    if not cache.exists():
        return found
    for p in cache.glob("models--*"):
        # format: models--org--repo
        name = p.name[len("models--"):]
        repo_id = name.replace("--", "/")
        found.append(repo_id)
    return sorted(set(found))


def read_models_yaml(path: Path) -> List[ModelSpec]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = []
    for m in data.get("models", []):
        models.append(ModelSpec(
            enterprise=m.get("enterprise") or m.get("publisher") or "",
            model_id=m.get("id"),
            params_b=m.get("params_b"),
            context_length=m.get("context_length"),
        ))
    return models


def have_api() -> Tuple[bool, Optional[str]]:
    tok = get_token_local()
    return (bool(tok), tok if tok else None)


def is_installed_locally(model_id: str) -> bool:
    # Try a quick local-only snapshot check if huggingface_hub is available
    if snapshot_download is None:
        # Fallback: check cache folder name
        return model_id in set(list_local_models_from_cache())
    try:
        snapshot_download(
            model_id,
            local_files_only=True,
            allow_patterns=["*.bin", "*.safetensors", "*config.json", "*tokenizer*"],
            tqdm_class=None,
        )
        return True
    except Exception:
        return False


# --------------------------- remote checks ---------------------------

def _model_info(api: HfApi, model_id: str, token: Optional[str]):
    return api.model_info(model_id, token=token)

def get_repo_type(api: HfApi, model_id: str, token: Optional[str]) -> Tuple[str, str]:
    """
    Returns (repo_type, note)
    repo_type: 'gated' | 'public' | 'unknown'
    note: 'gated' or tags excerpt
    """
    try:
        info = _model_info(api, model_id, token)
        gated = bool(getattr(info, "gated", False) or getattr(info, "private", False))
        tags = getattr(info, "tags", []) or []
        return ("gated" if gated else "public", "gated" if gated else ",".join(tags[:5]))
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 404:
            return ("unknown", "model not found")
        if status in (401, 403):
            # On ne sait pas si le repo est public ou gated, mais on n'a pas d'accès
            return ("unknown", "auth required or forbidden")
        return ("unknown", f"http {status} {e}")
    except Exception as e:
        return ("unknown", str(e))

def check_access(api: HfApi, model_id: str, token: Optional[str]) -> Tuple[str, str]:
    """
    Returns (access, reason)
    access: 'ok' | 'no' | 'unknown'
    - ok: list_repo_files or a tiny snapshot works
    - no: explicit 401/403
    - unknown: other errors
    """
    # Essai 1: lister quelques fichiers
    try:
        files = api.list_repo_files(model_id, token=token)
        # Si on obtient une liste (même vide, improbable), on considère ok
        return ("ok", "")
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403):
            return ("no", "auth required or forbidden")
        if status == 404:
            return ("unknown", "model not found")
        # autre erreur HTTP
        # on tente un snapshot local minimal pour trancher
    except Exception:
        pass

    # Essai 2: snapshot_download ultra restreint
    try:
        snapshot_download(
            model_id,
            allow_patterns=["*config.json"],
            local_dir_use_symlinks=False,
            tqdm_class=None,
            token=token,
        )
        return ("ok", "")
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403):
            return ("no", "auth required or forbidden")
        if status == 404:
            return ("unknown", "model not found")
        return ("unknown", f"http {status} {e}")
    except Exception as e:
        return ("unknown", str(e))


def check_remote(model_id: str, token: Optional[str]) -> Tuple[str, str, str, str]:
    """
    Returns (remote_status, repo_type, access, note)
    remote_status: available | gated | not_found | error | no_api
    repo_type: public | gated | unknown
    access: ok | no | unknown
    note: tags or reason
    """
    if HfApi is None:
        return ("no_api", "unknown", "unknown", "huggingface_hub not installed")
    api = HfApi()
    # D'abord récupérer le type de repo
    repo_type, note = get_repo_type(api, model_id, token)
    # Puis tester l'accès
    access, reason = check_access(api, model_id, token)

    # Construire remote_status
    if repo_type == "unknown":
        # déduire depuis reason
        if reason == "model not found" or note == "model not found":
            remote = "not_found"
        elif "auth required" in (reason or note):
            remote = "gated"
        else:
            # on ne sait pas si c'est public/gated mais l'hôte répond
            remote = "error" if reason else "error"
    else:
        if repo_type == "gated":
            # repo est de type gated; si access ok => on le marque available
            remote = "available" if access == "ok" else "gated"
        else:
            # repo public: s'il répond, available
            remote = "available" if access in ("ok", "unknown") else "error"

    final_note = note if access == "ok" or not reason else (reason if note == "gated" else f"{note}; {reason}")
    return (remote, repo_type, access, final_note)


def check_model(model: ModelSpec, token: Optional[str]) -> ModelCheck:
    installed = is_installed_locally(model.model_id)
    remote_status, repo_type, access, note = check_remote(model.model_id, token)
    return ModelCheck(installed=installed, remote=remote_status, repo_type=repo_type, access=access, note=note)


def save_outputs(rows: List[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # CSV
    csv_path = out_dir / "models_check.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Markdown
    md_path = out_dir / "models_check.md"
    if tabulate:
        table = tabulate([list(r.values()) for r in rows],
                         headers=list(rows[0].keys()), tablefmt="github")
    else:
        headers = list(rows[0].keys())
        table = "|" + "|".join(headers) + "|\n" \
                + "|" + "|".join(["---"] * len(headers)) + "|\n"
        for r in rows:
            table += "|" + "|".join(str(v) for v in r.values()) + "|\n"

    md_path.write_text(table, encoding="utf-8")
    print(f"\nSaved: {csv_path}\nSaved: {md_path}")


# --------------------------- main ---------------------------

def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Test models from models_api.yaml")
    p.add_argument("--yaml", default=DEFAULT_YAML, help="Path to models_api.yaml")
    p.add_argument("--out", default="out", help="Output directory for reports")
    args = p.parse_args(argv)

    print_install_procedure()

    # 1) Local models
    print("== Local HF cache models ==")
    local = list_local_models_from_cache()
    if local:
        for rid in local:
            print(" -", rid)
    else:
        print("(none found in cache)")

    # 2) API key test
    print("== API token check ==")
    has_token, token = have_api()
    if has_token:
        masked = token[:6] + "…" if token else "<unknown>"
        print("Token available:", masked)
    else:
        print("No token detected. Some gated models may be reported as 'gated' or 'no_api'.")
        print_no_token_help()

    # 3) Load YAML
    yaml_path = Path(args.yaml)
    if not yaml_path.exists():
        print(f"ERROR: YAML file not found: {yaml_path}")
        return 2
    specs = read_models_yaml(yaml_path)
    if not specs:
        print("No models found in YAML.")
        return 0

    # 4) Check all models
    print("== Checking models from YAML ==")
    rows: List[dict] = []
    for m in specs:
        chk = check_model(m, token)
        row = {
            "enterprise": m.enterprise,
            "model_id": m.model_id,
            "params_b": m.params_b if m.params_b is not None else "",
            "context_length": m.context_length if m.context_length is not None else "",
            "installed": "yes" if chk.installed else "no",
            "remote": chk.remote,
            "repo_type": chk.repo_type,   # public | gated | unknown
            "access": chk.access,         # ok | no | unknown
            "note": chk.note,
        }
        rows.append(row)
        print(f" - {m.model_id:45s} installed={row['installed']:<3s} "
              f"remote={row['remote']} repo_type={row['repo_type']} access={row['access']} "
              f"{('('+row['note']+')') if row['note'] else ''}")

    # 5) Summary table
    print("== Summary ==")
    if tabulate:
        print(tabulate(rows, headers="keys", tablefmt="github"))
    else:
        headers = list(rows[0].keys())
        print("|" + "|".join(headers) + "|")
        print("|" + "|".join(["---"] * len(headers)) + "|")
        for r in rows:
            print("|" + "|".join(str(v) for v in r.values()) + "|")

    save_outputs(rows, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
