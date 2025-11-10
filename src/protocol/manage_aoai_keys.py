#!/usr/bin/env python3
"""
Manager simple des clés Azure OpenAI par déploiement (config JSON).
- Liste tous les modèles de configs/models_azure.json et l'état des clés.
- "ensure": demande la clé manquante pour chaque deployment et l'enregistre (keyring).
- "delete": supprime la clé d'un deployment (ou --all pour tout effacer).

Dépendances: pip install keyring
Format attendu (extrait):
{
  "azure": { "endpoint": "https://...azure.com" },
  "models": [
    {
      "model_name": "gpt-4.1-nano",
      "deployment": "gpt-4.1-nano",
      "endpoint": "https://...azure.com"
    }
  ]
}
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import json

SERVICE = "aoai"  # namespace dans le keyring


def die(msg: str, code: int = 1):
    print(f"[ERR] {msg}", file=sys.stderr)
    sys.exit(code)


def load_json(p: Path) -> dict:
    if not p.exists():
        die(f"Fichier introuvable: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                die("Le fichier JSON racine doit être un objet { ... }")
            return data
    except json.JSONDecodeError as e:
        die(f"JSON invalide: {e}")


def iter_models(cfg: dict):
    models = cfg.get("models") or []
    if not isinstance(models, list):
        return
    default_ep = (cfg.get("azure") or {}).get("endpoint", "")
    for m in models:
        if not isinstance(m, dict):
            continue
        dep = m.get("deployment")
        if not dep:
            continue
        name = m.get("model_name", dep)
        endpoint = m.get("endpoint") or default_ep or ""
        yield {"deployment": dep, "model_name": name, "endpoint": endpoint}


def get_keyring():
    try:
        import keyring  # type: ignore
        return keyring
    except Exception:
        die("Module 'keyring' manquant. Installe: pip install keyring")


def key_present(keyring_mod, deployment: str) -> bool:
    return bool(keyring_mod.get_password(SERVICE, deployment) or "")


def cmd_list(args):
    cfg = load_json(Path(args.config))
    print(f"{'Deployment':<30} {'Model':<24} {'Endpoint':<45} Key")
    print("-" * 110)
    kr = get_keyring()
    for m in iter_models(cfg):
        dep = m["deployment"]; name = m["model_name"]; ep = m["endpoint"]
        present = "✅" if key_present(kr, dep) else "❌"
        print(f"{dep:<30} {name:<24} {ep:<45} {present}")


def cmd_ensure(args):
    cfg = load_json(Path(args.config))
    kr = get_keyring()
    from getpass import getpass
    changed = 0
    for m in iter_models(cfg):
        dep = m["deployment"]
        if key_present(kr, dep):
            continue
        print(f"[INFO] Clé manquante pour '{dep}'.")
        k = getpass(f" → Entrer la clé API pour {dep}: ").strip()
        if not k:
            print("[SKIP] Vide, ignoré.")
            continue
        kr.set_password(SERVICE, dep, k)
        print(f"[OK] Clé enregistrée pour {dep}.")
        changed += 1
    if changed == 0:
        print("[OK] Toutes les clés sont déjà présentes.")


def cmd_delete(args):
    cfg = load_json(Path(args.config))
    kr = get_keyring()
    if args.all:
        confirm = input("Confirmer la suppression de TOUTES les clés ? (yes/NO): ").strip().lower()
        if confirm != "yes":
            print("Annulé.")
            return
        count = 0
        for m in iter_models(cfg):
            dep = m["deployment"]
            try:
                kr.delete_password(SERVICE, dep)
                print(f"[DEL] {dep}")
                count += 1
            except Exception:
                pass
        print(f"[OK] Suppression terminée ({count} clés).")
        return

    if not args.deployment:
        die("Spécifie --deployment <nom> ou --all.")
    try:
        kr.delete_password(SERVICE, args.deployment)
        print(f"[OK] Clé supprimée pour {args.deployment}")
    except Exception:
        print(f"[WARN] Aucune clé trouvée pour {args.deployment}")


def main():
    default_cfg = Path(__file__).resolve().parents[2] / "configs" / "models_azure.json"
    ap = argparse.ArgumentParser(description="Manager des clés Azure OpenAI (par deployment, config JSON)")
    ap.add_argument("--config", default=str(default_cfg), help="Chemin du models_azure.json")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="Lister l'état des clés par deployment")
    sub.add_parser("ensure", help="Demander et enregistrer les clés manquantes")

    p_del = sub.add_parser("delete", help="Supprimer une clé")
    p_del.add_argument("--deployment", help="Nom du deployment à supprimer")
    p_del.add_argument("--all", action="store_true", help="Supprimer toutes les clés")

    args = ap.parse_args()
    if args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "ensure":
        cmd_ensure(args)
    elif args.cmd == "delete":
        cmd_delete(args)


if __name__ == "__main__":
    main()
