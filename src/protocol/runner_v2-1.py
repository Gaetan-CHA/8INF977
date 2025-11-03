#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os, json, time, re, importlib
from pathlib import Path
from typing import Dict, Any, List
import yaml

# === Chemins ===
ROOT = Path(__file__).resolve().parents[0]
CFG_MODELS = Path(__file__).resolve().parents[2] / "configs" / "models_azure.yaml"
CFG_ATTACKS = Path(__file__).resolve().parents[2] / "configs" / "attacks.yaml"
CFG_RUNNER  = Path(__file__).resolve().parents[2] / "configs" / "runner_attacks.yaml"


# === DEBUG FLAGS (toggle true/false) ===
DEBUG_YAML = False         # affiche le contenu YAML chargé
DEBUG_SELECTION = True    # affiche modèles/attaques demandés et retenus
DEBUG_CLIENTS = False      # affiche clients construits
DEBUG_ATTACKS = False      # affiche modules d'attaque importés et prompts générés
DEBUG_RUN = False          # affiche déroulement des runs (dry_run & résultats)
DEBUG_API = False          # affiche réponses API (si non dry_run)

# === Utils YAML ===
def load_yaml(p: Path) -> Any:
    if not p.exists():
        if DEBUG_YAML:
            print(f"[DEBUG_YAML] fichier absent: {p}")
        return {}
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    if DEBUG_YAML:
        print(f"[DEBUG_YAML] Chargé: {p.resolve()}")
        print(f"[DEBUG_YAML] Preview: {str(data)[:1000]}")
    return data

# === Modèles Azure ===
from openai import AzureOpenAI

def get_azure_key(deployment: str) -> str:
    try:
        import keyring
        k = keyring.get_password("aoai", deployment)
        if k:
            if DEBUG_CLIENTS:
                print(f"[DEBUG_CLIENTS] clé keyring pour {deployment}: {k[:6]}…")
            return k
    except Exception:
        pass
    env_name = f"AZURE_OPENAI_KEY__{deployment.upper().replace('-', '_')}"
    val = os.getenv(env_name) or os.getenv("AZURE_OPENAI_KEY", "")
    if DEBUG_CLIENTS:
        print(f"[DEBUG_CLIENTS] clé env {env_name}: {'SET' if val else 'MISSING'}")
    return val

class AzureClient:
    def __init__(self, *, endpoint: str, api_key: str, api_version: str,
                 deployment: str, temperature: float = 0.7,
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
        self._client = AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=api_key)
        if DEBUG_CLIENTS:
            print(f"[DEBUG_CLIENTS] AzureClient init: deployment={deployment}, endpoint={endpoint}, api_version={api_version}")

    def chat(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.deployment,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )
        out = resp.choices[0].message.content or ""
        if DEBUG_API:
            print(f"[DEBUG_API] API reply len={len(out)}")
        return out

# === Chargement modèles ===

def extract_models(cfg: dict) -> List[dict]:
    out = []
    for m in (cfg or {}).get("models", []):
        if not isinstance(m, dict):
            continue
        name = m.get("model_name") or m.get("id")
        dep  = m.get("deployment") or name
        if not name or not dep:
            continue
        out.append({
            "model_name": name,
            "deployment": dep,
            "temperature": float(m.get("temperature", 0.7)),
            "max_completion_tokens": int(m.get("max_completion_tokens", 512)),
            "top_p": float(m.get("top_p", 1.0)),
            "frequency_penalty": float(m.get("frequency_penalty", 0.0)),
            "presence_penalty": float(m.get("presence_penalty", 0.0)),
            "api_version": m.get("api_version"),
            "endpoint": m.get("endpoint"),
        })
    if DEBUG_YAML:
        print(f"[DEBUG_YAML] extract_models -> {len(out)} modèles trouvés")
    return out

# === Import dynamique d'attaques ===

def load_attack_modules(catalog: List[dict]) -> Dict[str, Any]:
    mods = {}
    for item in catalog:
        atk_id = item.get("id")
        module_path = item.get("module")
        if not atk_id or not module_path:
            continue
        try:
            mods[atk_id] = importlib.import_module(module_path)
            if DEBUG_ATTACKS:
                print(f"[DEBUG_ATTACKS] importé {atk_id} <- {module_path}")
        except Exception as e:
            print(f"[WARN] impossible d'importer {module_path}: {e}")
    return mods

# === Sélecteurs ===

def select_ids(all_ids: list[str], wanted: list[str] | None) -> list[str]:
    if not wanted:
        return list(all_ids)
    wanted_norm = [str(w).strip() for w in wanted]
    if any(w.lower() == "all" for w in wanted_norm):
        return list(all_ids)
    wanted_set = {w.lower() for w in wanted_norm}
    return [x for x in all_ids if x.lower() in wanted_set]

# === Main run ===

def main():
    cfg_models = load_yaml(CFG_MODELS)
    cfg_runner = load_yaml(CFG_RUNNER)
    cfg_attacks = load_yaml(CFG_ATTACKS)

    if DEBUG_YAML:
        print(f"[DEBUG_YAML] runner config preview: {str(cfg_runner)[:1000]}")

    # Runner opts
    dry_run = bool(cfg_runner.get("dry_run", True))
    runs_dir = Path(cfg_runner.get("runs_dir", "runs")); runs_dir.mkdir(parents=True, exist_ok=True)
    prompts_file = cfg_runner.get("prompts_file")

    # Azure defaults
    az_def = (cfg_runner.get("azure") or {})
    endpoint_default = az_def.get("endpoint", (cfg_models.get("azure") or {}).get("endpoint", ""))
    api_version_default = az_def.get("api_version", (cfg_models.get("azure") or {}).get("api_version", "2024-12-01-preview"))

    # Models
    all_models = extract_models(cfg_models)
    model_names = [m["deployment"] for m in all_models]
    if DEBUG_SELECTION:
        print(f"[DEBUG_SELECTION] modèles disponibles: {model_names}")

    wanted_models = select_ids(model_names, cfg_runner.get("models", ["all"]))
    if DEBUG_SELECTION:
        print(f"[DEBUG_SELECTION] modèles demandés (config): {cfg_runner.get('models', ['all'])}")
        print(f"[DEBUG_SELECTION] modèles retenus: {wanted_models}")

    # Build clients for selected models
    clients = {}
    for m in all_models:
        if m["deployment"] not in wanted_models:
            if DEBUG_CLIENTS:
                print(f"[DEBUG_CLIENTS] skip modèle {m['deployment']} (non demandé)")
            continue
        endpoint = m.get("endpoint") or endpoint_default
        api_version = m.get("api_version") or api_version_default
        api_key = get_azure_key(m["deployment"]) if not dry_run else ""
        try:
            clients[m["deployment"]] = AzureClient(
                endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
                deployment=m["deployment"],
                temperature=m["temperature"],
                max_completion_tokens=m["max_completion_tokens"],
                top_p=m["top_p"],
                frequency_penalty=m["frequency_penalty"],
                presence_penalty=m["presence_penalty"],
                model_name=m["model_name"],
            )
            if DEBUG_CLIENTS:
                print(f"[DEBUG_CLIENTS] client construit pour {m['deployment']}")
        except Exception as e:
            print(f"[ERR] impossible de construire client pour {m['deployment']}: {e}")

    if not clients:
        if DEBUG_SELECTION or DEBUG_CLIENTS:
            print(f"[DEBUG] all_models keys: {[m['deployment'] for m in all_models]}")
            print(f"[DEBUG] wanted_models: {wanted_models}")
        raise SystemExit("Aucun modèle sélectionné.")

    # Attacks
    catalog = cfg_attacks if isinstance(cfg_attacks, list) else cfg_attacks.get("attacks", [])
    all_attack_ids = [a.get("id") for a in catalog if isinstance(a, dict) and a.get("id")]
    if DEBUG_ATTACKS:
        print(f"[DEBUG_ATTACKS] attaques catalog: {all_attack_ids}")
    selected_attack_ids = select_ids(all_attack_ids, cfg_runner.get("attacks", ["all"]))
    if DEBUG_ATTACKS:
        print(f"[DEBUG_ATTACKS] attaques demandées (config): {cfg_runner.get('attacks', ['all'])}")
        print(f"[DEBUG_ATTACKS] attaques retenues: {selected_attack_ids}")
    atk_modules = load_attack_modules([a for a in catalog if a.get("id") in selected_attack_ids])

    if not atk_modules:
        raise SystemExit("Aucune attaque sélectionnée.")

    # Base prompt (optionnel) lu depuis prompts.yaml, sinon laissé aux attaques
    base_prompt = None
    if prompts_file and Path(prompts_file).exists():
        data = load_yaml(Path(prompts_file))
        if isinstance(data, dict) and isinstance(data.get("prompts"), list) and data["prompts"]:
            base_prompt = str(data["prompts"][0].get("text", ""))
            if DEBUG_YAML:
                print(f"[DEBUG_YAML] prompt de base: {base_prompt[:200]}")

    # Run matrix: pour chaque modèle, pour chaque attaque
    out_path = runs_dir / f"ATTACKS_{int(time.time())}.jsonl"
    with open(out_path, "w", encoding="utf-8") as w:
        for model_id, client in clients.items():
            for atk_id, mod in atk_modules.items():
                context = {"base_prompt": base_prompt or "Explique le mécanisme X en 3 points."}
                gen = getattr(mod, "generate", None)
                if not callable(gen):
                    print(f"[SKIP] attaque {atk_id}: pas de generate(context)")
                    continue
                try:
                    atk = gen(context) or {}
                except Exception as e:
                    print(f"[ERR] génération prompt pour {atk_id}: {e}")
                    w.write(json.dumps({"event":"error_gen", "model": model_id, "attack": atk_id, "error": str(e)}, ensure_ascii=False) + "\n")
                    w.flush()
                    continue

                prompt = atk.get("prompt", "")
                if DEBUG_ATTACKS:
                    safe_preview = prompt[:300].replace("\n", "\\n")
                    print(f"[DEBUG_ATTACKS] prompt généré pour {atk_id} (len={len(prompt)}): {safe_preview}")


                meta = {"model": model_id, "attack": atk_id, "ts": int(time.time())}

                if dry_run:
                    if DEBUG_RUN:
                        print("\n=== DRY RUN ===")
                        print(f"Model: {model_id}")
                        print(f"Attack: {atk_id}")
                        print(f"Prompt:\n{prompt[:400]}\n")
                    w.write(json.dumps({"event":"dry_run", **meta, "prompt": prompt}, ensure_ascii=False) + "\n")
                    w.flush()
                    continue

                try:
                    t0 = time.time()
                    system = "You are a helpful assistant."
                    out = client.chat(system=system, user=prompt)
                    dt = round((time.time()-t0)*1000, 1)
                    rec = {**meta, "latency_ms": dt, "prompt": prompt, "output": out}
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n"); w.flush()
                    if DEBUG_RUN:
                        print(f"[RUN] [{model_id}] {atk_id} -> {dt} ms, out_len={len(out)}")
                    if DEBUG_API:
                        safe_out = out[:400].replace("\n", "\\n")
                        print(f"[DEBUG_API] sortie (preview 400):\n{safe_out}")
                except Exception as e:
                    w.write(json.dumps({"event":"error", **meta, "error": str(e)}, ensure_ascii=False) + "\n")
                    w.flush()
                    print(f"[ERR] {model_id} × {atk_id}: {e}")

    if DEBUG_RUN:
        print(f"[RUN] Résultats: {out_path}")

if __name__ == "__main__":
    main()
