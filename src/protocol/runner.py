# src/protocol/runner.py
import os, sys, yaml, json, time, re
from datetime import datetime
from pathlib import Path
from src.llm_io.hf_client_api import HFClientAPI  # API only

ROOT = Path(__file__).resolve().parents[2]
CFG_API  = ROOT / "configs" / "models_api.yaml"
RUNS_DIR = ROOT / "runs"

def _load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "", str(name).split("/")[-1])

def _extract_models(cfg: dict):
    out = []
    # 1) Format model_1, model_2, ...
    for k, v in cfg.items():
        if isinstance(v, dict) and k.startswith("model_") and "model_name" in v:
            try:
                n = int(k.split("_", 1)[1])
            except Exception:
                n = 10**9
            out.append((n, v))
    # 2) Format models: [ {model_name: ...}, ... ]
    if not out:
        lst = cfg.get("models", [])
        if isinstance(lst, list):
            out = list(enumerate(lst, 1))
    out.sort(key=lambda x: x[0])
    return [v for _, v in out]

def _get_token(cfg: dict) -> str:
    tok = os.getenv("HF_TOKEN")
    if tok: return tok
    try:
        import keyring
        tok = keyring.get_password("hf", "HF_TOKEN")
        if tok: return tok
    except Exception:
        pass
    return cfg.get("hf_token", "")

def _make_api_client(m: dict, token: str):
    name = m["model_name"]
    mx   = int(m.get("max_new_tokens", 64))
    temp = float(m.get("temperature", 0.7))
    if not token:
        raise RuntimeError("HF_TOKEN manquant pour l’API Hugging Face.")
    return HFClientAPI(name, token, mx, temp)

def main():
    cfg = _load_yaml(CFG_API)
    models = _extract_models(cfg)
    if not models:
        raise SystemExit(f"{CFG_API.as_posix()}: aucun modèle trouvé (ni 'model_X', ni 'models').")
    token = _get_token(cfg)

    # API-only, bloque tout download local
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    # Print de contrôle
    print(f"Models détectés ({len(models)}):")
    for i, m in enumerate(models, 1):
        print(f"  [{i}] {m.get('model_name')}  (max_new_tokens={m.get('max_new_tokens',64)}, temp={m.get('temperature',0.7)})")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = RUNS_DIR / f"RUN_{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = ["1+1 = ?"]

    for idx, m in enumerate(models, 1):
        model_label = m.get("model_name", f"model_{idx}")
        safe_name = _sanitize(model_label)
        out_path = out_dir / f"{idx:02d}_{safe_name}.json"
        try:
            C = _make_api_client(m, token)  # A = B = C
            turns, ts = [], datetime.now().strftime("%Y%m%d_%H%M%S")
            for i, user in enumerate(prompts, 1):
                t0 = time.time(); a = C.generate(user); ta = (time.time() - t0) * 1000.0
                t0 = time.time(); b = C.generate(a);    tb = (time.time() - t0) * 1000.0
                turns.append({
                    "turn": i, "timestamp": ts, "user_prompt": user,
                    "a_output": a, "b_output": b, "latency_ms": round(ta+tb, 1)
                })
                print(f"[{idx}/{len(models)}][turn {i}] {safe_name}: {round(ta+tb,1)} ms")

            rec = {
                "run_dir": out_dir.as_posix(),
                "run_timestamp": ts,
                "mode": "api",
                "model_name": model_label,
                "params": {
                    "max_new_tokens": m.get("max_new_tokens", 64),
                    "temperature": m.get("temperature", 0.7),
                },
                "turns": turns
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)

        except Exception as e:
            err_path = out_dir / f"{idx:02d}_{safe_name}_ERROR.json"
            with open(err_path, "w", encoding="utf-8") as f:
                json.dump({
                    "run_dir": out_dir.as_posix(),
                    "model_name": model_label,
                    "error": str(e)
                }, f, ensure_ascii=False, indent=2)
            print(f"[{idx}/{len(models)}] ERROR {safe_name}: {e}")

    print(f"Résultats: {out_dir.as_posix()}")

if __name__ == "__main__":
    main()
