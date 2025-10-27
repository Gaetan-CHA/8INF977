# src/protocol/runner.py
import os, sys, yaml, json, time, re
from datetime import datetime
from pathlib import Path
from src.llm_io.hf_client_local import HFClientLocal
from src.llm_io.hf_client_api import HFClientAPI

ROOT = Path(__file__).resolve().parents[2]
CFG_API   = ROOT / "configs" / "models_api.yaml"
CFG_LOCAL = ROOT / "configs" / "models_local.yaml"
RUNS_DIR  = ROOT / "runs"

def _load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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

def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "", name.split("/")[-1])

def _load_cfg_for_mode(mode: str) -> dict:
    return _load_yaml(CFG_API if mode == "api" else CFG_LOCAL)

def load_models(mode: str):
    cfg = _load_cfg_for_mode(mode)
    a, b = cfg["model_a"], cfg["model_b"]
    if mode == "api":
        token = _get_token(cfg)
        if not token:
            raise RuntimeError("Token manquant: keyring(HF/HF_TOKEN) ou env HF_TOKEN ou cfg.hf_token.")
        A = HFClientAPI(a["model_name"], token, a.get("max_new_tokens",200), a.get("temperature",0.7))
        B = HFClientAPI(b["model_name"], token, b.get("max_new_tokens",200), b.get("temperature",0.7))
    else:
        A = HFClientLocal(a["model_name"], a.get("max_new_tokens",200), a.get("temperature",0.7))
        B = HFClientLocal(b["model_name"], b.get("max_new_tokens",200), b.get("temperature",0.7))
    return A, B

def dialogue(prompts, A, B, mode: str, out_dir: Path = RUNS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_a, name_b = _sanitize(getattr(A,"model_name","A")), _sanitize(getattr(B,"model_name","B"))
    out_file = out_dir / f"{mode.capitalize()}_{ts}_{name_a}_{name_b}.jsonl"
    with open(out_file, "w", encoding="utf-8") as w:
        for i, user in enumerate(prompts, 1):
            t0=time.time(); a=A.generate(user); ta=(time.time()-t0)*1000
            t0=time.time(); b=B.generate(a);    tb=(time.time()-t0)*1000
            row={"turn":i,"mode":mode,"timestamp":ts,"user_prompt":user,
                 "a_model":getattr(A,"model_name","A"),"b_model":getattr(B,"model_name","B"),
                 "a_output":a,"b_output":b,"latency_ms":round(ta+tb,1)}
            w.write(json.dumps(row, ensure_ascii=False)+"\n")
            print(f"[turn {i}] {row['latency_ms']} ms")
    print(f"Run stored in {out_file}")

def main():
    mode = (sys.argv[1].lower() if len(sys.argv)>1 else "api")
    if mode not in {"local","api"}:
        raise SystemExit("Usage: python -m src.protocol.runner [local|api]")
    A,B = load_models(mode)
    prompts = [
        "Présente-toi en une phrase.",
        "Explique brièvement comment fonctionne une blockchain.",
        "Donne 3 conseils pour sécuriser un mot de passe."
    ]
    dialogue(prompts, A, B, mode)

if __name__ == "__main__":
    main()
