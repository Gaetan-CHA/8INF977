# Catalogue — Attaques et Défenses LLM (avec sources / liens)

## Attaques

| id                   | brief description (ta ligne)                                                                 | source(s) (nom / liens) |
|----------------------|----------------------------------------------------------------------------------------------:|:------------------------|
| `manyshot_jailbreak` | Many-Shot Jailbreaking — context saturation / instruction conflict (NeurIPS 2024 inspired)     | GitHub - AjentDojo (llm-attacks) — https://github.com/llm-attacks/llm-attacks?utm_source=catalyzec.com  <br> Hugging Face - WizardLM 13B Uncensored — https://huggingface.co/QuixiAI/WizardLM-13B-Uncensored  <br> Hugging Face - paper 2405.21018 — https://huggingface.co/papers/2405.21018 |
| `tree_of_attacks`    | Tree-of-Attacks — automatic multi-step jailbreaking via search/exploration (NeurIPS 2024)      | Tree-of-Attacks (repo) — https://github.com/RICommunity/TAP?utm_source=catalyzec.com  <br> (paper / NeurIPS 2024 references where available) |
| `direct_injection`   | Direct prompt injection — inline malicious instructions (AgentDojo / prompt injection lit.)   | GitHub - AjentDojo (AgentDojo) — https://github.com/ethz-spylab/agentdojo  <br> AgentDojo project page / assets — https://github.com/ethz-spylab/agentdojo |
| `indirect_injection` | Indirect prompt injection — payload delivered via external content (HTML/files) (AgentDojo)    | GitHub - AjentDojo (AgentDojo) — https://github.com/ethz-spylab/agentdojo  |
| `soft_prompt_threat` | Soft-Prompt Threats — embedding / soft-prompt perturbation attack (NeurIPS 2024)              | Hugging Face - paper 2405.21018 — https://huggingface.co/papers/2405.21018  <br> LLM Embedding Attack repo — https://github.com/SchwinnL/LLM_Embedding_Attack?utm_source=catalyzec.com |
| `data_poisoning`     | Data poisoning (training-time) — dataset-level backdoor simulation (Tay incident + NeurIPS)    | SATML / CTF repo — https://github.com/ethz-spylab/satml-llm-ctf  <br> Example data-poisoning resources (community) — https://github.com/0xk1h0/ChatGPT_DAN?utm_source=catalyzec.com |
| `prompt_adversarial_tuning` | Prompt Adversarial Tuning (PAT/APO) — optimization-driven adversarial prompt search (NeurIPS 2024) | PKU-ML / PAT repo — https://github.com/PKU-ML/PAT  <br> SATML / related PAT materials — https://github.com/ethz-spylab/satml-llm-ctf |
| `prompt_leakage`     | Prompt / system-prompt leakage — techniques to extract hidden instructions (benchmarks)       | SG-Bench repo — https://github.com/MurrayTom/SG-Bench?utm_source=catalyzec.com  <br> Stanford Alpaca / resources — https://github.com/tatsu-lab/stanford_alpaca?utm_source=catalyzec.com |
| `role_confusion`     | Role confusion / hijack — force the model to adopt another role (system/admin) to bypass rules | GitHub - AjentDojo (AgentDojo) — https://github.com/ethz-spylab/agentdojo  |
| `token_noise`        | Token-level adversarial noise — homoglyphs / unicode perturbations to evade filters          | UnitaryAI / detoxify (tools & preproc) — https://github.com/unitaryai/detoxify?utm_source=catalyzec.com  <br> LLM Embedding Attack repo — https://github.com/SchwinnL/LLM_Embedding_Attack?utm_source=catalyzec.com |
| `roleplay_jailbreak` | Roleplay / persona jailbreak — social-engineering via fictional scenarios to disable rules     | NeurIPS 2023 - *Jailbroken* (paper PDF local / canonical) — **NeurIPS-2023-jailbroken-how-does-llm-safety-training-fail-Paper-Conference.pdf**  <br> (context / related model examples) — https://huggingface.co/QuixiAI/WizardLM-13B-Uncensored |

---

## Défenses

| id                          | type (pre/in/post) | brief description (ta ligne)                                            | source(s) (nom / liens) |
|-----------------------------|-------------------:|-------------------------------------------------------------------------:|:------------------------|
| `pre_sanitizer`             | pre                | Nettoyage HTML, déobfuscation unicode, trimming du prompt               | GitHub - AjentDojo (sanitization sections) — https://github.com/ethz-spylab/agentdojo  |
| `prompt_adversarial_tuning` | pre                | Ajout de guard-prompt adversarial (PAT) avant le LLM                    | PKU-ML / PAT — https://github.com/PKU-ML/PAT  <br> SATML materials — https://github.com/ethz-spylab/satml-llm-ctf |
| `prompt_injection_detector` | pre                | Détection heuristique ou ML des prompts suspects                        | GitHub - AjentDojo (detectors / env) — https://github.com/ethz-spylab/agentdojo  |
| `perplexity_detector`       | pre                | Détection statistique par perplexité ou surprisal                       | Mission Impossible / statistical analyses — https://github.com/ethz-spylab/satml-llm-ctf  <br> alignment handbook refs — https://github.com/huggingface/alignment-handbook?utm_source=catalyzec.com |
| `embedding_space_defense`   | pre                | Surveillance et normalisation des embeddings (soft-prompt defense)      | Protecting your LLMs (info-bottleneck) — https://www.catalyzex.com/paper/protecting-your-llms-with-information/code  <br> LLM Embedding Attack (for tests) — https://github.com/SchwinnL/LLM_Embedding_Attack?utm_source=catalyzec.com |
| `in_system_shield`          | in                 | Renforcement du system-prompt et policy injection                       | SG-Bench / system prompt evals — https://github.com/MurrayTom/SG-Bench?utm_source=catalyzec.com  <br> Stanford Alpaca resources — https://github.com/tatsu-lab/stanford_alpaca?utm_source=catalyzec.com |
| `tool_isolation`            | in                 | Restriction des outils disponibles (Tool filtering / Agent isolation)    | GitHub - AjentDojo (agent/tool isolation) — https://github.com/ethz-spylab/agentdojo  |
| `adversarial_finetune`      | in                 | Fine-tuning robuste sur exemples adversariaux                           | PKU-ML / PAT — https://github.com/PKU-ML/PAT  <br> SATML materials — https://github.com/ethz-spylab/satml-llm-ctf |
| `certified_defenses`        | in                 | Approches de certification formelle de robustesse                       | Protecting your LLMs (info-bottleneck) — https://www.catalyzex.com/paper/protecting-your-llms-with-information/code |
| `post_classifier`           | post               | Classifieur de sécurité / filtrage de sortie (redact on fail)           | SG-Bench / moderation evals — https://github.com/MurrayTom/SG-Bench?utm_source=catalyzec.com  <br> Stanford Alpaca — https://github.com/tatsu-lab/stanford_alpaca?utm_source=catalyzec.com |
| `output_filtering`          | post               | Filtrage syntaxique et suppression de données sensibles                 | SG-Bench / moderation evals — https://github.com/MurrayTom/SG-Bench?utm_source=catalyzec.com  |
| `ensemble_moderation`       | post               | Agrégation de verdicts multi-défenses (ensemble moderation)             | SG-Bench / ensemble ideas — https://github.com/MurrayTom/SG-Bench?utm_source=catalyzec.com  |
| `red_team_validator`        | post               | Seconde validation LLM (red-team / self-critique)                       | SATML / LLM CTF datasets & analysis — https://github.com/ethz-spylab/ctf-satml24-data-analysis  <br> SATML repo — https://github.com/ethz-spylab/satml-llm-ctf |
| `human_in_the_loop`         | post               | Escalade humaine pour cas sensibles ou incertains                       | SATML / operational lessons — https://github.com/ethz-spylab/ctf-satml24-data-analysis |

---

### Remarques
- J'ai remplacé les noms de fichiers `.pdf` par des libellés de type **GitHub - AjentDojo**, **Hugging Face - ...**, **PKU-ML / PAT**, etc., comme demandé.  
- Si tu veux que je remplace un libellé précis (par ex. **GitHub - AjentDojo** → **GitHub - llm-attacks** ou autre orthographe), dis-le et j'applique la modification.  
- Je peux aussi fournir ce fichier en tant que document téléchargeable (MD / PDF). Veux-tu que je crée et propose le téléchargement du fichier `sources_llm_attacks_defenses.md` ?  
