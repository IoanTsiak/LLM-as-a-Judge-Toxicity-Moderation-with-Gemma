#!/usr/bin/env python3
"""
LLM-as-a-Judge Toxicity Moderation
Single-file implementation for CSE 528 assignment.

Implements:
- A: Direct generation (strict JSON + validator + DEFER fallback)
- B: Log-likelihood scoring over candidate JSON completions
- C: Self-consistency with agreement-based uncertainty
- Uncertainty / DEFER thresholds
- Coverage-risk curves
- Robustness attacks + mitigation
- Consistency experiment across seeds
- CLI demo/eval/robustness/consistency

Dataset CSV expected columns:
- text/comment/comment_text/content (one of these), and
- label/toxic/target (one of these)
Where label is binary-ish: toxic=1, non-toxic=0.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None


LABELS = ["TOXIC", "NON_TOXIC", "DEFER"]
CATEGORIES = ["insult", "threat", "hate", "harassment", "profanity", "other", "none"]
TEXT_COLUMNS = ["text", "comment", "comment_text", "content"]
LABEL_COLUMNS = ["label", "toxic", "target"]


@dataclass
class JudgeConfig:
    model_name: str
    max_new_tokens: int = 128
    device: Optional[str] = None
    use_4bit: bool = False
    defer_threshold: float = 0.60
    agreement_threshold: float = 0.67
    max_input_chars: int = 1200


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label(value) -> int:
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "toxic", "yes"}:
            return 1
        if s in {"0", "false", "non_toxic", "non-toxic", "clean", "no"}:
            return 0
        try:
            return 1 if float(s) >= 0.5 else 0
        except Exception:
            return 1 if "toxic" in s and "non" not in s else 0
    try:
        return 1 if float(value) >= 0.5 else 0
    except Exception:
        return 0


def normalize_text_for_mitigation(text: str) -> str:
    t = str(text)
    repl = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
    t = t.translate(repl)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b", lambda m: re.sub(r"\s+", "", m.group(0)), t)
    return t.strip()


class RobustnessSuite:
    @staticmethod
    def typos_noise(text: str) -> str:
        text = str(text)
        if len(text) < 6:
            return text
        chars = list(text)
        idxs = [i for i, c in enumerate(chars[:-1]) if c.isalpha() and chars[i + 1].isalpha()]
        if not idxs:
            return text
        i = random.choice(idxs)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)

    @staticmethod
    def char_repetition(text: str) -> str:
        text = str(text)
        return re.sub(r"([A-Za-z])", lambda m: m.group(1) * random.choice([1, 1, 2, 3]), text, count=6)

    @staticmethod
    def spacing_attack(text: str) -> str:
        text = str(text)
        words = text.split()
        if not words:
            return text
        target = max(words, key=len)
        attacked = " ".join(list(target))
        return text.replace(target, attacked, 1)

    @staticmethod
    def leetspeak(text: str) -> str:
        text = str(text)
        mapping = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "A": "4", "E": "3", "I": "1", "O": "0", "S": "5", "T": "7"})
        return text.translate(mapping)

    @staticmethod
    def benign_quote(text: str) -> str:
        text = str(text)
        return f'He said, "{text}", but I disagree with that statement.'

    @classmethod
    def apply_all(cls, text: str) -> Dict[str, str]:
        return {
            "typos_noise": cls.typos_noise(text),
            "char_repetition": cls.char_repetition(text),
            "spacing_attack": cls.spacing_attack(text),
            "leetspeak": cls.leetspeak(text),
            "benign_quote": cls.benign_quote(text),
        }


class ToxicJudge:
    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}
        if cfg.use_4bit and self.device.startswith("cuda") and BitsAndBytesConfig is not None:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["dtype"] = torch.float16 if self.device.startswith("cuda") else torch.float32

        try:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
        except TypeError:
            if "dtype" in model_kwargs:
                model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

        if not (cfg.use_4bit and self.device.startswith("cuda")):
            self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _truncate(self, text: str) -> str:
        return str(text)[: self.cfg.max_input_chars]

    def _messages(self, text: str) -> List[Dict[str, str]]:
        text = self._truncate(text)
        system = "You are a toxicity moderation judge. Return ONLY one valid JSON object. No markdown, no code fences, no prose outside JSON."
        user = (
            "Classify the following text for toxicity.\n"
            f'Text: "{text}"\n\n'
            "Return exactly one JSON object with this schema:\n"
            "{\n"
            '  "label": "TOXIC|NON_TOXIC|DEFER",\n'
            '  "category": "insult|threat|hate|harassment|profanity|other|none",\n'
            '  "confidence": 0.0,\n'
            '  "short_rationale": "1-2 short sentences max"\n'
            "}\n\n"
            "Rules:\n"
            "- label must be exactly one of TOXIC, NON_TOXIC, DEFER\n"
            "- category must be exactly one of insult, threat, hate, harassment, profanity, other, none\n"
            "- confidence must be a number between 0 and 1\n"
            "- If uncertain, output DEFER\n"
            "- Return JSON only"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _render_prompt(self, text: str) -> str:
        messages = self._messages(text)
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return f"SYSTEM: {messages[0]['content']}\nUSER: {messages[1]['content']}\nASSISTANT:"

    def _generate_text(self, prompt: str, do_sample: bool, temperature: Optional[float], top_p: Optional[float], seed: Optional[int] = None) -> str:
        if seed is not None:
            set_seed(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        kwargs = {
            **inputs,
            "max_new_tokens": self.cfg.max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            kwargs["temperature"] = temperature if temperature is not None else 0.7
            kwargs["top_p"] = top_p if top_p is not None else 0.9
        with torch.no_grad():
            out = self.model.generate(**kwargs)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[dict]:
        text = str(text).strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None

    @staticmethod
    def _validate_json(obj: Optional[dict]) -> Tuple[bool, dict]:
        if not isinstance(obj, dict):
            return False, {"label": "DEFER", "category": "other", "confidence": 0.0, "short_rationale": "Invalid JSON output; deferred by validator.", "valid_json": False}
        has_required = all(k in obj for k in ["label", "category", "confidence", "short_rationale"])
        label = str(obj.get("label", "DEFER")).strip().upper()
        category = str(obj.get("category", "other")).strip().lower()
        rationale = str(obj.get("short_rationale", "No rationale.")).strip()
        try:
            confidence = float(obj.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        valid = True
        if label not in LABELS:
            label = "DEFER"
            valid = False
        if category not in CATEGORIES:
            category = "other"
            valid = False
        if not has_required:
            valid = False
        confidence = max(0.0, min(1.0, confidence))
        return bool(valid), {"label": label, "category": category, "confidence": confidence, "short_rationale": rationale[:240], "valid_json": bool(valid)}

    @staticmethod
    def _toxic_score_from_label(label: str, confidence: float) -> float:
        if label == "TOXIC":
            return max(confidence, 0.5)
        if label == "NON_TOXIC":
            return 1.0 - max(confidence, 0.5)
        return 0.5

    def predict_direct(self, text: str) -> dict:
        prompt = self._render_prompt(text)
        t0 = now_ms()
        raw = self._generate_text(prompt, do_sample=False, temperature=None, top_p=None)
        latency = now_ms() - t0
        obj = self._extract_first_json_object(raw)
        valid, pred = self._validate_json(obj)
        if not valid:
            repair_prompt = prompt + "\nReturn JSON only. No extra text.\n"
            raw2 = self._generate_text(repair_prompt, do_sample=False, temperature=None, top_p=None)
            obj2 = self._extract_first_json_object(raw2)
            valid2, pred2 = self._validate_json(obj2)
            if valid2:
                raw = raw2
                pred = pred2
        if pred["confidence"] < self.cfg.defer_threshold:
            pred["label"] = "DEFER"
        pred.update({"latency_ms": round(latency, 2), "method": "direct", "raw_output": raw})
        pred["toxic_score"] = round(self._toxic_score_from_label(pred["label"], pred["confidence"]), 6)
        return pred

    def _completion_logprob(self, prompt: str, completion: str) -> float:
        full = prompt + completion
        enc_full = self.tokenizer(full, return_tensors="pt").to(self.device)
        enc_prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = enc_prompt["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model(**enc_full)
            logits = outputs.logits[:, :-1, :]
            labels = enc_full["input_ids"][:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            start = max(prompt_len - 1, 0)
            completion_logp = token_logp[:, start:]
            if completion_logp.numel() == 0:
                return float("-inf")
            score = completion_logp.mean().item()
        return float(score)

    def predict_loglikelihood(self, text: str) -> dict:
        prompt = self._render_prompt(text)
        candidates = {
            "TOXIC": '{"label":"TOXIC","category":"other","confidence":0.95,"short_rationale":"The text appears toxic."}',
            "NON_TOXIC": '{"label":"NON_TOXIC","category":"none","confidence":0.95,"short_rationale":"The text appears non-toxic."}',
            "DEFER": '{"label":"DEFER","category":"other","confidence":0.50,"short_rationale":"The model is uncertain."}',
        }
        t0 = now_ms()
        scores = {label: self._completion_logprob(prompt, comp) for label, comp in candidates.items()}
        latency = now_ms() - t0
        labels = list(scores.keys())
        vals = np.array([scores[k] for k in labels], dtype=np.float64)
        vals = vals - vals.max()
        probs = np.exp(vals)
        probs = probs / probs.sum()
        best_idx = int(np.argmax(probs))
        best_label = labels[best_idx]
        best_prob = float(probs[best_idx])
        second_prob = float(np.partition(probs, -2)[-2]) if len(probs) > 1 else 0.0
        margin = best_prob - second_prob
        final_label = best_label if best_prob >= self.cfg.defer_threshold else "DEFER"
        pred = {
            "label": final_label,
            "category": "other" if final_label != "NON_TOXIC" else "none",
            "confidence": round(best_prob, 6),
            "short_rationale": f"Chosen by mean completion likelihood; margin={margin:.3f}.",
            "latency_ms": round(latency, 2),
            "valid_json": True,
            "method": "loglikelihood",
            "scores": {k: round(v, 6) for k, v in scores.items()},
            "label_probs": {k: round(float(probs[i]), 6) for i, k in enumerate(labels)},
        }
        pred["toxic_score"] = round(float(probs[labels.index("TOXIC")]), 6)
        return pred

    def predict_self_consistency(self, text: str, n: int = 5, seed: int = 42) -> dict:
        prompt = self._render_prompt(text)
        t0 = now_ms()
        votes = []
        parsed = []
        for i in range(n):
            raw = self._generate_text(prompt, do_sample=True, temperature=0.7, top_p=0.9, seed=seed + i)
            obj = self._extract_first_json_object(raw)
            _, pred = self._validate_json(obj)
            parsed.append(pred)
            votes.append(pred["label"])
        latency = now_ms() - t0
        counts = {k: votes.count(k) for k in LABELS}
        best_label = max(counts, key=counts.get)
        agreement = counts[best_label] / max(n, 1)
        mean_conf = float(np.mean([p["confidence"] for p in parsed])) if parsed else 0.0
        final_label = best_label if agreement >= self.cfg.agreement_threshold and mean_conf >= self.cfg.defer_threshold else "DEFER"
        toxic_votes = counts["TOXIC"] / max(n, 1)
        pred = {
            "label": final_label,
            "category": "other" if final_label != "NON_TOXIC" else "none",
            "confidence": round(agreement, 6),
            "short_rationale": f"Self-consistency over {n} samples; agreement={agreement:.2f}.",
            "latency_ms": round(latency, 2),
            "valid_json": all(p.get("valid_json", False) for p in parsed) if parsed else False,
            "method": "self_consistency",
            "votes": counts,
            "mean_sample_confidence": round(mean_conf, 6),
        }
        pred["toxic_score"] = round(float(toxic_votes), 6)
        return pred

    def predict(self, text: str, method: str, self_n: int = 5, seed: int = 42) -> dict:
        if method == "A":
            return self.predict_direct(text)
        if method == "B":
            return self.predict_loglikelihood(text)
        if method == "C":
            return self.predict_self_consistency(text, n=self_n, seed=seed)
        raise ValueError("method must be one of A/B/C")


def find_text_column(df: pd.DataFrame) -> str:
    for c in TEXT_COLUMNS:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find text column. Tried: {TEXT_COLUMNS}")


def find_label_column(df: pd.DataFrame) -> str:
    for c in LABEL_COLUMNS:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find label column. Tried: {LABEL_COLUMNS}")


def load_dataset(path: str, sample_n: Optional[int], seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = find_text_column(df)
    label_col = find_label_column(df)
    df = df[[text_col, label_col]].dropna().copy()
    df.columns = ["text", "gold_raw"]
    df["gold"] = df["gold_raw"].map(normalize_label)
    if sample_n is not None and sample_n < len(df):
        df = df.sample(sample_n, random_state=seed).reset_index(drop=True)
    return df


def evaluate_predictions(df: pd.DataFrame, out_dir: Path, prefix: str) -> Dict[str, float]:
    y_true = df["gold"].astype(int).tolist()
    y_score = df["toxic_score"].astype(float).tolist()
    covered_mask = df["pred_label"] != "DEFER"
    coverage = float(covered_mask.mean())
    metrics = {
        "n_examples": int(len(df)),
        "coverage": coverage,
        "defer_rate": float(1.0 - coverage),
        "json_validity_rate": float(df["valid_json"].astype(bool).mean()) if "valid_json" in df else float("nan"),
        "avg_latency_ms": float(df["latency_ms"].astype(float).mean()) if "latency_ms" in df else float("nan"),
    }
    if len(set(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    if covered_mask.any():
        yt = df.loc[covered_mask, "gold"].astype(int).to_numpy()
        yp = np.array([1 if x == "TOXIC" else 0 for x in df.loc[covered_mask, "pred_label"]])
        p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
        metrics.update({"precision": float(p), "recall": float(r), "f1": float(f1)})
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tp = int(((yt == 1) & (yp == 1)).sum())
        metrics.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp, "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1)})
    else:
        metrics.update({"precision": 0.0, "recall": 0.0, "f1": 0.0, "tn": 0, "fp": 0, "fn": 0, "tp": 0, "fpr": 0.0, "fnr": 0.0})

    thresholds = np.linspace(0.0, 1.0, 51)
    curve_rows = []
    for th in thresholds:
        # Correct coverage-risk: ignore DEFER predictions.
        active = (df["pred_label"] != "DEFER") & (df["confidence"].astype(float) >= th)
        if active.any():
            yt = df.loc[active, "gold"].astype(int).to_numpy()
            yp = np.array([1 if x == "TOXIC" else 0 for x in df.loc[active, "pred_label"]])
            risk = 1.0 - f1_score(yt, yp, zero_division=0)
            cov = float(active.mean())
            err = float((yt != yp).mean())
        else:
            risk, cov, err = 0.0, 0.0, 0.0
        curve_rows.append({"threshold": float(th), "coverage": cov, "risk_1_minus_f1": float(risk), "error_rate": err})
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(out_dir / f"{prefix}_coverage_risk.csv", index=False)
    plt.figure(figsize=(6, 4))
    plt.plot(curve_df["coverage"], curve_df["risk_1_minus_f1"], marker="o", markersize=3)
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - F1)")
    plt.title(f"Coverage-Risk Curve ({prefix})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_coverage_risk.png", dpi=200)
    plt.close()
    with open(out_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def run_eval(judge: ToxicJudge, dataset: str, method: str, out_dir: Path, sample_n: Optional[int], seed: int, self_n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(dataset, sample_n, seed)
    rows = []
    for i, row in df.iterrows():
        print(f"[eval {method}] {i + 1}/{len(df)}", flush=True)
        pred = judge.predict(row["text"], method=method, self_n=self_n, seed=seed)
        rows.append({"text": row["text"], "gold": row["gold"], "pred_label": pred["label"], "category": pred["category"], "confidence": pred["confidence"], "toxic_score": pred["toxic_score"], "latency_ms": pred["latency_ms"], "valid_json": pred.get("valid_json", False), "method": pred["method"]})
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_dir / f"predictions_{method}.csv", index=False)
    metrics = evaluate_predictions(pred_df, out_dir, f"method_{method}")
    print(json.dumps(metrics, indent=2))


def run_robustness(judge: ToxicJudge, dataset: str, method: str, out_dir: Path, sample_n: Optional[int], seed: int, self_n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(dataset, sample_n, seed)
    rows = []
    for i, row in df.iterrows():
        print(f"[robustness {method}] {i + 1}/{len(df)}", flush=True)
        clean = judge.predict(row["text"], method=method, self_n=self_n, seed=seed)
        rows.append({"attack": "clean", "variant": "clean", "text": row["text"], "gold": row["gold"], "pred_label": clean["label"], "confidence": clean["confidence"], "toxic_score": clean["toxic_score"]})
        for attack_name, attacked_text in RobustnessSuite.apply_all(row["text"]).items():
            attacked = judge.predict(attacked_text, method=method, self_n=self_n, seed=seed)
            mitigated_text = normalize_text_for_mitigation(attacked_text)
            mitigated = judge.predict(mitigated_text, method=method, self_n=self_n, seed=seed)
            rows.append({"attack": attack_name, "variant": "attacked", "text": attacked_text, "gold": row["gold"], "pred_label": attacked["label"], "confidence": attacked["confidence"], "toxic_score": attacked["toxic_score"]})
            rows.append({"attack": attack_name, "variant": "mitigated", "text": mitigated_text, "gold": row["gold"], "pred_label": mitigated["label"], "confidence": mitigated["confidence"], "toxic_score": mitigated["toxic_score"]})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"robustness_{method}.csv", index=False)
    summary = []
    for attack in sorted(out["attack"].unique()):
        for variant in sorted(out[out["attack"] == attack]["variant"].unique()):
            part = out[(out["attack"] == attack) & (out["variant"] == variant)]
            covered = part[part["pred_label"] != "DEFER"]
            if len(covered) == 0:
                f1 = 0.0
            else:
                yp = [1 if x == "TOXIC" else 0 for x in covered["pred_label"]]
                f1 = f1_score(covered["gold"], yp, zero_division=0)
            summary.append({"attack": attack, "variant": variant, "f1": float(f1), "coverage": float((part["pred_label"] != "DEFER").mean())})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / f"robustness_summary_{method}.csv", index=False)
    plt.figure(figsize=(8, 4))
    for variant in ["clean", "attacked", "mitigated"]:
        part = summary_df[summary_df["variant"] == variant]
        if len(part) > 0:
            plt.plot(part["attack"], part["f1"], marker="o", label=variant)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("F1")
    plt.title(f"Robustness Before/After Mitigation ({method})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"robustness_{method}.png", dpi=200)
    plt.close()
    print(summary_df.to_string(index=False))


def run_consistency(judge: ToxicJudge, dataset: str, method: str, out_dir: Path, sample_n: Optional[int], seeds: List[int], self_n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(dataset, sample_n, seeds[0] if seeds else 42)
    records = []
    for seed in seeds:
        print(f"[consistency {method}] seed={seed}", flush=True)
        for idx, row in df.iterrows():
            pred = judge.predict(row["text"], method=method, self_n=self_n, seed=seed)
            records.append({"idx": idx, "seed": seed, "pred_label": pred["label"], "confidence": pred["confidence"]})
    out = pd.DataFrame(records)
    out.to_csv(out_dir / f"consistency_{method}.csv", index=False)
    consistencies = []
    for _, grp in out.groupby("idx"):
        labels = grp["pred_label"].tolist()
        major = max(set(labels), key=labels.count)
        consistencies.append(labels.count(major) / len(labels))
    result = {"mean_agreement": float(np.mean(consistencies)) if consistencies else 0.0, "std_agreement": float(np.std(consistencies)) if consistencies else 0.0, "n_items": int(len(consistencies)), "n_seeds": int(len(seeds))}
    with open(out_dir / f"consistency_{method}_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def print_demo(pred: dict) -> None:
    safe = {k: v for k, v in pred.items() if k != "raw_output"}
    print(json.dumps(safe, indent=2, ensure_ascii=False))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM-as-a-Judge toxicity moderation")
    p.add_argument("--model", type=str, required=True, help="HF model name or local path")
    p.add_argument("--device", type=str, default=None, help="cpu or cuda")
    p.add_argument("--use_4bit", action="store_true", help="Enable 4-bit loading when CUDA is available")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--defer_threshold", type=float, default=0.60)
    p.add_argument("--agreement_threshold", type=float, default=0.67)
    p.add_argument("--seed", type=int, default=42)
    sub = p.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--method", choices=["A", "B", "C"], required=True)
    demo.add_argument("--text", required=True)
    demo.add_argument("--self_n", type=int, default=5)
    ev = sub.add_parser("eval")
    ev.add_argument("--dataset", required=True)
    ev.add_argument("--method", choices=["A", "B", "C"], required=True)
    ev.add_argument("--sample_n", type=int, default=None)
    ev.add_argument("--self_n", type=int, default=5)
    ev.add_argument("--out_dir", required=True)
    rb = sub.add_parser("robustness")
    rb.add_argument("--dataset", required=True)
    rb.add_argument("--method", choices=["A", "B", "C"], required=True)
    rb.add_argument("--sample_n", type=int, default=None)
    rb.add_argument("--self_n", type=int, default=5)
    rb.add_argument("--out_dir", required=True)
    cs = sub.add_parser("consistency")
    cs.add_argument("--dataset", required=True)
    cs.add_argument("--method", choices=["A", "B", "C"], required=True)
    cs.add_argument("--sample_n", type=int, default=None)
    cs.add_argument("--self_n", type=int, default=5)
    cs.add_argument("--seeds", nargs="+", type=int, required=True)
    cs.add_argument("--out_dir", required=True)
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    cfg = JudgeConfig(model_name=args.model, max_new_tokens=args.max_new_tokens, device=args.device, use_4bit=args.use_4bit, defer_threshold=args.defer_threshold, agreement_threshold=args.agreement_threshold)
    try:
        judge = ToxicJudge(cfg)
    except Exception as e:
        msg = str(e)
        if "gated repo" in msg.lower() or "authorized list" in msg.lower() or "401" in msg or "403" in msg:
            raise SystemExit("Model access failed. If you are using Gemma, make sure you accepted access on Hugging Face and logged in.\n" f"Original error: {e}")
        raise
    if args.command == "demo":
        pred = judge.predict(args.text, method=args.method, self_n=args.self_n, seed=args.seed)
        print_demo(pred)
        return
    if args.command == "eval":
        run_eval(judge, args.dataset, args.method, Path(args.out_dir), args.sample_n, args.seed, args.self_n)
        return
    if args.command == "robustness":
        run_robustness(judge, args.dataset, args.method, Path(args.out_dir), args.sample_n, args.seed, args.self_n)
        return
    if args.command == "consistency":
        run_consistency(judge, args.dataset, args.method, Path(args.out_dir), args.sample_n, args.seeds, args.self_n)
        return


if __name__ == "__main__":
    main()
