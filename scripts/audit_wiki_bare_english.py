from __future__ import annotations

import collections
import re
from pathlib import Path

WIKI = Path("wiki")
PROTECTED = re.compile(r"(`[^`]*`|!?\[[^\]]*\]\([^\n)]*\))")
TOKEN = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9'-]*(?![A-Za-z0-9_])")

# Proper names, mathematical notation, protocols, and short abbreviations that are
# acceptable when the surrounding Wiki already provides a dedicated explanation.
ALLOW = {
    "AASSR", "APASSR", "HTTP", "HTTPS", "HTML", "URL", "URI", "API", "JSON", "CSV",
    "Git", "GitHub", "Python", "PyTorch", "Torch", "CUDA", "GPU", "CPU", "JAX", "WSL",
    "TF32", "NIST", "DreamerV3", "Dreamer", "DQN", "GRU", "MDP", "POMDP", "RL", "TD",
    "ASEQ", "ASeq", "Q", "Bellman", "Markov", "Softmax", "Adam", "ReLU", "MSE", "MAE",
    "KL", "ECE", "Brier", "AUROC", "F1", "ECE", "ECE-style", "CI", "PR", "SHA", "SHA-256",
    "Mermaid", "README", "Wiki", "Linux", "Windows", "PowerShell", "YAML", "Markdown",
    "GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "CSRF", "JWT", "ID", "IDs",
    "L0", "L1", "L2", "L3", "L4", "L5", "H0", "H1", "RQ1", "RQ2", "RQ3", "RQ4", "RQ5",
    "RQ6", "RQ7", "RQ8", "RQ9", "OFF", "ON", "OK", "TODO", "TODOs", "NaN", "inf", "argmax",
    "max", "min", "mean", "std", "log", "exp", "tanh", "sigmoid", "ReLU", "one-hot", "top-k",
    "proof", "seed", "seeds", "route", "profile", "object", "token", "browse", "login", "request",
    "web", "pentest", "HTTP", "status", "main", "README", "manifest", "workflow", "workflows",
    "v1", "v2", "v3", "v4", "v5", "v6", "v0", "CUDA", "float32", "float64", "bfloat16",
}

# Words that are ordinary English glue inside exact upstream names or equations and
# are not useful as standalone jargon audit findings.
ALLOW_LOWER = {w.lower() for w in ALLOW} | {
    "a", "an", "the", "and", "or", "not", "vs", "to", "of", "for", "from", "with", "without",
    "in", "on", "off", "by", "per", "if", "else", "true", "false", "none", "same", "after", "before",
}


def visible_text(path: Path):
    in_fence = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # Existing links and inline code are intentional/defined or exact identifiers.
        parts = PROTECTED.split(line)
        plain = "".join(part for i, part in enumerate(parts) if i % 2 == 0)
        yield lineno, plain


def main() -> None:
    counts: collections.Counter[str] = collections.Counter()
    examples: dict[str, list[str]] = collections.defaultdict(list)
    for path in sorted(WIKI.glob("*.md")):
        if path.name == "README.md":
            continue
        for lineno, text in visible_text(path):
            for match in TOKEN.finditer(text):
                word = match.group(0)
                if word.lower() in ALLOW_LOWER:
                    continue
                counts[word] += 1
                if len(examples[word]) < 3:
                    examples[word].append(f"{path.name}:{lineno}")

    print(f"Unexpected bare-English audit: {sum(counts.values())} tokens, {len(counts)} unique")
    for word, count in counts.most_common(250):
        print(f"{count:4d}  {word:32s}  {' '.join(examples[word])}")


if __name__ == "__main__":
    main()
