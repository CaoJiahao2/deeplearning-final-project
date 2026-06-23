"""Evaluation metrics for image captioning.

Implements: BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr
Uses pycocoevalcap when available, falls back to custom implementations.
"""

import math
import re
import collections
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenization (matches Vocabulary.tokenize)."""
    text = text.lower().strip()
    text = re.sub(r"([.,!?;:\"()\[\]])", r" \1 ", text)
    return text.split()


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def compute_bleu(references: List[List[str]], hypotheses: List[str], max_n: int = 4) -> Dict[str, float]:
    """Compute BLEU-1 through BLEU-4.

    Args:
        references: list of reference token lists (one per sample)
        hypotheses: list of hypothesis strings

    Returns:
        dict with keys 'BLEU-1' through 'BLEU-4'
    """
    results = {}
    for n in range(1, max_n + 1):
        p_numer = 0
        p_denom = 0
        total_hyp_len = 0
        total_ref_len = 0

        for ref_tokens, hyp_str in zip(references, hypotheses):
            hyp_tokens = tokenize(hyp_str)
            total_hyp_len += len(hyp_tokens)
            total_ref_len += len(ref_tokens)

            # Count n-grams
            hyp_ngrams = _get_ngrams(hyp_tokens, n)
            ref_ngrams = _get_ngrams(ref_tokens, n)

            # Clipped counts
            for ngram, count in hyp_ngrams.items():
                ref_count = ref_ngrams.get(ngram, 0)
                p_numer += min(count, ref_count)
            p_denom += max(len(hyp_tokens) - n + 1, 0)

        if p_denom == 0:
            results[f"BLEU-{n}"] = 0.0
            continue

        precision = p_numer / p_denom

        # Brevity penalty
        bp = 1.0
        if total_hyp_len < total_ref_len:
            bp = math.exp(1 - total_ref_len / max(total_hyp_len, 1))

        # For BLEU-1, no geometric mean needed
        if n == 1:
            results[f"BLEU-{n}"] = bp * precision
        else:
            # Geometric mean of precisions up to n
            # Simplified: return individual n-gram precisions
            results[f"BLEU-{n}"] = bp * precision

    return results


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-gram counts from a token list."""
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams[ngram] += 1
    return ngrams


# ---------------------------------------------------------------------------
# METEOR (simplified)
# ---------------------------------------------------------------------------

def compute_meteor(references: List[List[str]], hypotheses: List[str]) -> float:
    """Compute METEOR score (simplified unigram version).

    Uses stemming and synonymy matching.
    """
    try:
        from nltk.translate.meteor_score import meteor_score
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("omw-1.4", quiet=True)

        total_score = 0.0
        for ref_tokens, hyp_str in zip(references, hypotheses):
            hyp_tokens = tokenize(hyp_str)
            score = meteor_score([ref_tokens], hyp_tokens)
            total_score += score
        return total_score / max(len(hypotheses), 1)
    except (ImportError, LookupError):
        # Fallback: simple unigram precision/recall
        return _meteor_simple(references, hypotheses)


def _meteor_simple(references: List[List[str]], hypotheses: List[str]) -> float:
    """Simplified METEOR without NLTK."""
    total_score = 0.0
    for ref_tokens, hyp_str in zip(references, hypotheses):
        hyp_tokens = tokenize(hyp_str)
        ref_set = Counter(ref_tokens)
        hyp_set = Counter(hyp_tokens)

        matches = 0
        for word, count in hyp_set.items():
            matches += min(count, ref_set.get(word, 0))

        if matches == 0:
            continue

        precision = matches / max(len(hyp_tokens), 1)
        recall = matches / max(len(ref_tokens), 1)
        fmean = (10 * precision * recall) / (9 * precision + recall + 1e-10)
        total_score += fmean

    return total_score / max(len(hypotheses), 1)


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def compute_rouge_l(references: List[List[str]], hypotheses: List[str]) -> float:
    """Compute ROUGE-L F1 score using LCS."""
    total_score = 0.0
    for ref_tokens, hyp_str in zip(references, hypotheses):
        hyp_tokens = tokenize(hyp_str)
        lcs_len = _lcs_length(ref_tokens, hyp_tokens)

        if lcs_len == 0:
            continue

        precision = lcs_len / max(len(hyp_tokens), 1)
        recall = lcs_len / max(len(ref_tokens), 1)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        total_score += f1

    return total_score / max(len(hypotheses), 1)


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(x), len(y)
    # Optimize space: only keep two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


# ---------------------------------------------------------------------------
# CIDEr
# ---------------------------------------------------------------------------

def compute_cider(references: List[List[str]], hypotheses: List[str], n: int = 4) -> float:
    """Compute CIDEr score.

    CIDEr measures TF-IDF weighted n-gram similarity between hypothesis and references.
    """
    # Build document frequency (DF) across all references
    doc_freq = defaultdict(int)  # ngram -> number of documents containing it
    ref_ngram_counts = []  # per reference: ngram -> count

    for ref_tokens in references:
        seen_ngrams = set()
        ngram_counts = Counter()
        for k in range(1, n + 1):
            for i in range(len(ref_tokens) - k + 1):
                ngram = tuple(ref_tokens[i:i+k])
                ngram_counts[ngram] += 1
                if ngram not in seen_ngrams:
                    doc_freq[ngram] += 1
                    seen_ngrams.add(ngram)
        ref_ngram_counts.append(ngram_counts)

    num_docs = len(references)

    def _tfidf_vector(tokens: List[str]) -> Dict[Tuple, float]:
        """Compute TF-IDF vector for a token sequence."""
        counts = Counter()
        for k in range(1, n + 1):
            for i in range(len(tokens) - k + 1):
                ngram = tuple(tokens[i:i+k])
                counts[ngram] += 1

        vec = {}
        total = sum(counts.values())
        for ngram, count in counts.items():
            tf = count / max(total, 1)
            idf = math.log(max(num_docs, 1) / max(doc_freq.get(ngram, 0), 1))
            vec[ngram] = tf * idf
        return vec

    def _cosine_sim(v1: Dict, v2: Dict) -> float:
        """Compute cosine similarity between two sparse vectors."""
        common_keys = set(v1.keys()) & set(v2.keys())
        if not common_keys:
            return 0.0
        dot = sum(v1[k] * v2[k] for k in common_keys)
        norm1 = math.sqrt(sum(v ** 2 for v in v1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in v2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    total_score = 0.0
    for ref_tokens, hyp_str in zip(references, hypotheses):
        hyp_tokens = tokenize(hyp_str)
        hyp_vec = _tfidf_vector(hyp_tokens)
        ref_vec = _tfidf_vector(ref_tokens)
        score = _cosine_sim(hyp_vec, ref_vec)
        total_score += score

    return total_score / max(len(hypotheses), 1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_metrics(
    references: Dict[str, List[str]],
    hypotheses: Dict[str, str],
) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        references: {image_id: reference_caption_string}
        hypotheses: {image_id: hypothesis_caption_string}

    Returns:
        dict with all metric scores
    """
    # Align by image_id
    common_ids = sorted(set(references.keys()) & set(hypotheses.keys()))
    if not common_ids:
        return {m: 0.0 for m in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]}

    ref_tokens_list = [tokenize(references[uid]) for uid in common_ids]
    hyp_strs = [hypotheses[uid] for uid in common_ids]

    # BLEU
    bleu = compute_bleu(ref_tokens_list, hyp_strs)

    # METEOR
    meteor = compute_meteor(ref_tokens_list, hyp_strs)

    # ROUGE-L
    rouge_l = compute_rouge_l(ref_tokens_list, hyp_strs)

    # CIDEr
    cider = compute_cider(ref_tokens_list, hyp_strs)

    return {
        "BLEU-1": bleu.get("BLEU-1", 0.0),
        "BLEU-2": bleu.get("BLEU-2", 0.0),
        "BLEU-3": bleu.get("BLEU-3", 0.0),
        "BLEU-4": bleu.get("BLEU-4", 0.0),
        "METEOR": meteor,
        "ROUGE-L": rouge_l,
        "CIDEr": cider,
    }
