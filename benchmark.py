"""Compara inferencia local con y sin caché LRU de tokenización."""

import csv
import json
import math
import platform
import random
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parent
SEED = 2026
BLOCKS = 40
TARGET_MS = 1.0
PATTERN = re.compile(r"\b\w\w+\b")


def tokenize(text):
    return Counter(PATTERN.findall(text))


def run():
    with (ROOT / "datos/datos_depurados_y_particiones.csv").open(
        encoding="utf-8-sig", newline=""
    ) as source:
        rows = list(csv.DictReader(source))
    counts = defaultdict(Counter)
    classes = Counter()
    for row in rows:
        if row["fold"] != "1":
            counts[row["label"]].update(tokenize(row["text"]))
            classes[row["label"]] += 1
    vocabulary = set().union(*counts.values())
    labels = sorted(classes)
    coefficients = {}
    for label in labels:
        denominator = counts[label].total() + len(vocabulary)
        coefficients[label] = {
            word: math.log((counts[label][word] + 1) / denominator)
            for word in vocabulary
        }
    priors = {label: math.log(classes[label] / classes.total()) for label in labels}
    cached_tokens = lru_cache(maxsize=128)(tokenize)

    def infer(text, tokenizer):
        features = tokenizer(text)
        scores = [
            priors[label] + sum(
                count * coefficients[label][word]
                for word, count in features.items()
                if word in vocabulary
            )
            for label in labels
        ]
        return labels[max(range(len(labels)), key=scores.__getitem__)]

    profiles = list(dict.fromkeys(row["text"] for row in rows if row["fold"] == "1"))
    randomizer = random.Random(SEED)
    records, block_deltas = [], []
    for block in range(1, BLOCKS + 1):
        selected = randomizer.sample(range(len(profiles)), 128)
        requests = selected + selected
        randomizer.shuffle(requests)
        cached_tokens.cache_clear()
        pairs = []
        for position, index in enumerate(requests, 1):
            text = profiles[index]
            order = [("base", tokenize), ("cache", cached_tokens)]
            if randomizer.randrange(2):
                order.reverse()
            timings, predictions = {}, {}
            for name, tokenizer in order:
                start = perf_counter_ns()
                predictions[name] = infer(text, tokenizer)
                timings[name] = perf_counter_ns() - start
            assert predictions["base"] == predictions["cache"]
            pairs.append(timings["base"] - timings["cache"])
            records.append([block, position, index, order[0][0],
                            timings["base"], timings["cache"]])
        block_deltas.append(median(pairs))
    with (ROOT / "latencias.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["bloque", "posicion", "perfil", "primero", "base_ns", "cache_ns"])
        writer.writerows(records)

    def percentiles(column):
        values = [record[column] / 1e6 for record in records]
        cuts = quantiles(values, n=100, method="inclusive")
        return {"mediana_ms": median(values), "p95_ms": cuts[94], "p99_ms": cuts[98]}

    positive = sum(value > 0 for value in block_deltas)
    negative = sum(value < 0 for value in block_deltas)
    nonzero = positive + negative
    probability = sum(math.comb(nonzero, k) for k in range(positive, nonzero + 1))
    return {
        "python": platform.python_version(), "sistema": platform.platform(),
        "cpu": platform.processor(), "bloques": BLOCKS, "pares": len(records),
        "base": percentiles(4), "cache": percentiles(5), "umbral_ms": TARGET_MS,
        "p_signos_unilateral": probability / 2**nonzero,
        "bloques_favorables": positive, "bloques_desfavorables": negative,
        "efecto_mediana_delta_ms": median(block_deltas) / 1e6,
    }


if __name__ == "__main__":
    (ROOT / "rendimiento.json").write_text(
        json.dumps(run(), indent=2) + "\n", encoding="utf-8"
    )
