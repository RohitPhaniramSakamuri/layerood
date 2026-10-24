"""Official CLINC150 data adapter with disjoint text identities and pinned provenance."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from .artifacts import text_hash, write_json


def _fetch(url):
    with urlopen(Request(url, headers={"User-Agent": "LayerOOD-research/0.1"}), timeout=60) as response:
        data = response.read(32 * 1024 * 1024 + 1)
    if len(data) > 32 * 1024 * 1024:
        raise ValueError("Source response exceeds 32 MiB")
    return data


def prepare_clinc(output, profile="quick", protocol="official", seed=42, id_fraction=.75, source=None):
    profiles = {"mini": (5,8,4,5,4,20), "quick": (10,20,10,10,10,100), "full": (150,60,40,20,30,1000)}
    if profile not in profiles or protocol not in ("official", "heldout-intents") or not 0 < id_fraction < 1:
        raise ValueError("Invalid profile, protocol or ID fraction")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Dataset directory already exists: {output}; choose a fresh path")
    if source:
        raw = Path(source).read_bytes()
        url, revision = str(Path(source).resolve()), "user-provided-local-file"
    else:
        commit = json.loads(_fetch("https://api.github.com/repos/clinc/oos-eval/commits/master"))["sha"]
        url = f"https://raw.githubusercontent.com/clinc/oos-eval/{commit}/data/data_full.json"
        raw, revision = _fetch(url), commit
    data = json.loads(raw)
    rng = np.random.default_rng(seed)
    intents = sorted({label for _, label in data["train"]})
    shuffled = rng.permutation(intents).tolist()
    max_intents, nfit, nref, ncal, ntest, nood = profiles[profile]
    if protocol == "heldout-intents":
        nknown = max(2, min(len(intents)-1, int(len(intents)*id_fraction)))
        all_known, unknown = set(shuffled[:nknown]), set(shuffled[nknown:])
    else:
        all_known, unknown = set(intents), set()
    known = [intent for intent in shuffled if intent in all_known][:max_intents]
    splits = {"train": [], "reference": [], "calibration": [], "evaluation": []}
    seen, duplicates = set(), defaultdict(int)
    def add(split, text, label, identifier, ood=False, group="id"):
        fingerprint = text_hash(text)
        if fingerprint in seen:
            duplicates[split] += 1
            return False
        seen.add(fingerprint)
        splits[split].append({"id": identifier, "text": text, "label": label,
                              "is_ood": ood, "group": group, "split": split})
        return True
    indexed = {}
    for split in ("train", "val", "test"):
        indexed[split] = defaultdict(list)
        for i, (text, label) in enumerate(data[split]):
            indexed[split][label].append((i,text))
    for label in known:
        training = indexed["train"][label]
        order = rng.permutation(len(training))
        accepted = 0
        for j in order:
            i, text = training[j]
            split = "train" if accepted < nfit else "reference"
            if add(split,text,label,f"clinc-train-{i}"):
                accepted += 1
            if accepted >= nfit+nref:
                break
        for original, destination, cap in (("val","calibration",ncal),("test","evaluation",ntest)):
            items = indexed[original][label]
            accepted = 0
            for j in rng.permutation(len(items)):
                i,text = items[j]
                if add(destination,text,label,f"clinc-{original}-{i}"):
                    accepted += 1
                if accepted >= cap:
                    break
    if protocol == "official":
        for i in rng.permutation(len(data["oos_test"])):
            text,label = data["oos_test"][i]
            add("evaluation",text,label,f"clinc-oos-test-{i}",True,"official_oos")
            if sum(r["is_ood"] for r in splits["evaluation"]) >= nood:
                break
    else:
        unknown_selected = [i for i in shuffled if i in unknown][:max_intents]
        for label in unknown_selected:
            accepted = 0
            for j in rng.permutation(len(indexed["test"][label])):
                i,text = indexed["test"][label][j]
                if add("evaluation",text,label,f"clinc-test-{i}",True,"heldout_intent"):
                    accepted += 1
                if accepted >= ntest:
                    break
    if len(splits["calibration"]) < 20:
        raise ValueError("Deduplication left fewer than 20 calibration texts; use a larger profile")
    manifest = {"dataset":"CLINC150 data_full.json", "source_url":url, "source_revision":revision,
                "source_sha256":hashlib.sha256(raw).hexdigest(), "license":"CC BY 3.0",
                "license_url":"https://github.com/clinc/oos-eval/blob/master/LICENSE",
                "citation":"Larson et al. (EMNLP-IJCNLP 2019), https://aclanthology.org/D19-1131/",
                "profile":profile,"protocol":protocol,"seed":seed,"known_intents":known,
                "evaluated_ood_intents":sorted({r["label"] for r in splits["evaluation"] if r["is_ood"]}),
                "heldout_intents":sorted(unknown), "counts":{k:len(v) for k,v in splits.items()},
                "duplicates_removed":dict(duplicates), "ood_training_used":False,
                "scope":"Full profile uses official text pools with a custom fit/reference split and deduplication. Mini/quick are subsets, not the published full benchmark."}
    output.mkdir(parents=True)
    (output/"source_data_full.json").write_bytes(raw)
    for name,rows in splits.items():
        with (output/(name+".jsonl")).open("w") as f:
            for row in rows:
                f.write(json.dumps(row,ensure_ascii=False)+"\n")
    write_json(output/"manifest.json",manifest)
    return manifest
