import re

ALLOWED_FEATURES = [
    "rank_u",
    "rank_d",
    "depth",
    "in_degree",
    "out_degree",
    "fork",
    "comm_weight",
]

STRUCTURAL_FEATURES = {"depth", "in_degree", "out_degree", "fork", "comm_weight"}


def parse_feature_names(llm_response: str):
    text = llm_response.strip().lower()

    found = []
    for feat in ALLOWED_FEATURES:
        pattern = r"\b" + re.escape(feat.lower()) + r"\b"
        if re.search(pattern, text):
            found.append(feat)

    # remove duplicates preserving order
    deduped = []
    seen = set()
    for f in found:
        if f not in seen:
            deduped.append(f)
            seen.add(f)

    # enforcng minimum richness
    if len(deduped) < 3:
        return ["rank_u", "depth", "fork", "comm_weight"]

    # enforce at least one structural feature
    if not any(f in STRUCTURAL_FEATURES for f in deduped):
        return ["rank_u", "depth", "fork", "comm_weight"]

    # avoid pure HEFT like behavior
    if deduped == ["rank_u"] or deduped == ["rank_u", "rank_d"]:
        return ["rank_u", "depth", "fork", "comm_weight"]

    return deduped[:5]