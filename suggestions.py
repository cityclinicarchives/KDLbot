from rapidfuzz import fuzz


def get_suggestions(query: str, price_list, limit: int = 5) -> list[str]:
    query = str(query or "").strip().lower()
    if len(query) < 2:
        return []

    scored = []
    for item in price_list:
        name = str(getattr(item, "name", "") or "")
        short_name = str(getattr(item, "short_name", "") or "")
        haystack = f"{name} {short_name}".lower()

        if query in haystack:
            score = 100
        else:
            score = fuzz.WRatio(query, haystack)

        if score >= 70:
            scored.append((score, name))

    scored.sort(key=lambda x: (-x[0], x[1]))
    results = []
    seen = set()
    for _, name in scored:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            results.append(name)
        if len(results) >= limit:
            break
    return results
