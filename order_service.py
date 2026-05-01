def format_order(items, usage=None):
    tokens = usage.total_tokens if usage else 0
    return "\n".join(items)+f"\nТокены: {tokens}"
