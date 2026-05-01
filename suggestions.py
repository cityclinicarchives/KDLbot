def get_suggestions(query, price_list):
    q=query.lower()
    return [i.name for i in price_list if q in i.name.lower()][:5]
