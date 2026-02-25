import re

def parse_prices(html):
    """Extract product names and prices from HTML."""
    products = []
    # Fix Bug 1: Add re.DOTALL flag to handle multi-line HTML
    pattern = r'<div class="product">.*?<span class="name">(.*?)</span>.*?<span class="price">\$(.*?)</span>'
    matches = re.findall(pattern, html, re.DOTALL)
    for name, price in matches:
        # Fix Bug 2: Strip whitespace from price before converting to float
        products.append({"name": name, "price": float(price.strip())})
    # Fix Bug 3: Sort by price ascending (remove reverse=True)
    products.sort(key=lambda x: x["price"])
    return products

def find_cheapest(html, n=3):
    """Find the n cheapest products."""
    products = parse_prices(html)
    # Bug 4 is fixed by fixing Bug 3 - now returns first n (cheapest)
    return products[:n]

def format_report(products):
    """Format products into a report string."""
    lines = []
    for i, p in enumerate(products):
        # Fix Bug 5: Start numbering at 1 instead of 0
        lines.append(f"{i+1}. {p['name']} - ${p['price']:.2f}")
    return "\n".join(lines)