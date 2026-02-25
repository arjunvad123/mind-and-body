from processor import process

# Test data to verify behavior
test_data = [
    {"id": 1, "type": "A", "value": 25, "active": True},   # 25 * 2 = 50
    {"id": 2, "type": "B", "value": 30, "active": True},   # 30 + 10 = 40
    {"id": 3, "type": "C", "value": 8, "active": True},    # 8^2 = 64
    {"id": 4, "type": "A", "value": 60, "active": True},   # 60 * 2 = 100 (capped)
    {"id": 5, "type": "B", "value": 95, "active": True},   # 95 + 10 = 100 (capped)
    {"id": 6, "type": "C", "value": 15, "active": True},   # 15^2 = 100 (capped)
    {"id": 7, "type": "A", "value": 20, "active": False},  # inactive, result = 0
    {"id": 8, "type": "B", "value": 50, "active": False},  # inactive, result = 0
]

result = process(test_data)
print("Original result:")
print(result)