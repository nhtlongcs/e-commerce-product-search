def calc_weights(scores, power=3):
    """
    Tính trọng số cho ensemble từ danh sách scores.
    
    Args:
        scores (list[float]): danh sách score các mô hình
        power (int): số mũ dùng để nhấn mạnh mô hình mạnh hơn
    
    Returns:
        list[float]: trọng số đã chuẩn hóa
    """
    powered = [s ** power for s in scores]
    total = sum(powered)
    weights = [p / total for p in powered]
    return weights

# Ví dụ:
scores = [0.8422, 0.8727, 0.8760]
weights = calc_weights(scores, power=3)
print("Weights:", weights)