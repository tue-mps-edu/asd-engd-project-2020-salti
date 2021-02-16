def find_digits(i):
    d1 = i
    d2 = 0
    d3 = 0
    d4 = 0
    if i >= 10:
        d1 = i % 10
        d2 = int((i - d1) / 10)
        d3 = 0
        d4 = 0
        if i >= 100:
            d1 = i % 10
            d2 = int(((i - d1) / 10) % 10)
            d3 = int((((i - d1) / 10) - d2) / 10)
            d4 = 0
            if i >= 1000:
                d1 = i % 10
                d2 = int(((i - d1) / 10) % 10)
                d3 = int(((((i - d1) / 10) - d2) / 10) % 10)
                d4 = int((((((i - d1) / 10) - d2) / 10) - d3) / 10)
    return d4,d3,d2,d1