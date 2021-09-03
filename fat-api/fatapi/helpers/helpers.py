def keep_cols(data, cols):
    return data[:, cols]

def not_in_range(data, list):
        return any((ind >= data or ind < 0) for ind in list)