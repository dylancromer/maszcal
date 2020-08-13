def log_gaussian_shape(model, data, fisher):
    diff = model - data
    return -(diff@fisher@diff.T) / 2
