
def log_parameters(experiment, **kwargs):
    params = {}
    for key, value in kwargs.items():
        params[key] = value

    experiment.log_parameters(params)