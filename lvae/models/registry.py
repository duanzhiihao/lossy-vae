_all_models = dict()


def register_model(func):
    name = func.__name__
    if name in _all_models:
        msg = f'Warning: model function *{name}* is multiply defined.'
        print(f'\u001b[93m' + msg + '\u001b[0m')
    _all_models[name] = func
    return func


def get_model(name, *args, **kwargs):
    model_func = _all_models[name]
    return model_func(*args, **kwargs)
