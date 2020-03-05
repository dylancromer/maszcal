import maszcal.model


def select_model(data: object, model: str, cm_relation: bool, emulation: bool, stacked: bool) -> object:
    model_choice = (model, cm_relation, emulation, stacked)

    available_models = {
        ('nfw', False, False, False): maszcal.model.SingleMass,
        ('nfw', True, False, False): maszcal.model.SingleMass,
    }

    backends = {
        ('nfw', False, False, False): maszcal.lensing.SingleMassNfwLensingSignal,
        ('nfw', True, False, False): maszcal.lensing.SingleMassNfwLensingSignal,
    }

    def _get_model_class(model_choice):
        try:
            return available_models[model_choice]
        except KeyError:
            raise ValueError('Invalid model selected.')

    def _get_backend(model_choice):
        try:
            return backends[model_choice]
        except KeyError:
            raise ValueError('Invalid backend selected.')

    model = _get_model_class(model_choice)
    backend = _get_backend(model_choice)

    return model(
        lensing_signal_class=backend,
        cm_relation=cm_relation,
    )
