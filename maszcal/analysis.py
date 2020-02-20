import maszcal.lensing


def select_model(data: object, model: str, cm_relation: bool, emulation: bool, stacked: bool) -> object:
    model_choice = (model, cm_relation, emulation, stacked)

    def _get_model_class(model_choice):
        available_models = {
            ('nfw', False, False, False): maszcal.lensing.SingleMassNfwLensingSignal,
            ('baryon', False, False, False): maszcal.lensing.SingleBaryonLensingSignal,
        }

        try:
            return available_models[model_choice]
        except KeyError:
            raise ValueError('Invalid model selected.')

    model = _get_model_class(model_choice)

    return model(
        redshift=data.redshifts,
    )
