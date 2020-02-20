import maszcal.lensing


def select_model(data: object, model: str, cm_relation: bool, emulation: bool, stacked: bool) -> object:
    return maszcal.lensing.SingleMassNfwLensingSignal(
        redshift=data.redshifts,
    )
