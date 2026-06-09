import torch
from .. import ddafx
from .filters import rbj


def get_5BandParametricEQ(samplerate=44100):
    eq = ddafx.DDAFXChain(
        rbj.LowShelf(
            ranges_parameters=[[20, 200], [-10, 10], [0.1, 3]], samplerate=samplerate
        ),
        rbj.Peak(
            ranges_parameters=[[40, 300], [-10, 10], [0.1, 3]], samplerate=samplerate
        ),
        rbj.Peak(
            ranges_parameters=[[200, 3000], [-10, 10], [0.1, 3]], samplerate=samplerate
        ),
        rbj.Peak(
            ranges_parameters=[[1000, 6000], [-10, 10], [0.1, 3]], samplerate=samplerate
        ),
        rbj.HighShelf(
            ranges_parameters=[[2000, 20000], [-10, 10], [0.1, 3]], samplerate=samplerate
        ),
        samplerate=samplerate,
    )

    return eq

def get_7BandParametricEQ(samplerate=44100):
    eq = ddafx.DDAFXChain(
        rbj.LowPass(
            ranges_parameters=[[20, 100], [0.1, 3]], samplerate=samplerate
        ),
        rbj.LowShelf(
            ranges_parameters=[[20, 200], [-10, 10], [0.1, 3]], samplerate=samplerate
        ),
        rbj.Peak(
            ranges_parameters=[[40, 300], [-10, 10], [0.1, 3]], samplerate=samplerate
        ),
        rbj.Peak(
            ranges_parameters=[[200, 3000], [-10, 10], [0.1, 3]], samplerate=samplerate
        ),
        rbj.Peak(
            ranges_parameters=[[1000, 6000], [-10, 10], [0.1, 3]], samplerate=samplerate
        ),
        rbj.HighShelf(
            ranges_parameters=[[2000, 20000], [-10, 10], [0.1, 3]], samplerate=samplerate
        ),
        rbj.HighPass(
            ranges_parameters=[[8000, 22000], [0.1, 3]], samplerate=samplerate
        ),
        samplerate=samplerate,
    )

    return eq