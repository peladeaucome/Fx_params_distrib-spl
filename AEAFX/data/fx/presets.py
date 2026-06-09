from . import equalizer, compressor, main


def get_5BandParamEQ(samplerate):
    chain = main.DAFx_Series(
        equalizer.LowShelf([[20, 200], [-10, 10], [0.1, 3]], samplerate),
        equalizer.Peak([[40, 300], [-10, 10], [0.1, 3]], samplerate),
        equalizer.Peak([[200, 3000], [-10, 10], [0.1, 3]], samplerate),
        equalizer.Peak([[1000, 6000], [-10, 10], [0.1, 3]], samplerate),
        equalizer.HighShelf([[2000, 20000], [-10, 10], [0.1, 3]], samplerate),
        samplerate=samplerate,
    )
    return chain