import numpy as np


class LLConfig:

    def __init__(self, degrad_coef: float, inc_coef: float):
        self.degrad_coef = degrad_coef
        self.inc_coef = inc_coef


class LLNeuron:

    def __init__(self, no_inputs: int, config: LLConfig):
        self.weights = np.random.rand(no_inputs) / 10
        self.weights_upd = np.zeros(no_inputs)
        self.transform = np.vectorize(lambda t: np.floor(t) if t >= 1 else 0)
        self.config = config

    def proc(self, inp: [int]) -> np.ndarray:
        self.weights_upd = self.weights
        xw = self.weights_upd * inp
        raw_out = np.sum(xw)
        log_out = np.log(raw_out)
        out = self.transform(log_out)
        self.weights_upd /= self.config.degrad_coef
        if out > 1:
            self.weights_upd *= inp * self.config.degrad_coef * self.config.degrad_coef
        else:
            # TODO: Re-normalize
            pass
        return out
