import numpy as np
from . import learn_fns as lf


def inspect_1starg_shape(fn):
    def wrapper(*args, **kwargs):
        print("Function {} received data shaped as {}".format(fn.__name__, args[1].shape))
        return fn(*args, **kwargs)
    return wrapper


class LLLayer:

    _floor_fn = np.vectorize(lambda t: np.floor(t) if t >= 1 else 0)

    def __init__(self, input_dim: int, output_dim: int = -1,
                 coeffs: (float, float) = (2, 5),
                 learn_fn: lf.LearnFn = lf.linear):
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim > 0 else input_dim
        self.weights = np.random.rand(self.output_dim, self.input_dim)
        self.weights_upd = np.copy(self.weights)
        self.coeffs = coeffs
        self.learn_fn = learn_fn

    def _gen_output(self, data: np.ndarray) -> np.ndarray:
        y = self.weights @ data
        y = self._floor_fn(np.log(y + 0.000001))
        y = y - y.min()
        return y

    def _update_weights(self, y: np.ndarray):
        a = self.coeffs[0]
        b = self.coeffs[1]
        y_factors = self.learn_fn(y, a, b)
        wu = [wc * y_factors[idx] / wc.sum() for idx, wc in enumerate(self.weights)]
        self.weights_upd = np.array(wu)

    def proc(self, data: np.ndarray) -> np.ndarray:
        return self._gen_output(data)

    def learn(self, data: np.ndarray) -> np.ndarray:
        y = self.proc(data)
        self._update_weights(y)
        return y

    def confirm(self):
        self.weights = self.weights_upd


class LLNetwork:

    def __init__(self, input_dim: int,
                 coeffs: (float, float) = (2, 5),
                 learn_fn: lf.LearnFn = lf.linear):
        self.layers = []
        self.input_dim = input_dim
        self.coeffs = coeffs
        self.learn_fn = learn_fn

    def add_layer(self, output_dim: int):
        input_dim = self.input_dim if len(self.layers) == 0 else self.layers[-1].output_dim
        layer = LLLayer(input_dim, output_dim, self.coeffs, self.learn_fn)
        self.layers.append(layer)

    def proc(self, ndata: np.ndarray) -> np.ndarray:
        results = []
        for no, data in enumerate(ndata):
            result = np.copy(data)
            for layer in self.layers:
                result = layer.learn(result)
        return np.ndarray(results)

    def learn(self, ndata: np.ndarray, outputs: np.ndarray) -> np.ndarray:
        results = []
        for no, data in enumerate(ndata):
            result = np.copy(data)
            for layer in self.layers:
                result = layer.learn(result)
            diff = np.ceil(result / (result + 0.0001)) - outputs[no]
            success = (np.abs(diff).sum() == 0)
            if success:
                for layer in self.layers:
                    layer.confirm()
            print("Step {}, success: {}".format(no, success))
            results.append(result)
        return results
