import numpy as np


class LLLayer:

    def __init__(self, data: np.ndarray, coeffs: (float, float) = (2, 5)):
        self.step = 0
        self.data = data
        self.size = data.shape[1]
        self.weights = np.exp(np.random.rand(self.size, self.size) * 10) / 10000
        self.weights_upd = np.copy(self.weights)
        self.floor_fn = np.vectorize(lambda t: np.floor(t) if t >= 1 else 0)
        self.coeffs = coeffs
        print("Initialized ANN of size {}".format(self.size))
        print("            weights sum={}, min={}, max={}".format(
            self.weights.sum(), self.weights.min(), self.weights.max()
        ))

    def run_step(self, data) -> np.ndarray:
        y = self.weights @ data
        y = self.floor_fn(np.log(y))
        y = y - y.min()
        print("Step no {}:       y sum={}, min={}, max={}".format(self.step, y.sum(), y.min(), y.max()))

        self.weights_upd = self.weights + np.array(
            [(self.coeffs[0] * y - 1 / self.coeffs[1]) * w / w.sum() for w in self.weights])
        print("            weights sum={}, min={}, max={}".format(
                self.weights_upd.sum(), self.weights_upd.min(), self.weights_upd.max()
              ))
        diff = self.weights_upd - self.weights
        print("               diff sum={}, min={}, max={}".format(diff.sum(), diff.min(), diff.max()))
        return y

    def run(self, batch=range(1), iterations: int = 1) -> np.ndarray:
        data = np.copy(self.data[batch])
        self.step = 0
        for it in range(iterations):
            self.step += 1
            data_new = np.copy(data)
            for idx, d in enumerate(data):
                data_new[idx] = self.run_step(d)
                self.weights = np.copy(self.weights_upd)
            data = data_new
        return data



