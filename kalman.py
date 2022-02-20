import logging
import numpy as np


class KalmanFilter:
    def __init__(self):
        pass

    def predict(self, t1, t2):
        return np.add(t1, t2)

    def measurement(self, x: float, mu: float, sd: float):
        try:
            coef = np.divide(1, np.sqrt(np.prod([2.0, sd, np.pi])))
        except ZeroDivisionError:
            logging.error("Division by zero in [timestep] for [coef]")
            exit(1)

        try:
            exp = np.exp(np.prod([-0.5, np.divide(np.subtract(x, mu), sd)]))
        except ZeroDivisionError:
            logging.error("Division by zero in [timestep] for [exp]")
            exit(1)

        return coef * exp

    def update(self, mu1, mu2, sd1, sd2):
        try:
            mu = np.divide(np.add(np.multiply(mu1, sd1),
                                  np.multiply(mu2, sd2)),
                           np.add(sd1, sd2))
        except ZeroDivisionError:
            logging.error("Division by zero in [update] for [mu]")
            exit(1)

        try:
            sd = np.divide(np.add(np.multiply(sd1, sd2),
                                  np.multiply(sd2, sd2)),
                           np.add(sd1, sd2))
        except ZeroDivisionError:
            logging.error("Division by zero in [update] for [sd]")
            exit(1)

        return mu, sd
