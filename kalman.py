import logging
import numpy as np

START = 1.0
TIMESTEPS = 5


def main():
    kf = KalmanFilter(sigma2=1.7)

    x = kf.timestep(x=START)
    for i in range(TIMESTEPS):
        x = kf.timestep(x=x)
        logging.info(f"@ t={i} -> x={x}")

    return x


class KalmanFilter:
    def __init__(self, mu: float = 0.0, sigma2: float = 1.0,):
        self.mu = mu
        self.sd = sigma2

        N = 2

        self.mu1 = np.zeros(N)
        self.sd1 = np.zeros(N)
        self.mu2 = np.zeros(N)
        self.sd2 = np.zeros(N)

    def timestep(self, x: float):
        """
            _summary_

            Args:
                x (_type_): _description_

            Returns:
                _type_: _description_

        """
        try:
            coef = np.divide(1, np.sqrt(np.prod([2.0, self.sigma2, np.pi])))
        except ZeroDivisionError:
            logging.error("Division by zero in [timestep] for [coef]")
            exit(1)

        try:
            exp = np.exp(np.prod([-0.5, np.divide(np.subtract(x, self.mu),
                                                  self.sigma2)]))
        except ZeroDivisionError:
            logging.error("Division by zero in [timestep] for [exp]")
            exit(1)

        return coef * exp

    def update(self):
        """
            _summary_

            Returns:
                _type_: _description_
        """
        try:
            mu = np.divide(np.add(np.multiply(self.mu1, self.sd1),
                                  np.multiply(self.mu2, self.sd2)),
                           np.add(self.sd1, self.sd2))
        except ZeroDivisionError:
            logging.error("Division by zero in [update] for [mu]")
            exit(1)

        try:
            sd = np.divide(np.add(np.multiply(self.sd1, self.sd1),
                                  np.multiply(self.sd2, self.sd2)),
                           np.add(self.sd1, self.sd2))
        except ZeroDivisionError:
            logging.error("Division by zero in [update] for [sd]")
            exit(1)

        return mu, sd

    def predict(self):
        """
            _summary_

            Returns:
                _type_: _description_
        """
        try:
            mu = np.add(self.mu1, self.mu2)
        except ZeroDivisionError:
            logging.error("Division by zero in [predict] for [mu]")
            exit(1)

        try:
            sd = np.add(self.sd1, self.sd2)
        except ZeroDivisionError:
            logging.error("Division by zero in [predict] for [sd]")
            exit(1)

        return mu, sd


class Board:
    def __init__(self, length: int = 20):
        self.length = length
        self.board = ['.'] * length
        self.board[0] = '&'

    def logBoard(self):
        [print(self.board[i], end='') for i in range(self.length)]
        print()


if __name__ == '__main__':
    main()
