import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

START = 1.0
TIMESTEPS = 5


def main():
    logging.getLogger().setLevel(logging.INFO)
    kf = KalmanFilter(sigma2=1.7)

    x = kf.timestep(x=START)
    for i in range(TIMESTEPS + 1):
        x = kf.timestep(x=x)
        logging.info(f"  @ t={i} -> x={x}")

    sMu = 0.0
    sStd = 1.7

    for i in range(1, TIMESTEPS + 1):
        print(f"timestep={i} -> mu={sMu} -> std={1.7 * i}")
        Charts().makeTimestepPlot(mu=0.0, sd=1.7 * i, timestep=i)

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
            coef = np.divide(1, np.sqrt(np.prod([2.0, self.sd, np.pi])))
        except ZeroDivisionError:
            logging.error("Division by zero in [timestep] for [coef]")
            exit(1)

        try:
            exp = np.exp(np.prod([-0.5, np.divide(np.subtract(x, self.mu),
                                                  self.sd)]))
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

    def predict(self, mu1, sd1, mu2, sd2):
        """
            _summary_

            Returns:
                _type_: _description_
        """
        mu = np.add(mu1, mu2)
        sd = np.add(sd1, sd2)
        return mu, sd


class Charts:
    def __init__(self, xmin: int = -10, xmax: int = 10, points: int = 100):
        self.xmin = xmin
        self.xmax = xmax
        self.points = points

        self.savepath = os.path.join(os.path.dirname(__file__), "charts")

    def makeTimestepPlot(self, mu: float = 0.0, sd: float = 1.0, timestep: int = 1):
        x = np.linspace(self.xmin, self.xmax, self.points)
        y = scipy.stats.norm.pdf(x, mu, sd)

        plt.figure()
        plt.plot(x, y, color='black')
        plt.grid()

        regions = [
            (mu + (1 * sd), mu - (1 * sd), '#0b559f'),
            (mu + (1 * sd), mu + (2 * sd), '#2b7bba'),
            (mu - (1 * sd), mu - (2 * sd), '#2b7bba'),
            (mu + (2 * sd), mu + (3 * sd), '#539ecd'),
            (mu - (2 * sd), mu - (3 * sd), '#539ecd'),
            (mu + (3 * sd), mu + (9 * sd), '#89bedc'),
            (mu - (3 * sd), mu - (9 * sd), '#89bedc'),
        ]

        for l, r, c in regions:
            plt.plot([l, l],
                     [0.0, scipy.stats.norm.pdf(l, mu, sd)],
                     color='black')
            plt.plot([r, r],
                     [0.0, scipy.stats.norm.pdf(r, mu, sd)],
                     color='black')

            xp = np.linspace(l, r, 10)
            yp = scipy.stats.norm.pdf(xp, mu, sd)
            plt.fill_between(xp, yp, color=c, alpha=1.0)

            plt.annotate(f"{l:.2f}",
                         xy=(xp[0], yp[0] + 0.05),
                         textcoords='data')

        plt.xlim(self.xmin, self.xmax)
        plt.ylim(0, 1)

        plt.title(
            "State Distribution at timestep=%s" % timestep)
        plt.xlabel("X Position")
        plt.ylabel("Probability Distribution for being at position X")

        plt.savefig(os.path.join(self.savepath,
                    "StateDistributions-timestep=%s.png" % timestep))

    def makePosteriorPlot(self, pos, vel, muPos, muVel):
        pass


if __name__ == '__main__':
    main()
