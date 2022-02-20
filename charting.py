import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


class Charts:
    def __init__(self, xmin: int = -10, xmax: int = 10, points: int = 100):
        self.xmin = xmin
        self.xmax = xmax
        self.points = points

        self.savepath = os.path.join(os.path.dirname(__file__), "charts")
        plt.rcParams['font.family'] = 'monospace'

    def makeStatePositionPlot(self, mu: float = 0.0, sd: float = 1.0, timestep: int = 1):
        x = np.linspace(self.xmin, self.xmax, self.points)
        y = scipy.stats.norm.pdf(x, mu, sd)

        plt.figure()
        plt.plot(x, y, color='black')
        plt.grid()

        regions = [
            (mu + (1 * sd), mu - (1 * sd), '#0b559f', 'μ ± 1σ'),
            (mu + (1 * sd), mu + (2 * sd), '#2b7bba', 'μ + 2σ'),
            (mu - (1 * sd), mu - (2 * sd), '#2b7bba', 'μ - 2σ'),
            (mu + (2 * sd), mu + (3 * sd), '#539ecd', 'μ + 3σ'),
            (mu - (2 * sd), mu - (3 * sd), '#539ecd', 'μ - 3σ'),
            (mu + (3 * sd), mu + (9 * sd), '#89bedc', 'μ + 4σ'),
            (mu - (3 * sd), mu - (9 * sd), '#89bedc', 'μ - 4σ'),
        ]

        for l, r, c, label in regions:
            plt.plot([l, l],
                     [0.0, scipy.stats.norm.pdf(l, mu, sd)],
                     color='black')
            plt.plot([r, r],
                     [0.0, scipy.stats.norm.pdf(r, mu, sd)],
                     color='black')

            xp = np.linspace(l, r, 10)
            yp = scipy.stats.norm.pdf(xp, mu, sd)

            desc = f"{label} = (min: {l:.2f}, max: {r:.2f})"
            plt.fill_between(xp, yp, color=c, label=desc, alpha=1.0)

        plt.legend(loc="upper right")
        plt.xlim(-5, 100)
        plt.ylim(0, 1)

        plt.title("State Distribution at timestep=%s" % timestep)
        plt.xlabel("X Position")
        plt.ylabel("Probability of being at position X")

        plt.savefig(os.path.join(self.savepath,
                    "XPosition-StateDistribution-timestep=%s.png" % timestep))

    def makeStateVelocityPlot(self, mu: float = 0.0, sd: float = 1.0, timestep: int = 1):
        x = np.linspace(self.xmin, self.xmax, self.points)
        y = scipy.stats.norm.pdf(x, mu, sd)

        plt.figure()
        plt.plot(x, y, color='black')
        plt.grid()

        regions = [
            (mu + (1 * sd), mu - (1 * sd), '#0b559f', 'μ ± 1σ'),
            (mu + (1 * sd), mu + (2 * sd), '#2b7bba', 'μ + 2σ'),
            (mu - (1 * sd), mu - (2 * sd), '#2b7bba', 'μ - 2σ'),
            (mu + (2 * sd), mu + (3 * sd), '#539ecd', 'μ + 3σ'),
            (mu - (2 * sd), mu - (3 * sd), '#539ecd', 'μ - 3σ'),
            (mu + (3 * sd), mu + (9 * sd), '#89bedc', 'μ + 4σ'),
            (mu - (3 * sd), mu - (9 * sd), '#89bedc', 'μ - 4σ'),
        ]

        for l, r, c, label in regions:
            plt.plot([l, l],
                     [0.0, scipy.stats.norm.pdf(l, mu, sd)],
                     color='black')
            plt.plot([r, r],
                     [0.0, scipy.stats.norm.pdf(r, mu, sd)],
                     color='black')

            xp = np.linspace(l, r, 10)
            yp = scipy.stats.norm.pdf(xp, mu, sd)

            desc = f"{label} = (min: {l:.2f}, max: {r:.2f})"
            plt.fill_between(xp, yp, color=c, label=desc, alpha=1.0)

        plt.legend(loc="upper right")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(0, 1)

        plt.title("State Distribution at timestep=%s" % timestep)
        plt.xlabel("X Velocity")
        plt.ylabel("Probability of having velocity X")

        plt.savefig(os.path.join(self.savepath,
                    "XVelocity-StateDistribution-timestep=%s.png" % timestep))

    def makePosteriorPlot(self, positions, velocities, maxtime):

        plt.figure()
        plt.grid()

        pt = np.linspace(0, 2 * np.pi, self.points)

        for i, ((muPos, sdPos), (muVel, sdVel)) in enumerate(zip(positions, velocities)):

            plt.plot(muPos + sdPos * np.cos(pt),
                     muVel + sdVel * np.sin(pt),
                     label='@ timestep %d, σ-pos: %.2f, σ-vel: %.2f' % (i + 1, sdPos, sdVel))

        plt.legend(loc="upper right")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.xmin, self.xmax)

        plt.title("Joint Posterior for maxtime=%s" % maxtime)
        plt.xlabel("X Position")
        plt.ylabel("Velocity")
        plt.savefig(os.path.join(self.savepath, "JointPosterior.png"))

    def makeExpectedErrorPlot(self, xs: list(), ms: list(), est: float, pfail: float):

        plt.figure()
        plt.grid()

        errors = [(m - est) for m in ms]
        print(
            f"For pfail={pfail} Mean Error: {np.mean(errors):.2f}, Std Dev: {np.std(errors):.2f}")
        plt.scatter(xs, errors, label="Observed Errors")

        plt.legend(loc="upper right")
        plt.xlim(min(xs), max(xs))
        plt.ylim(0, 0.25)

        plt.legend(loc="upper right")

        plt.title(f"Expected Error with pfail={pfail}")
        plt.ylabel("Expected Errors")
        plt.xlabel("Sampled GPS Xs @ timestep=20")
        plt.savefig(os.path.join(self.savepath,
                    f"ExpectedErrors-pfail={pfail}.png"))
