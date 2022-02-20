
import numpy as np
import random
from charting import Charts
from kalman import KalmanFilter

START = 1.0
TIMESTEPS = 100


def main():
    kf = KalmanFilter()

    mx = mv = 0.0
    sxMu = xMu = 1.0
    sxSd = xSd = 1.0
    svMu = vMu = 1.0
    svSd = vSd = 1.7
    sdGps = 12.0

    pos, vel = [], []
    for i in range(0, TIMESTEPS):

        print(f"@ timestep={i + 1} Predict XPosition -> μ={xMu} -> σ={xSd}")
        Charts().makeStatePositionPlot(mu=xMu, sd=xSd, timestep=i)

        print(f"@ timestep={i + 1} Predict XVelocity -> μ={vMu} -> σ={vSd}")
        Charts().makeStateVelocityPlot(mu=vMu, sd=vSd, timestep=i)

        pos.append((xMu, xSd))
        vel.append((vMu, vSd))

        xMu = kf.predict(t1=xMu, t2=sxMu)
        xSd = kf.predict(t1=xSd, t2=0.05)
        vMu = kf.predict(t1=vMu, t2=svMu)
        vSd = kf.predict(t1=vSd, t2=0.05)

        mx = kf.measurement(x=sxMu, mu=xMu, sd=xSd)
        mv = kf.measurement(x=svMu, mu=vMu, sd=vSd)

        print(f"@ timestep={i + 1} Position Measurement -> x={mx}")
        print(f"@ timestep={i + 1} Velocity Measurement -> x={mv} \n")

    print(
        f"@ After measurement z=18 and timestep={i + 1} Position Measurement -> x={mx}")

    Charts().makePosteriorPlot(positions=pos,
                               velocities=vel,
                               maxtime=i)

    sampleGPS = np.random.normal(loc=sxMu, scale=sdGps, size=500)

    for pfail in [0.1, 0.5, 0.9]:
        sampleXs = []
        for x in sampleGPS:
            if random.random() > pfail:
                sampleXs.append(kf.measurement(x=x, mu=xMu, sd=xSd))
            else:
                sampleXs.append(1.0)

        Charts().makeExpectedErrorPlot(xs=sampleGPS,
                                       ms=sampleXs,
                                       est=xMu,
                                       pfail=pfail)

    Charts().makeStatePositionPlot(mu=5.0, sd=1.0, timestep="n + 1")
    Charts().makeStatePositionPlot(mu=1.0, sd=1.7, timestep="n + 1")


if __name__ == "__main__":
    main()
