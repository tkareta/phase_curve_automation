import numpy
from matplotlib import pyplot as plt
from astroquery.mpc import MPC
from astroquery.jplhorizons import Horizons
from astropy.time import Time
from matplotlib import rcParams
import matplotlib.font_manager as font_manager
from scipy.interpolate import interp1d

plt.clf()
plt.style.use('/Users/tkareta/Research/utilities/teddy.mplstyle')

colors = ['#000104', '#ed1c24', '#1e90ff', '#bcbec0']

def convert_to_V(mags, filters):
    conversions = numpy.genfromtxt("BandConversion.txt", dtype=str)
    for i in range(len(mags)):
        index = numpy.where(conversions[:,0] == filters[i])
        print(i, filters[i])
        if filters[i]!="--" and filters[i]!="T" and filters[i]!="N":
            mags[i] += float(conversions[:,1][index])
        else:
            mags[i] += -0.8
    return mags

def reduce(m, r, delta):
    return m - 5.0 * numpy.log10(r*delta)

def working_area(targetname, tzero_isot = "1995-01-01"):
    obs = MPC.get_observations(targetname)
    times = obs["epoch"].value #Julian Dates
    times_isot = Time(times, format="jd").isot
    tzero = Time(tzero_isot).jd
    start = times_isot[0]
    end = times_isot[-1]
    mags = obs["mag"].value
    filters = obs["band"]
    observers = obs["observatory"]

    clip = numpy.where(mags < 30.0) # temp!
    times, mags, filters, observers = times[clip], mags[clip], filters[clip], observers[clip]
    mags = convert_to_V(mags, filters)
    clip = numpy.where(mags > -98)
    times, mags, filters, observers = times[clip], mags[clip], filters[clip], observers[clip]

    eph = MPC.get_ephemeris(targetname, step="30d", start = tzero_isot, number = 36*12)

    #horizons_output = numpy.genfromtxt("horizons_results.txt")
    mpc_times = Time(eph["Date"], format="isot").jd #JD
    mpc_r = eph["r"] #AU
    mpc_delta = eph["Delta"] #AU
    mpc_phase = eph["Phase"] #degrees

    r_obs = numpy.interp(times, mpc_times, mpc_r)
    delta_obs = numpy.interp(times, mpc_times, mpc_delta)
    phase_obs = numpy.interp(times, mpc_times, mpc_phase)

    H = reduce(mags, r_obs, delta_obs)


    fig, (ax1, ax2) = plt.subplots(2,1)

    H = 12
    #V_model = H + 5.0 * numpy.log10(r_obs * delta_obs) + 0.04 * phase_obs
    ax1.plot(times - tzero, mags, 'o', color=colors[0], label="Observations from the MPC", zorder=2)

    #tmarker = Time("2024-01-01").jd
    #plt.plot([tmarker, tmarker] - tzero, [28.9, 28.4], ls='dashed', color="black")
    #plt.text(tmarker-tzero, 28.2, "2024", ha="center", va="center", font="Futura", fontsize=10)
    #tmarker = Time("2025-01-01").jd
    #plt.plot([tmarker, tmarker] - tzero, [28.9, 28.4], ls='dashed', color="black")
    #plt.text(tmarker-tzero, 28.2, "2025", ha="center", va="center", font="Futura", fontsize=10)

    #tmarker = 2460650.240903
    #plt.annotate("", xytext=(tmarker-tzero, 26.1), xy=(tmarker-tzero, 27.1), arrowprops=dict(arrowstyle="->", lw = 2.0, color=colors[1]))
    #plt.plot([tmarker - 10, tmarker + 10] - tzero, [26.1, 26.1], ls='dashed', color=colors[1], label="HST Single Frame Limit")

    #plt.plot(2460650.240903 - tzero, 27.9, 'o', color=colors[1], label="HST 8-frame Stack", zorder=2, markersize=6)
    #plt.plot([2460650.240903 - tzero, 2460650.240903 - tzero], [27.4,28.4], color=colors[1], zorder=2, lw = 2.0)

    #plt.plot(times - tzero, V_model, color=colors[2], ls='dashed', zorder=1, label="App. Mag. model, Temp.")

    plt.legend(loc='best')
    #plt.ylim(28.9, 20.2)
    #plt.xlim(0.0, 575)
    ax1.set_xlabel("Days Since " + str(tzero_isot), font="Futura")
    ax1.set_ylabel("Apparent V Magnitude", font="Futura")
    plt.tight_layout()
    plt.savefig("secular_draft.png", dpi=400)
    plt.show()

if __name__ == '__main__':
    targetname = "60558"

    working_area(targetname)
