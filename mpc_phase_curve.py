import numpy
from matplotlib import pyplot as plt
from astroquery.mpc import MPC
from astroquery.jplhorizons import Horizons
from astropy.time import Time
from matplotlib import rcParams
import matplotlib.font_manager as font_manager
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial
from scipy.optimize import minimize
import emcee
import corner

plt.clf()
plt.close('all')
plt.style.use('/Users/tkareta/Research/utilities/teddy.mplstyle')

colors = ['#000104', '#ed1c24', '#1e90ff', '#bcbec0']

def convert_to_V(mags, filters):
    conversions = numpy.genfromtxt("BandConversion.txt", dtype=str)
    for i in range(len(mags)):
        index = numpy.where(conversions[:,0] == filters[i])
        #print(i, filters[i])
        if filters[i]!="--" and filters[i]!="T" and filters[i]!="N":
            mags[i] += float(conversions[:,1][index])
        else:
            mags[i] += -0.8
    return mags

def reduce(m, r, delta):    
    H = m - 5.0 * numpy.log10(r*delta)
    print(m.size, r.size, delta.size, H.size)
    return H

def log_likelihood(theta, x, y, yerr):
    b, H, log_f = theta
    model = b * x + H
    sigma2 = yerr**2 + model**2 * numpy.exp(2 * log_f)
    return -0.5 * numpy.sum((y - model) ** 2 / sigma2 + numpy.log(sigma2))

def log_prior(theta):
    b, H, log_f = theta
    if 0.0 < b < 1.0 and 0.0 < H < 20.0 and -10.0 < log_f < 1.0:
        return 0.0
    return -numpy.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not numpy.isfinite(lp):
        return -numpy.inf
    return lp + log_likelihood(theta, x, y, yerr)

def working_area(targetname, tzero_isot = "1999-01-01"):
    obs = MPC.get_observations(targetname)
    times = obs["epoch"].value #Julian Dates
    times_isot = Time(times, format="jd").isot
    tzero = Time(tzero_isot).jd
    start = times_isot[0]
    end = times_isot[-1]
    mags = obs["mag"].value
    filters = obs["band"]
    observers = obs["observatory"]

    est_err = numpy.zeros(len(mags))
    for i in range(len(mags)):
        if len(str(mags[i])) == 4 or len(str(mags[i])) == 3:
            est_err[i] = 0.1
        else:
            est_err[i] = 0.04


    clip = numpy.where(mags < 30.0) # temp!
    times, mags, filters, observers = times[clip], mags[clip], filters[clip], observers[clip]
    mags = convert_to_V(mags, filters)
    est_err = est_err[clip]
    clip = numpy.where(mags > -98)
    times, mags, filters, observers = times[clip], mags[clip], filters[clip], observers[clip]
    times_year = Time(times, format="jd").decimalyear
    est_err = est_err[clip]

    eph = MPC.get_ephemeris(targetname, step="20d", start = tzero_isot, number = 1100)

    #horizons_output = numpy.genfromtxt("horizons_results.txt")
    mpc_times = Time(eph["Date"], format="isot").jd #JD


    print("MPC Ephemeris Date Range:", Time(mpc_times.min(), format="jd").decimalyear, Time(mpc_times.max(), format="jd").decimalyear)
    mpc_r = eph["r"] #AU
    mpc_delta = eph["Delta"] #AU
    mpc_phase = eph["Phase"] #degrees

    # interpolation set-up

    r_obs = numpy.interp(times, mpc_times, mpc_r)
    delta_obs = numpy.interp(times, mpc_times, mpc_delta)
    phase_obs = numpy.interp(times, mpc_times, mpc_phase)
    times_year = Time(times, format="jd").decimalyear

    H = reduce(mags, r_obs, delta_obs)
    print(H.shape)

    # time-based outlier rejection
    times_all = times
    times_year_all = times_year
    mags_all = mags
    est_err_all = est_err
    H_all = H
    phase_obs_all = phase_obs
    x1 = numpy.logical_or(times_year > 2006.5, times_year < 2005.5)
    x2 = numpy.logical_or(times_year > 2018.1, times_year < 2017.9)
    conds = numpy.logical_and(x1,x2)
    x3 = numpy.logical_or(times_year > 2011.7, times_year < 2011.2)
    x4 = numpy.logical_or(times_year > 2016.75, times_year < 2016.65)
    conds = numpy.logical_and(conds,x3)
    conds = numpy.logical_and(conds,x4)

    times = times[conds]
    times_year = times_year[conds]
    mags = mags[conds]
    H = H[conds]
    phase_obs = phase_obs[conds]
    est_err = est_err[conds]
    # preliminary outlier rejection to improve initial guess for MCMC

    p = Polynomial.fit(phase_obs, H, 1)
    prelim_phase_curve = p(phase_obs)
    diffs = H - prelim_phase_curve
    sigma_for_clipping = diffs.std()
    p1 = p.convert().coef
    print("Estimated 'real' magnitude scatter:", sigma_for_clipping)
    sigclip = numpy.where(numpy.abs(diffs) < 0.5 * sigma_for_clipping)

    p = Polynomial.fit(phase_obs[sigclip], H[sigclip], 1)
    prelim_phase_curve = p(phase_obs)
    p2 = p.convert().coef
    #print(p1, p2)


    print("all observations reduced and initial guess at phase curve completed")
    print("now attempting MCMC fitting of phase curve....")
    nll = lambda *args: -log_likelihood(*args)
    initial = numpy.array([p2[1], p2[0], numpy.log(0.01)]) + 0.05 * numpy.random.randn(3)
    print("initial guess b, H, log_f:", initial)
    soln = minimize(nll, initial, args=(phase_obs, H, est_err))
    b_ml, H_ml, log_f_ml = soln.x

    print("Maximum likelihood estimates:")
    print("b = {0:.3f}".format(b_ml))
    print("H = {0:.3f}".format(H_ml))
    print("f = {0:.3f}".format(numpy.exp(log_f_ml)))

    
    pos = soln.x + 1e-4 * numpy.random.randn(32, 3)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(phase_obs, H, est_err)
    )
    sampler.run_mcmc(pos, 10000, progress=True);
    tau = sampler.get_autocorr_time()
    print("tau : ", tau)
    medtau = numpy.median(tau)
    flat_samples = sampler.get_chain(discard=3 * int(medtau), thin=int(medtau/2.), flat=True)

    labels = ["Slope [Mags/Deg]", 'H(1,1,0)', 'log(f)']
    fig = corner.corner(flat_samples, labels=labels, color="black", label_kwargs = {'font':"Futura"})

    for i in range(ndim):
        mcmc = numpy.percentile(flat_samples[:, i], [16, 50, 84])
        q = numpy.diff(mcmc)
        print(mcmc[1], q[0], q[1], labels[i])
    samples = sampler.get_chain(discard=0, flat=True)
    theta_max = samples[numpy.argmax(sampler.get_log_prob())]
    print("Max likelihood combo:", theta_max)
    p_max = Polynomial(numpy.array([theta_max[1], theta_max[0]]))
    time_model = numpy.linspace(times.min(), times.max(), 10000)
    
    r_model = numpy.interp(time_model, mpc_times, mpc_r)
    delta_model = numpy.interp(time_model, mpc_times, mpc_delta)
    phase_model = numpy.interp(time_model, mpc_times, mpc_phase)
    times_model = Time(time_model, format="jd").decimalyear
    print("time range:", times_model.min(), times_model.max())
    H_11a_model = p_max(phase_model)
    m_model = H_11a_model + 5.0 * numpy.log10(r_model * delta_model)
    #print(times_model.size, H_11a_model.size)

    #plt.savefig("corner.png", dpi=300)
    plt.savefig("MPC_corner_draft.png", dpi=300)

    #plt.close('all')
    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    fig.set_size_inches(7, 10, forward=True)

    #V_model = H + 5.0 * numpy.log10(r_obs * delta_obs) + 0.04 * phase_obs
    ax1.plot(times_year, mags, 'o', color=colors[0], label="MPC Obs. (Clipped)", zorder=2)
    ax1.plot(times_year_all, mags_all, 'o', color=colors[2], label="All MPC Observations", zorder=1, alpha=0.5)
    ax1.plot(times_model, m_model, color="orangered", zorder=3, label=r'$m_V$'" Model")

    ax1.legend(loc='best',ncol=1)
    ax1.set_ylim(int(mags_all.max()), int(mags_all.min()-1))
    ax1.set_xlabel("Calendar Year", font="Futura")
    ax1.set_ylabel(r'$m_V$', font="Futura")

    ax2.plot(times_year, H, 'o', color=colors[0], label="MPC Obs. (Clipped)", zorder=2)
    ax2.plot(times_year_all, H_all, 'o', color=colors[2], zorder=1, alpha=0.5)
    ax2.plot(times_model, H_11a_model, color="orangered", zorder=3, label=r'$H_V$'" Model")

    #print(times_year_all.size, H_11a_model.size)

    ax2.legend(loc='lower center',ncol=2)
    ax2.set_ylim(int(H.max()), int(H.min()-1))
    ax2.set_xlabel("Calendar Year", font="Futura")
    ax2.set_ylabel(r'$H_V(1,1,\alpha)$', font="Futura")

    ax3.plot(phase_obs, H, 'o', color=colors[0], label="MPC Obs. (Clipped)", zorder=2)
    #ax3.plot(phase_obs, prelim_phase_curve, color="orangered", zorder=3)
    ax3.plot(phase_obs_all, H_all, 'o', color=colors[2], zorder=1, alpha=0.5)
    
    inds = numpy.random.randint(len(flat_samples), size=100)
    for ind in inds:
        sample = flat_samples[ind]
        p = Polynomial(numpy.array([sample[1], sample[0]]))
        ax3.plot(phase_obs, p(phase_obs), color="orangered", zorder=3, alpha=0.1)

    # this is meant to be off the plot, to keep the legend legible. it's lazy, yes.
    ax3.plot(phase_obs, p(phase_obs) - 99.0, color="orangered", zorder=3, alpha=1.0, label="MCMC Phase Curve")

    ax3.legend(loc='lower center',ncol=2)
    ax3.set_ylim(int(H.max()), int(H.min()-1))
    ax3.set_xlabel(r'$\alpha$'+" "+r'$(^{\circ})$', font="Futura")
    ax3.set_ylabel(r'$H_V(1,1,\alpha)$', font="Futura")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.22)

    
    plt.savefig("MPC_draft.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    targetname = "60558"

    working_area(targetname)
