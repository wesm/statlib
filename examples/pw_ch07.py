from statlib.arma import *
import statlib.plotting as plotting
import matplotlib.pyplot as plt

def simulate_71(a=0.9, s=1, T=1000):
    root_s = np.sqrt(s)

    x = 0
    out = np.empty(T)
    for t in xrange(T):
        flip = np.random.binomial(1, a)
        out[t] = x = x if flip else randn() * root_s

    return out

def simulate_76(pi=0.9, phi=0.9, v=1, T=5000):
    rv = np.sqrt(v)

    s = v / (1 - phi**2)
    root_s = np.sqrt(s)

    x = 1
    out = np.empty(T)
    for t in xrange(T):
        flip = np.random.binomial(1, pi)
        if flip:
            out[t] = x = phi * x + randn() * rv
        else:
            out[t] = x = randn() * root_s

    return out

if __name__ == '__main__':
    sim = simulate_76(pi=1, T=10000)
    print acf(sim, nlags=10)
    plotting.plot_acf(sim)

    plt.show()

    yenusd = np.loadtxt('statlib/data/japan-usa1000.txt', delimiter=',')
    ukusd = np.loadtxt('statlib/data/uk-usa1000.txt', delimiter=',')
b
