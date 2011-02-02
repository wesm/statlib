from numpy import sqrt, log, exp, pi
import numpy as np

from scipy.stats import norm, chi2
from scipy.special import gammaln as gamln, gamma as gam
import scipy.special as special

## Non-central T distribution

from scipy.stats import rv_continuous

class tpt_gen(rv_continuous):

    def _rvs(self, df, nc):
        return norm.rvs(loc=nc,size=self._size)*sqrt(df) / sqrt(chi2.rvs(df,size=self._size))

    def _pdf(self, x, df, nc):
        n = df*1.0
        nc = nc*1.0
        x2 = x*x
        ncx2 = nc*nc*x2
        fac1 = n + x2
        trm1 = n/2.*log(n) + gamln(n+1)
        trm1 -= n*log(2)+nc*nc/2.+(n/2.)*log(fac1)+gamln(n/2.)
        Px = exp(trm1)
        valF = ncx2 / (2*fac1)
        trm1 = sqrt(2)*nc*x*special.hyp1f1(n/2+1,1.5,valF)
        trm1 /= np.array(fac1*special.gamma((n+1)/2))
        trm2 = special.hyp1f1((n+1)/2,0.5,valF)
        trm2 /= np.array(sqrt(fac1)*special.gamma(n/2+1))
        Px *= trm1+trm2
        return Px

    def _cdf(self, x, df, nc):
        return special.nctdtr(df, nc, x)

    def _ppf(self, q, df, nc):
        return special.nctdtrit(df, nc, q)

    def _stats(self, df, nc, moments='mv'):
        mu, mu2, g1, g2 = None, None, None, None
        val1 = gam((df-1.0)/2.0)
        val2 = gam(df/2.0)
        if 'm' in moments:
            mu = nc*sqrt(df/2.0)*val1/val2
        if 'v' in moments:
            var = (nc*nc+1.0)*df/(df-2.0)
            var -= nc*nc*df* val1**2 / 2.0 / val2**2
            mu2 = var
        if 's' in moments:
            g1n = 2*nc*sqrt(df)*val1*((nc*nc*(2*df-7)-3)*val2**2 \
                                      -nc*nc*(df-2)*(df-3)*val1**2)
            g1d = (df-3)*sqrt(2*df*(nc*nc+1)/(df-2) - \
                              nc*nc*df*(val1/val2)**2) * val2 * \
                              (nc*nc*(df-2)*val1**2 - \
                               2*(nc*nc+1)*val2**2)
            g1 = g1n/g1d
        if 'k' in moments:
            g2n = 2*(-3*nc**4*(df-2)**2 *(df-3) *(df-4)*val1**4 + \
                     2**(6-2*df) * nc*nc*(df-2)*(df-4)* \
                     (nc*nc*(2*df-7)-3)*pi* gam(df+1)**2 - \
                     4*(nc**4*(df-5)-6*nc*nc-3)*(df-3)*val2**4)
            g2d = (df-3)*(df-4)*(nc*nc*(df-2)*val1**2 - \
                                 2*(nc*nc+1)*val2)**2
            g2 = g2n / g2d
        return mu, mu2, g1, g2
# nct = nct_gen(name="nct", longname="A Noncentral T",
#               shapes="df, nc", extradoc="""

# Non-central Student T distribution

#                                  df**(df/2) * gamma(df+1)
# nct.pdf(x,df,nc) = --------------------------------------------------
#                    2**df*exp(nc**2/2)*(df+x**2)**(df/2) * gamma(df/2)
# for df > 0, nc > 0.
# """
#               )

def mvt_pdf(x, mode, scale):
    pass
