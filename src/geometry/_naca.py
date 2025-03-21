import numpy as np

#https://en.wikipedia.org/wiki/NACA_airfoil#Equation_for_a_cambered_4-digit_NACA_airfoil
def _camber_line( x, m, p, c ):
    return np.where((x>=0)&(x<=(c*p)),
                    m * (x / np.power(p,2)) * (2.0 * p - (x / c)),
                    m * ((c - x) / np.power(1-p,2)) * (1.0 + (x / c) - 2.0 * p ))

def _dyc_over_dx( x, m, p, c ):
    return np.where((x>=0)&(x<=(c*p)),
                    ((2.0 * m) / np.power(p,2)) * (p - x / c),
                    ((2.0 * m ) / np.power(1-p,2)) * (p - x / c ))

def _thickness( x, t, c ):
    term1 =  0.2969 * (np.sqrt(x/c))
    term2 = -0.1260 * (x/c)
    term3 = -0.3516 * np.power(x/c,2)
    term4 =  0.2843 * np.power(x/c,3)
    term5 = -0.1015 * np.power(x/c,4)
    return 5 * t * c * (term1 + term2 + term3 + term4 + term5)

def _naca4(x, m, p, t, c=1):
    dyc_dx = _dyc_over_dx(x, m, p, c)
    th = np.arctan(dyc_dx)
    yt = _thickness(x, t, c)
    yc = _camber_line(x, m, p, c)
    part1 = np.array((x - yt*np.sin(th), yc + yt*np.cos(th)))
    part1 = np.flip(part1, axis=1)[:, :-1]
    part2 = np.array((x + yt*np.sin(th), yc - yt*np.cos(th)))
    return np.vstack((part1.T, part2.T))
