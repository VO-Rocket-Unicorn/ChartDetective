def optimal_fit(x):
    """
    Fourier series approximation with 20 terms.
    
    Period: 10.000000
    Domain: [0.000000, 10.000000]
    
    Args:
        x: Input value(s)
    
    Returns:
        Computed function value(s)
    """
    import numpy as np
    
    x = np.asarray(x)
    scalar_input = x.ndim == 0
    if scalar_input:
        x = x.reshape(1)
    
    # Fourier series with period 10.000000
    omega = 2 * np.pi / 1.0000000000000000e+01
    result = 5.3527822209838338e-01
    
    result += -6.3477928682571949e-01 * np.cos(1 * omega * x)
    result += -1.5158415559768193e-01 * np.cos(2 * omega * x)
    result += -7.7475408297201565e-02 * np.cos(3 * omega * x)
    result += -7.1273873033404572e-02 * np.cos(4 * omega * x)
    result += -2.3231353791511311e-03 * np.cos(5 * omega * x)
    result += -1.2408831365855465e-04 * np.cos(6 * omega * x)
    result += -1.5024053550682603e-02 * np.cos(7 * omega * x)
    result += -2.7612499252813304e-02 * np.cos(8 * omega * x)
    result += -2.0167458473389750e-02 * np.cos(9 * omega * x)
    result += -2.5170069951419471e-02 * np.cos(10 * omega * x)
    result += -2.4955899208261818e-02 * np.cos(11 * omega * x)
    result += -2.0145950753313869e-02 * np.cos(12 * omega * x)
    result += -3.2929256371441104e-02 * np.cos(13 * omega * x)
    result += 2.4466651949153844e-03 * np.cos(14 * omega * x)
    result += -1.0710165678414327e-02 * np.cos(15 * omega * x)
    result += 2.3099877032904424e-02 * np.cos(16 * omega * x)
    result += 9.3940034766885148e-03 * np.cos(17 * omega * x)
    result += -2.5716859456085546e-02 * np.cos(18 * omega * x)
    result += -1.2667793434754970e-02 * np.cos(19 * omega * x)
    result += 1.6042853751029418e-02 * np.cos(20 * omega * x)
    result += 1.6001091918140002e+00 * np.sin(1 * omega * x)
    result += 2.6645594474550699e-01 * np.sin(2 * omega * x)
    result += 1.4726095347588117e-01 * np.sin(3 * omega * x)
    result += 8.5974605843556137e-02 * np.sin(4 * omega * x)
    result += 8.1302581168134408e-02 * np.sin(5 * omega * x)
    result += 4.9719712429303994e-02 * np.sin(6 * omega * x)
    result += 2.2732185904452234e-02 * np.sin(7 * omega * x)
    result += 2.7750592586450316e-02 * np.sin(8 * omega * x)
    result += 2.3904292637469460e-02 * np.sin(9 * omega * x)
    result += 6.4476668736900750e-02 * np.sin(10 * omega * x)
    result += 2.8000890898269822e-03 * np.sin(11 * omega * x)
    result += 9.1726548500897787e-05 * np.sin(12 * omega * x)
    result += 2.9878242637245589e-02 * np.sin(13 * omega * x)
    result += -1.4055042283108027e-03 * np.sin(14 * omega * x)
    result += 1.2823542771296801e-02 * np.sin(15 * omega * x)
    result += 3.1102538663524163e-03 * np.sin(16 * omega * x)
    result += 4.8200277202270821e-03 * np.sin(17 * omega * x)
    result += 4.7022979616996356e-02 * np.sin(18 * omega * x)
    result += 5.2677313567945309e-03 * np.sin(19 * omega * x)
    result += -3.1063986549034264e-03 * np.sin(20 * omega * x)
    
    return result.item() if scalar_input else result