#!/usr/bin/env python
# coding: utf-8

#===========================================================================================================================
#============= ============= ============= ============  FOLPS  ============== ============== ============= =============== 
# This is a code for efficiently evaluating the redshift space power spectrum in the presence of massive neutrinos.

# We recommend to use NumPy versions ≥ 1.20.0. For older versions, one needs to rescale by a factor 1/N the FFT computation. (see: https://github.com/henoriega/FOLPS-nu).
#===========================================================================================================================

#standard libraries

#For VDG change:   
#        remove_DeltaP=False
#to  
#        remove_DeltaP=True


import numpy as np
import scipy
from scipy import integrate
from scipy.interpolate import CubicSpline
from scipy import interpolate
from scipy.fft import dst, idst
from scipy.special import gamma
from scipy.special import spherical_jn
from scipy.special import eval_legendre
from scipy.integrate import quad
import sys
from scipy.interpolate import interp1d
from scipy.fftpack import dst, idst

def interp(k, x, y):
    '''Cubic spline interpolation.
    
    Args:
        k: coordinates at which to evaluate the interpolated values.
        x: x-coordinates of the data points.
        y: y-coordinates of the data points.
    Returns:
        Cubic interpolation of ‘y’ evaluated at ‘k’.
    '''
    inter = CubicSpline(x, y)
    return inter(k)  
    


def Matrices(Nfftlog = None):    
    '''M matrices. They do not depend on the cosmology, so they are computed only one time.
    
    Args:
        if 'Nfftlog = None' (or not specified) the code use as default 'Nfftlog = 128'. 
        to use a different number of sample points, just specify it as 'Nfftlog =  number'.
        we recommend using the default mode, see Fig.~8 at arXiv:2208.02791. 
    Returns:
        All the M matrices.
    '''
    global M22matrices, M13vectors, bnu_b, N


    remove_DeltaP=False  #change to True for VDG

    
    k_min = 10**(-7); k_max = 100.
    b_nu = -0.1;   #Not yet tested for other values
    
    if Nfftlog == None:
        N = 128
        
    else:
        N = Nfftlog


    if remove_DeltaP:
        print("removing $\Delta P(k,\mu)$")#... WARNING: This violates momentum conservation!!!")
    # else:
    #     print("keeping $\Delta P(k,\mu)$")
    
    #Eq.~ 4.19 at arXiv:2208.02791
    def Imatrix(nu1, nu2):
        return 1/(8 * np.pi**(3/2)) * ( gamma(3/2-nu1)*gamma(3/2-nu2)*gamma(nu1+nu2-3/2) )/( gamma(nu1)*gamma(nu2)*gamma(3-nu1-nu2) )

    
    #M22-type
    def M22(nu1, nu2):
    
        #Overdensity and velocity
        def M22_dd(nu1, nu2):
            return Imatrix(nu1,nu2)*(3/2-nu1-nu2)*(1/2-nu1-nu2)*( (nu1*nu2)*(98*(nu1+nu2)**2 - 14*(nu1+nu2) + 36) - 91*(nu1+nu2)**2+ 3*(nu1+nu2) + 58)/(196*nu1*(1+nu1)*(1/2-nu1)*nu2*(1+nu2)*(1/2-nu2))
        
        def M22_dt_fp(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(-23-21*nu1+(-38+7*nu1*(-1+7*nu1))*nu2+7*(3+7*nu1)*nu2**2) )/(196*nu1*(1+nu1)*nu2*(1+nu2)*(-1+2*nu2))
        
        def M22_tt_fpfp(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-12*(1-2*nu2)**2 + 98*nu1**(3)*nu2 + 7*nu1**2*(1+2*nu2*(-8+7*nu2))- nu1*(53+2*nu2*(17+7*nu2))))/(98*nu1*(1+nu1)*nu2*(1+nu2)*(-1+2*nu2))
        
        def M22_tt_fkmpfp(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-37+7*nu1**(2)*(3+7*nu2) + nu2*(-10+21*nu2) + nu1*(-10+7*nu2*(-1+7*nu2))))/(98*nu1*(1+nu1)*nu2*(1+nu2))
        
        #A function
        def MtAfp_11(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(-5+nu1*(-4+7*(nu1+nu2))))/(7*nu1*(1+nu1)*(-1+2*nu1)*nu2)
        
        def MtAfkmpfp_12(nu1, nu2):
            return -Imatrix(nu1,nu2)*(((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(6+7*(nu1+nu2)))/(56*nu1*(1+nu1)*nu2*(1+nu2)))
        
        def MtAfkmpfp_22(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-18+3*nu1*(1+4*(10-9*nu1)*nu1)+75*nu2+8*nu1*(41+2*nu1*(-28+nu1*(-4+7*nu1)))*nu2+48*nu1*(-9+nu1*(-3+7*nu1))*nu2**2+4*(-39+4*nu1*(-19+35*nu1))*nu2**3+336*nu1*nu2**4) )/(56*nu1*(1+nu1)*(-1+2*nu1)*nu2*(1+nu2)*(-1+2*nu2))
        
        def MtAfpfp_22(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-5+3*nu2+nu1*(-4+7*(nu1+nu2))))/(7*nu1*(1+nu1)*nu2)
        
        def MtAfkmpfpfp_23(nu1, nu2):
            return -Imatrix(nu1,nu2)*(((-1+7*nu1)*(-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(28*nu1*(1+nu1)*nu2*(1+nu2)))
        
        def MtAfkmpfpfp_33(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(-13*(1+nu1)+2*(-11+nu1*(-1+14*nu1))*nu2 + 4*(3+7*nu1)*nu2**2))/(28*nu1*(1+nu1)*nu2*(1+nu2)*(-1+2*nu2))

# Some B functions, not called by default


        def MB2_21(nu1, nu2):
        
        	matrix=-2*((-15*Imatrix(-3 + nu1,2 + nu2))/64. + (15*Imatrix(-2 + nu1,1 + nu2))/16. + (3*Imatrix(-2 + nu1,2 + nu2))/4. - (45*Imatrix(-1 + nu1,nu2))/32. - (9*Imatrix(-1 + nu1,1 + nu2))/8. - (27*Imatrix(-1 + nu1,2 + nu2))/32. + (15*Imatrix(nu1,-1 + nu2))/16. + (3*Imatrix(nu1,1 + nu2))/16. + (3*Imatrix(nu1,2 + nu2))/8. - (15*Imatrix(1 + nu1,-2 + nu2))/64. + (3*Imatrix(1 + nu1,-1 + nu2))/8. - (3*Imatrix(1 + nu1,nu2))/32. - (3*Imatrix(1 + nu1,2 + nu2))/64.)
        	return matrix
        
        
         
        
             
        def MB3_21(nu1, nu2):
        	matrix=-2*((35*Imatrix(-3 + nu1,2 + nu2))/128. - (35*Imatrix(-2 + nu1,1 + nu2))/32. - (25*Imatrix(-2 + nu1,2 + nu2))/32. + (105*Imatrix(-1 + nu1,nu2))/64. + (45*Imatrix(-1 + nu1,1 + nu2))/32. + (45*Imatrix(-1 + nu1,2 + nu2))/64. - (35*Imatrix(nu1,-1 + nu2))/32. - (15*Imatrix(nu1,nu2))/32. - (9*Imatrix(nu1,1 + nu2))/32. - (5*Imatrix(nu1,2 + nu2))/32. + (35*Imatrix(1 + nu1,-2 + nu2))/128. - (5*Imatrix(1 + nu1,-1 + nu2))/32. - (3*Imatrix(1 + nu1,nu2))/64. - Imatrix(1 + nu1,1 + nu2)/32. - (5*Imatrix(1 + nu1,2 + nu2))/128.)
        	return  matrix
                  

        
        def MB2_22(nu1, nu2):
            matrix= Imatrix(nu1, nu2)*(-9*(-3 + 2*nu1 + 2*nu2)*(-1 + 2*nu1 + 2*nu2)*(3 + 4*nu1**2 + nu1*(2 - 12*nu2) + 2*nu2 + 4*nu2**2))/(64.*nu1*(1 + nu1)*nu2*(1 + nu2)*(-4 + nu1 + nu2)*(-3 + nu1 + nu2))
            return matrix 
                  
        def MB3_22(nu1, nu2):
            matrix= Imatrix(nu1, nu2)*(3*(-3 + 2*nu1 + 2*nu2)*(-1 + 2*nu1 + 2*nu2)*(1 + 2*nu1 + 2*nu2)*(3 + 4*nu1**2 + nu1*(2 - 12*nu2) + 2*nu2 + 4*nu2**2))/(64.*nu1*(1 + nu1)*nu2*(1 + nu2)*(-4 + nu1 + nu2)*(-3 + nu1 + nu2))
            return matrix 
               
        def MB4_22(nu1, nu2):
            matrix= Imatrix(nu1, nu2)*((-3 + 2*nu1)*(-3 + 2*nu2)*(-3 + 2*nu1 + 2*nu2)*(-1 + 2*nu1 + 2*nu2)*(1 + 2*nu1 + 2*nu2)*(3 + 2*nu1 + 2*nu2))/(64.*nu1*(1 + nu1)*nu2*(1 + nu2)*(-4 + nu1 + nu2)*(-3 + nu1 + nu2))
            return matrix 

    
        

        
        #D function
        def MB1_11(nu1, nu2):  #(B.55)
            return Imatrix(nu1,nu2)*(3-2*(nu1+nu2))/(4*nu1*nu2)
        
        def MC1_11(nu1, nu2):  #(B.58)
            if remove_DeltaP:
                matrix=0.0* Imatrix(nu1, nu2)
            else: 
                matrix=Imatrix(nu1,nu2)*((-3+2*nu1)*(-3+2*(nu1+nu2)))/(4*nu2*(1+nu2)*(-1+2*nu2))
            return matrix
        
        def MB2_11(nu1, nu2):   #(B.56)
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(4*nu1*nu2)
        
        def MC2_11(nu1, nu2):  #(B.59)
            if remove_DeltaP:
                matrix=0.0* Imatrix(nu1, nu2)
            else: 
                matrix=Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(4*nu2*(1+nu2))
            return matrix

        
        def MD2_21(nu1, nu2):
            if remove_DeltaP:
                matrix=MB2_21(nu1, nu2)
            else: 
                matrix=Imatrix(nu1,nu2)*((-1+2*nu1-4*nu2)*(-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(4*nu1*nu2*(-1+nu2+2*nu2**2))
            return matrix
        
        def MD3_21(nu1, nu2):
            if remove_DeltaP:
                matrix=MB3_21(nu1, nu2)
            else: 
                matrix=Imatrix(nu1,nu2)*((3-2*(nu1+nu2))*(1-4*(nu1+nu2)**2))/(4*nu1*nu2*(1+nu2))
            return matrix
        
        def MD2_22(nu1, nu2):
            if remove_DeltaP:
                matrix=MB2_22(nu1, nu2)
            else: 
                matrix=Imatrix(nu1,nu2)*(3*(3-2*(nu1+nu2))*(1-2*(nu1+nu2)))/(32*nu1*(1+nu1)*nu2*(1+nu2))
            return matrix
        
        def MD3_22(nu1, nu2):
            if remove_DeltaP:
                matrix=MB3_22(nu1, nu2)
            else: 
                matrix=Imatrix(nu1,nu2)*((3-2*(nu1+nu2))*(1-4*(nu1+nu2)**2)*(1+2*(nu1**2-4*nu1*nu2+nu2**2)))/(16*nu1*(1+nu1)*(-1+2*nu1)*nu2*(1+nu2)*(-1+2*nu2))
            return matrix
        
        def MD4_22(nu1, nu2):
            if remove_DeltaP:
                matrix=MB4_22(nu1, nu2)
            else: 
                matrix=Imatrix(nu1,nu2)*((9-4*(nu1+nu2)**2)*(1-4*(nu1+nu2)**2))/(32*nu1*(1+nu1)*nu2*(1+nu2))
            return matrix
    
        
        return (M22_dd(nu1, nu2), M22_dt_fp(nu1, nu2), M22_tt_fpfp(nu1, nu2), M22_tt_fkmpfp(nu1, nu2),
                MtAfp_11(nu1, nu2), MtAfkmpfp_12(nu1, nu2), MtAfkmpfp_22(nu1, nu2), MtAfpfp_22(nu1, nu2), 
                MtAfkmpfpfp_23(nu1, nu2), MtAfkmpfpfp_33(nu1, nu2), MB1_11(nu1, nu2), MC1_11(nu1, nu2), 
                MB2_11(nu1, nu2), MC2_11(nu1, nu2), MD2_21(nu1, nu2), MD3_21(nu1, nu2), MD2_22(nu1, nu2), 
                MD3_22(nu1, nu2), MD4_22(nu1, nu2))
    
    
    #M22-type Biasing
    def M22bias(nu1, nu2):
        
        def MPb1b2(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-4+7*(nu1+nu2)))/(28*nu1*nu2)
        
        def MPb1bs2(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(2+14*nu1**2 *(-1+2*nu2)-nu2*(3+14*nu2)+nu1*(-3+4*nu2*(-11+7*nu2))))/(168*nu1*(1+nu1)*nu2*(1+nu2))
        
        def MPb22(nu1, nu2):
            return 1/2 * Imatrix(nu1, nu2)

        def MPb2bs2(nu1, nu2):
            return Imatrix(nu1,nu2)*((-3+2*nu1)*(-3+2*nu2))/(12*nu1*nu2)

        def MPb2s2(nu1, nu2):
            return Imatrix(nu1,nu2)*((63-60*nu2+4*(3*(-5+nu1)*nu1+(17-4*nu1)*nu1*nu2+(3+2*(-2+nu1)*nu1)*nu2**2)))/(36*nu1*(1+nu1)*nu2*(1+nu2))

        def MPb2t(nu1, nu2):
            return Imatrix(nu1,nu2)*((-4+7*nu1)*(-3+2*(nu1+nu2)))/(14*nu1*nu2)

        def MPbs2t(nu1, nu2):
            return  Imatrix(nu1,nu2)*((-3+2*(nu1+nu2))*(-19-10*nu2+nu1*(39-30*nu2+14*nu1*(-1+2*nu2))))/(84*nu1*(1+nu1)*nu2*(1+nu2))

        def MB1_21(nu1, nu2):
            if remove_DeltaP:
                matrix=Imatrix(nu1, nu2)*(3*(-3 + 2*nu1)*(-3 + 2*nu1 + 2*nu2))/ (8.*nu1*nu2*(1 + nu2)*(-3 + nu1 + nu2))
            else:
                matrix=0.0* Imatrix(nu1, nu2)
            return  matrix

        def MB1_22(nu1, nu2):
            if remove_DeltaP:
                matrix= Imatrix(nu1, nu2)*(-15*(-3 + 2*nu1)*(-3 + 2*nu2)*(-3 + 2*nu1 + 2*nu2)) / (64.*nu1*(1 + nu1)*nu2*(1 + nu2)*(-4 + nu1 + nu2)*(-3 + nu1 + nu2))
            else:
                matrix=0.0* Imatrix(nu1, nu2)
            return  matrix        
        
        return (MPb1b2(nu1, nu2), MPb1bs2(nu1, nu2), MPb22(nu1, nu2), MPb2bs2(nu1, nu2), 
                MPb2s2(nu1, nu2), MPb2t(nu1, nu2), MPbs2t(nu1, nu2),MB1_21(nu1, nu2),MB1_22(nu1, nu2))
    
    
    #M13-type
    def M13(nu1):
        
        #Overdensity and velocity
        def M13_dd(nu1):
            return ((1+9*nu1)/4) * np.tan(nu1*np.pi)/( 28*np.pi*(nu1+1)*nu1*(nu1-1)*(nu1-2)*(nu1-3) )
        
        def M13_dt_fk(nu1):
            return ((-7+9*nu1)*np.tan(nu1*np.pi))/(112*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        def M13_tt_fk(nu1):
            return -(np.tan(nu1*np.pi)/(14*np.pi*(-3 + nu1)*(-2 + nu1)*(-1 + nu1)*nu1*(1 + nu1) ))
        
        # A function
        def Mafk_11(nu1):
            return ((15-7*nu1)*np.tan(nu1*np.pi))/(56*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1)
        
        def Mafp_11(nu1):
            return ((-6+7*nu1)*np.tan(nu1*np.pi))/(56*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1)
        
        def Mafkfp_12(nu1):
            return (3*(-13+7*nu1)*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        def Mafpfp_12(nu1):
            return (3*(1-7*nu1)*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        def Mafkfkfp_33(nu1):
            return ((21+(53-28*nu1)*nu1)*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        def Mafkfpfp_33(nu1):
            return ((-21+nu1*(-17+28*nu1))*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))

        
        return (M13_dd(nu1), M13_dt_fk(nu1), M13_tt_fk(nu1), Mafk_11(nu1),  Mafp_11(nu1), Mafkfp_12(nu1),
                Mafpfp_12(nu1), Mafkfkfp_33(nu1), Mafkfpfp_33(nu1))

    
    #M13-type Biasing
    def M13bias(nu1):
        
        def Msigma23(nu1):
            return (45*np.tan(nu1*np.pi))/(128*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))
        
        return (Msigma23(nu1))
    
    
    #Computation of M22-type matrices
    def M22type(k_min, k_max, N, b_nu, M22):
        
        #nuT = -etaT/2, etaT = bias_nu + i*eta_m        
        nuT = np.zeros(N+1, dtype = complex)
        
        for jj in range(N+1):
            nuT[jj] = -0.5 * (b_nu +  (2*np.pi*1j/np.log(k_max/k_min)) * (jj - N/2) *(N-1)/(N))
            
        #reduce time x10 compared to "for" iterations
        nuT_x, nuT_y = np.meshgrid(nuT, nuT) 
        M22matrix = M22(nuT_y, nuT_x)
        
        return np.array(M22matrix)
    
    
    #Computation of M13-type matrices
    def M13type(k_min, k_max, N, b_nu, M13):
           
        #nuT = -etaT/2, etaT = bias_nu + i*eta_m 
        nuT = np.zeros(N+1, dtype = complex)
        
        for ii in range(N+1):
            nuT[ii] = -0.5 * (b_nu +  (2*np.pi*1j/np.log(k_max/k_min)) * (ii - N/2) *(N-1)/(N))
        
        M13vector = M13(nuT)
            
        return np.array(M13vector)
    
    
    #FFTLog bias for the biasing spectra Pb1b2,...
    bnu_b = 15.1*b_nu
    
    M22T =  M22type(k_min, k_max, N, b_nu, M22)
    M22biasT = M22type(k_min, k_max, N, bnu_b, M22bias)
    M22matrices = np.concatenate((M22T, M22biasT))
    
    M13T = M13type(k_min, k_max, N, b_nu, M13)
    M13biasT = np.reshape(M13type(k_min, k_max, N, bnu_b, M13bias), (1, int(N+1)))
    M13vectors = np.concatenate((M13T, M13biasT))
    
    print('N = '+str(N)+' sampling points')
    print('M matrices have been computed')
  
    return (M22matrices, M13vectors)




def CosmoParam(h, ombh2, omch2, omnuh2):
    '''Gives some inputs for the function 'fOverf0EH'.
    
    Args:
        h = H0/100.
        ombh2: Omega_b h² (baryons)
        omch2: Omega_c h² (CDM)
        omnuh2: Omega_nu h² (massive neutrinos)
    Returns:
        h: H0/100.
        OmM0: Omega_b + Omega_c + Omega_nu (dimensionless matter density parameter)
        fnu: Omega_nu/OmM0
        Massnu: Total neutrino mass [eV]
    '''
                
    Omb = ombh2/h**2;
    Omc = omch2/h**2;
    Omnu = omnuh2/h**2;
        
    OmM0 = Omb + Omc + Omnu; 
    fnu = Omnu/OmM0;
    Massnu = Omnu*93.14*h**2;
        
    return(h, OmM0, fnu, Massnu)




def fOverf0EH(zev, k, OmM0, h, fnu):
    '''Rutine to get f(k)/f0 and f0.
    f(k)/f0 is obtained following H&E (1998), arXiv:astro-ph/9710216
    f0 is obtained by solving directly the differential equation for the linear growth at large scales.
    
    Args:
        zev: redshift
        k: wave-number
        OmM0: Omega_b + Omega_c + Omega_nu (dimensionless matter density parameter)
        h = H0/100
        fnu: Omega_nu/OmM0
    Returns:
        f(k)/f0 (when 'EdSkernels = True' f(k)/f0 = 1)
        f0
    '''
    if (fnu > 0):
        eta = np.log(1/(1+zev))   #log of scale factor
        Neff = 3.046              #effective number of neutrinos
        omrv = 2.469*10**(-5)/(h**2 * (1 + 7/8*(4/11)**(4/3)*Neff)) #rad: including neutrinos
        aeq = omrv/OmM0           #matter-radiation equality
            
        pcb = 5/4 - np.sqrt(1 + 24*(1 - fnu))/4     #neutrino supression
        c = 0.7         
        Nnu = 3                                     #number of neutrinos
        theta272 = (1.00)**2                        # T_{CMB} = 2.7*(theta272)
        pf = (k * theta272)/(OmM0 * h**2)  
        DEdS = np.exp(eta)/aeq                      #growth function: EdS cosmology
            
        yFS = 17.2*fnu*(1 + 0.488*fnu**(-7/6))*(pf*Nnu/fnu)**2  #yFreeStreaming 
        rf = DEdS/(1 + yFS)
        fFit = 1 - pcb/(1 + (rf)**c)                #f(k)/f0
            
    else:
        fFit = np.full(len(k), 1.0)
            
        
    #Getting f0
    def OmM(eta):
        return 1/(1 + ((1-OmM0)/OmM0)* np.exp(3*eta) )
        
    def f1(eta):
        return 2 - 3/2 * OmM(eta)
        
    def f2(eta):
        return 3/2 * OmM(eta)
        
    etaini = -6;  #initial eta, early enough to evolve as EdS (D + \propto a)
    zfin = -0.99;
        
    def etaofz(z):
        return np.log(1/(1 + z))
        
    etafin = etaofz(zfin); 
        
    from scipy.integrate import odeint
        
    #Differential eqs.
    def Deqs(Df, eta):
        Df, Dprime = Df
        return [Dprime, f2(eta)*Df - f1(eta)*Dprime]
        
    #eta range and initial conditions
    eta = np.linspace(etaini, etafin, 1001)   
    Df0 = np.exp(etaini)
    Df_p0 = np.exp(etaini)
        
    #solution
    Dplus, Dplusp = odeint(Deqs, [Df0,Df_p0], eta).T
    
    Dplusp_ = interp(etaofz(zev), eta, Dplusp)
    Dplus_ = interp(etaofz(zev), eta, Dplus)
    f0 = Dplusp_/Dplus_ 
        
    return (k, fFit, f0)




def pknwJ(k, PSLk, h):
    '''Routine (based on J. Hamann et. al. 2010, arXiv:1003.3999) to get the non-wiggle piece of the linear power spectrum.    
    
    Args:
        k: wave-number.
        PSLk: linear power spectrum.
        h: H0/100.
    Returns:
        non-wiggle piece of the linear power spectrum.
    '''
    #ksmin(max): k-range and Nks: points
    ksmin = 7*10**(-5)/h; ksmax = 7/h; Nks = 2**16

    #sample ln(kP_L(k)) in Nks points, k range (equidistant)
    ksT = [ksmin + ii*(ksmax-ksmin)/(Nks-1) for ii in range(Nks)]
    PSL = interp(ksT, k, PSLk)
    logkpkT = np.log(ksT*PSL)
        
    #Discrete sine transf., check documentation
    FSTtype = 1; m = int(len(ksT)/2)
    FSTlogkpkT = dst(logkpkT, type = FSTtype, norm = "ortho")
    FSTlogkpkOddT = FSTlogkpkT[::2]
    FSTlogkpkEvenT = FSTlogkpkT[1::2]
        
    #cut range (remove the harmonics around BAO peak)
    mcutmin = 120; mcutmax = 240;
        
    #Even
    xEvenTcutmin = np.linspace(1, mcutmin-2, mcutmin-2)
    xEvenTcutmax = np.linspace(mcutmax+2, len(FSTlogkpkEvenT), len(FSTlogkpkEvenT)-mcutmax-1)
    EvenTcutmin = FSTlogkpkEvenT[0:mcutmin-2] 
    EvenTcutmax = FSTlogkpkEvenT[mcutmax+1:len(FSTlogkpkEvenT)]
    xEvenTcuttedT = np.concatenate((xEvenTcutmin, xEvenTcutmax))
    nFSTlogkpkEvenTcuttedT = np.concatenate((EvenTcutmin, EvenTcutmax))


    #Odd
    xOddTcutmin = np.linspace(1, mcutmin-1, mcutmin-1)
    xOddTcutmax = np.linspace(mcutmax+1, len(FSTlogkpkEvenT), len(FSTlogkpkEvenT)-mcutmax)
    OddTcutmin = FSTlogkpkOddT[0:mcutmin-1]
    OddTcutmax = FSTlogkpkOddT[mcutmax:len(FSTlogkpkEvenT)]
    xOddTcuttedT = np.concatenate((xOddTcutmin, xOddTcutmax))
    nFSTlogkpkOddTcuttedT = np.concatenate((OddTcutmin, OddTcutmax))

    #Interpolate the FST harmonics in the BAO range
    preT, = map(np.zeros,(len(FSTlogkpkT),))
    PreEvenT = interp(np.linspace(2, mcutmax, mcutmax-1), xEvenTcuttedT, nFSTlogkpkEvenTcuttedT)
    PreOddT = interp(np.linspace(0, mcutmax-2, mcutmax-1), xOddTcuttedT, nFSTlogkpkOddTcuttedT)

    for ii in range(m):
        if (mcutmin < ii+1 < mcutmax):
            preT[2*ii+1] = PreEvenT[ii]
            preT[2*ii] = PreOddT[ii]
        if (mcutmin >= ii+1 or mcutmax <= ii+1):
            preT[2*ii+1] = FSTlogkpkT[2*ii+1]
            preT[2*ii] = FSTlogkpkT[2*ii]

         
                
        
    #Inverse Sine transf.
    FSTofFSTlogkpkNWT = idst(preT, type = FSTtype, norm = "ortho")
    PNWT = np.exp(FSTofFSTlogkpkNWT)/ksT

    PNWk = interp(k, ksT, PNWT)
    DeltaAppf = k*(PSL[7]-PNWT[7])/PNWT[7]/ksT[7]

    irange1 = np.where((k < 1e-3))
    PNWk1 = PSLk[irange1]/(DeltaAppf[irange1] + 1)

    irange2 = np.where((1e-3 <= k) & (k <= ksT[len(ksT)-1]))
    PNWk2 = PNWk[irange2]
        
    irange3 = np.where((k > ksT[len(ksT)-1]))
    PNWk3 = PSLk[irange3]
        
    PNWkTot = np.concatenate((PNWk1, PNWk2, PNWk3))
        
    return(k, PNWkTot)




def cmM(k_min, k_max, N, b_nu, inputpkT):
    '''coefficients c_m, see eq.~ 4.2 - 4.5 at arXiv:2208.02791
    
    Args:
        kmin, kmax: minimal and maximal range of the wave-number k.
        N: number of sampling points (we recommend using N=128).
        b_nu: FFTLog bias (use b_nu = -0.1. Not yet tested for other values).
        inputpkT: k-coordinates and linear power spectrum.
    Returns:
        coefficients c_m (cosmological dependent terms)
    '''
    #define de zero matrices
    M = int(N/2)
    kBins = np.zeros(N)
    c_m = np.zeros(N+1, dtype = complex)
        
    #"kBins" trought "delta" gives logspaced k's in [k_min,k_max] 
    for ii in range (N):
        delta = 1/(N-1) * np.log(k_max/k_min)
        kBins[ii] = k_min * np.exp((ii) * delta)
    f_kl = interp(kBins, inputpkT[0], inputpkT[1]) * (kBins/k_min)**(-b_nu)

    #F_m is the Discrete Fourier Transform (DFT) of f_kl
    #"forward" has the direct transforms scaled by 1/N (numpy version >= 1.20.0)
    F_m = np.fft.fft(f_kl, n = N, norm = "forward" )
        
    #etaT = bias_nu + i*eta_m
    #to get c_m: 1) reality condition, 2) W_m factor
    for ii in range(N+1):
        etaT = b_nu +  (2*np.pi*1j/np.log(k_max/k_min)) * (ii - N/2) *(N-1)/(N)
        if (ii - M < 0):
            c_m[ii] = k_min**(-(etaT))*np.conj(F_m[-ii+M])
        c_m[ii] = k_min**(-(etaT)) * F_m[ii-M]
    c_m[0] = c_m[0]/2
    c_m[int(N)] = c_m[int(N)]/2 
        
    return(c_m)









# import numpy as np
# from scipy import integrate
# from scipy.interpolate import interp1d

def NonLinear(inputpkl, CosmoParams, EdSkernels=False):
    '''Optimized 1-loop corrections to the linear power spectrum.
    
    Args:
        inputpkl: k-coordinates and linear power spectrum.
        CosmoParams: [z_pk, omega_b, omega_cdm, omega_ncdm, h]
        EdSkernels: If True, use EdS-kernels (default: False)
    Returns:
        Tuple of (TableOut, TableOut_NW) with 1-loop contributions
    '''
    # global M22matrices, M13vectors  # Access the matrices defined by Matrices()
    global TableOut, TableOut_NW, f0, kTout, sigma2w, sigma2w_NW
    global z_pk, omega_b, omega_cdm, omega_ncdm, h

    remove_DeltaP=False     #change to True for VDG
    
    # Check if matrices are defined
    if 'M22matrices' not in globals() or 'M13vectors' not in globals():
        raise RuntimeError("Matrices must be computed first by calling Matrices()")
    
    # Extract cosmological parameters
    z_pk, omega_b, omega_cdm, omega_ncdm, h = CosmoParams
    
    # Constants
    k_min, k_max = 1e-7, 100.0
    b_nu = -0.1
    N = 128  # Default FFTLog points
    
    # Extrapolate input power spectrum
    inputpkT = Extrapolate_inputpkl(inputpkl)
    
    # Output k-range
    kminout, kmaxout = 0.001, 0.5
    kTout = np.logspace(np.log10(kminout), np.log10(kmaxout), num=120)
    

    
    # Precompute frequently used values
    log_kratio = np.log(k_max/k_min)
    jj = np.arange(N+1)
    ietam = (2*np.pi*1j/log_kratio) * (jj - N/2) * (N-1)/N
    etamT = b_nu + ietam
    bnu_b = 15.1*b_nu
    etamT_b = bnu_b + ietam
    
    # Evaluation: f(k)/f0 and linear power spectrums
    h, OmM0, fnu, Massnu = CosmoParam(h, omega_b, omega_cdm, omega_ncdm)    
    inputfkT = fOverf0EH(z_pk, inputpkT[0], OmM0, h, fnu)
    f0 = inputfkT[2]
    
    # Handle EdS vs fk kernels
    if EdSkernels:
        Fkoverf0 = np.ones_like(kTout)
        inputpkTf = (inputpkT[0], inputpkT[1])
        inputpkTff = (inputpkT[0], inputpkT[1])
    else:
        Fkoverf0 = interp1d(inputfkT[0], inputfkT[1], bounds_error=False, fill_value="extrapolate")(kTout)
        inputpkTf = (inputpkT[0], inputpkT[1]*inputfkT[1])
        inputpkTff = (inputpkT[0], inputpkT[1]*(inputfkT[1])**2)
    
    # Non-wiggle linear power spectrum
    inputpkT_NW = pknwJ_beta(inputpkT[0], inputpkT[1], h)
    
    if EdSkernels:
        inputpkTf_NW = (inputpkT_NW[0], inputpkT_NW[1])
        inputpkTff_NW = (inputpkT_NW[0], inputpkT_NW[1])
    else:
        inputpkTf_NW = (inputpkT_NW[0], inputpkT_NW[1]*inputfkT[1])
        inputpkTff_NW = (inputpkT_NW[0], inputpkT_NW[1]*(inputfkT[1])**2)
    
    # Vectorized P22type calculation
    def vectorized_P22(kTout, inputpkT, inputpkTf, inputpkTff):
        
        
        (M22_dd, M22_dt_fp, M22_tt_fpfp, M22_tt_fkmpfp, MtAfp_11, MtAfkmpfp_12, 
         MtAfkmpfp_22, MtAfpfp_22, MtAfkmpfpfp_23, MtAfkmpfpfp_33, MB1_11, MC1_11, 
         MB2_11, MC2_11, MD2_21, MD3_21, MD2_22, MD3_22, MD4_22, MPb1b2, MPb1bs2, 
         MPb22, MPb2bs2, MPb2s2, MPb2t, MPbs2t,MB1_21,MB1_22) = M22matrices


        # Precompute coefficients
        cmT = cmM(k_min, k_max, N, b_nu, inputpkT)
        cmTf = cmM(k_min, k_max, N, b_nu, inputpkTf)
        cmTff = cmM(k_min, k_max, N, b_nu, inputpkTff)
        cmT_b = cmM(k_min, k_max, N, bnu_b, inputpkT)
        cmTf_b = cmM(k_min, k_max, N, bnu_b, inputpkTf)
        
        # Prepare output arrays
        results = [np.zeros_like(kTout) for _ in range(24)]
        
        # Vectorized computation over kTout
        K = kTout[:, None]  # Shape (120, 1)
        precvec = K**etamT  # Shape (120, N+1)
        
        vec = cmT * precvec
        vecf = cmTf * precvec
        vecff = cmTff * precvec
        vec_b = cmT_b * (K**etamT_b)
        vecf_b = cmTf_b * (K**etamT_b)
        
        # Compute all P22 terms
        results[0] = (    kTout**3 * np.sum(vec  @ M22_dd      * vec, axis=1)).real
        results[1] = (2 * kTout**3 * np.sum(vecf @ M22_dt_fp   * vec, axis=1)).real
        results[2] = (kTout**3 * (np.sum(vecff   @ M22_tt_fpfp * vec, axis=1) + 
                      np.sum(vecf @ M22_tt_fkmpfp * vecf, axis=1))).real
        
        # Bias terms
        results[3] = (kTout**3 * np.sum(vec_b  @ MPb1b2  * vec_b, axis=1)).real
        results[4] = (kTout**3 * np.sum(vec_b  @ MPb1bs2 * vec_b, axis=1)).real
        results[5] = (kTout**3 * np.sum(vec_b  @ MPb22   * vec_b, axis=1)).real
        results[6] = (kTout**3 * np.sum(vec_b  @ MPb2bs2 * vec_b, axis=1)).real
        results[7] = (kTout**3 * np.sum(vec_b  @ MPb2s2  * vec_b, axis=1)).real
        results[8] = (kTout**3 * np.sum(vecf_b @ MPb2t   * vec_b, axis=1)).real
        results[9] = (kTout**3 * np.sum(vecf_b @ MPbs2t  * vec_b, axis=1)).real
        
        # A-TNS terms
        results[10] = (kTout**3 * np.sum(vecf  @ MtAfp_11       * vec, axis=1)).real
        results[11] = (kTout**3 * np.sum(vecf  @ MtAfkmpfp_12   * vecf, axis=1)).real
        results[12] = (kTout**3 * np.sum(vecff @ MtAfkmpfpfp_33 * vecf, axis=1)).real
        results[13] = (kTout**3 * (np.sum(vecf @ MtAfkmpfp_22   * vecf, axis=1) + 
                         np.sum(vecff @ MtAfpfp_22 * vec, axis=1))).real
        results[14] = (kTout**3 * np.sum(vecff @ MtAfkmpfpfp_23 * vecf, axis=1)).real


        if remove_DeltaP==False:
            results[15] =  (kTout**3 * np.sum(vecf @ MB1_11 * vecf, axis=1)).real + (kTout**3 * np.sum(vec @ MC1_11 * vecff, axis=1)).real
            results[16] =  (kTout**3 * np.sum(vecf @ MB2_11 * vecf, axis=1)).real + (kTout**3 * np.sum(vec @ MC2_11 * vecff, axis=1)).real
        if remove_DeltaP==True:
            results[15] =  (kTout**3 * np.sum(vecf @ MB1_11 * vecf, axis=1)).real
            results[16] =  (kTout**3 * np.sum(vecf @ MB2_11 * vecf, axis=1)).real
        
        # D-RSD terms
        results[17] = (kTout**3 * np.sum(vecf  @ MD2_21 * vecff, axis=1)).real
        results[18] = (kTout**3 * np.sum(vecf  @ MD3_21 * vecff, axis=1)).real
        results[19] = (kTout**3 * np.sum(vecff @ MD2_22 * vecff, axis=1)).real
        results[20] = (kTout**3 * np.sum(vecff @ MD3_22 * vecff, axis=1)).real
        results[21] = (kTout**3 * np.sum(vecff @ MD4_22 * vecff, axis=1)).real
        # =0 if C is kept:  
        results[22] = (kTout**3 * np.sum(vecf_b @ MB1_21 * vec_b, axis=1)).real
        results[23] = (kTout**3 * np.sum(vecf_b @ MB1_22 * vecf_b, axis=1)).real

        
        return tuple(results)
    
    # Vectorized P13type calculation
    def vectorized_P13(kTout, inputpkT, inputpkTf, inputpkTff, inputfkT):
        
        (M13_dd, M13_dt_fk, M13_tt_fk, Mafk_11, Mafp_11, Mafkfp_12, Mafpfp_12, 
         Mafkfkfp_33, Mafkfpfp_33, Msigma23) = M13vectors
        # Precompute coefficients
        cmT = cmM(k_min, k_max, N, b_nu, inputpkT)
        cmTf = cmM(k_min, k_max, N, b_nu, inputpkTf)
        cmTff = cmM(k_min, k_max, N, b_nu, inputpkTff)
        cmT_b = cmM(k_min, k_max, N, bnu_b, inputpkT)

        # Prepare output arrays
        results = [np.zeros_like(kTout) for _ in range(7)]

        # Compute sigma values
        sigma2psi = integrate.simps(inputpkT[1], inputpkT[0]) / (6 * np.pi**2)
        sigma2v   = integrate.simps(inputpkTf[1], inputpkTf[0]) / (6 * np.pi**2)
        sigma2w   = integrate.simps(inputpkTff[1], inputpkTff[0]) / (6 * np.pi**2)

        # Vectorized computation over kTout
        K = kTout[:, None]  # Shape (120, 1)
        precvec = K**etamT   # Shape (120, N+1)

        vec = cmT * precvec  # Shape (120, N+1)
        vecf = cmTf * precvec
        vecff = cmTff * precvec
        vec_b = cmT_b * (K**etamT_b)

        # Fix axis handling for matrix products
        vec_M13_dd = vec @ M13_dd  # Result shape depends on M13_dd dimensions

        # Compute P13 terms - remove axis=1 since results are already 1D
        M13dd = (kTout**3 * (vec @ M13_dd)).real - (61/105) * kTout**2 * sigma2psi
        vecfM13dt_fk = vecf @ M13_dt_fk

        M13dt = 0.5 * (kTout**3 * (Fkoverf0 * (vec @ M13_dt_fk) + vecfM13dt_fk)).real - (
            (23/21)*sigma2psi * Fkoverf0 + (2/21)*sigma2v) * kTout**2

        M13tt = (kTout**3 * Fkoverf0 * (Fkoverf0 * (vec @ M13_tt_fk) + vecfM13dt_fk)).real - (
            (169/105)*sigma2psi * Fkoverf0 + (4/21)*sigma2v) * Fkoverf0 * kTout**2

        results[0] = M13dd
        results[1] = M13dt
        results[2] = M13tt
        results[3] = (kTout**3 * (vec_b @ Msigma23)).real

        # A-TNS terms
        results[4] = (kTout**3 * (Fkoverf0 * (vec @ Mafk_11) + (vecf @ Mafp_11))).real + (
                      (92/35)*sigma2psi * Fkoverf0 - (18/7)*sigma2v) * kTout**2

        results[5] = (kTout**3 * (Fkoverf0 * (vecf @ Mafkfp_12) + (vecff @ Mafpfp_12))).real - (
                      (38/35)*Fkoverf0 * sigma2v + (2/7)*sigma2w) * kTout**2

        results[6] = (kTout**3 * Fkoverf0 * (Fkoverf0 * (vecf @ Mafkfkfp_33) + 
                      (vecff @ Mafkfpfp_33))).real - (
                      (16/35)*Fkoverf0 * sigma2v + (6/7)*sigma2w) * Fkoverf0 * kTout**2
        
        return tuple(results), sigma2w
    
    # Compute P22 and P13 terms
    P22    = vectorized_P22(kTout, inputpkT, inputpkTf, inputpkTff)
    P22_NW = vectorized_P22(kTout, inputpkT_NW, inputpkTf_NW, inputpkTff_NW)
    
    P13overpkl,    sigma2w    = vectorized_P13(kTout, inputpkT, inputpkTf, inputpkTff, inputfkT)
    P13overpkl_NW, sigma2w_NW = vectorized_P13(kTout, inputpkT_NW, inputpkTf_NW, inputpkTff_NW, inputfkT)
    
    # Interpolate linear power spectra
    pk_l = interp1d(inputpkT[0], inputpkT[1], bounds_error=False, fill_value="extrapolate")(kTout)
    pk_l_NW = interp1d(inputpkT_NW[0], inputpkT_NW[1], bounds_error=False, fill_value="extrapolate")(kTout)
    
    # Compute final results
    def compute_final_results(P22, P13, pk, Fkoverf0, sigma2w):
        Ploop_dd = P22[0] + P13[0] * pk
        Ploop_dt = P22[1] + P13[1] * pk
        Ploop_tt = P22[2] + P13[2] * pk
        
        Pb1b2  = P22[3]
        Pb1bs2 = P22[4]
        Pb22   = P22[5] - interp1d(kTout, P22[5], fill_value="extrapolate")(1e-10)
        Pb2bs2 = P22[6] - interp1d(kTout, P22[6], fill_value="extrapolate")(1e-10)
        Pb2s2  = P22[7] - interp1d(kTout, P22[7], fill_value="extrapolate")(1e-10)
        Pb2t   = P22[8]
        Pbs2t  = P22[9]
        sigma23pkl = P13[3] * pk

        # A function:
        I1udd_1 = P13[4] * pk + P22[10]
        I2uud_1 = P13[5] * pk + P22[11]
        I2uud_2 = (P13[6] * pk) / Fkoverf0 + Fkoverf0 * P13[4] * pk + P22[13]
        I3uuu_2 = Fkoverf0 * P13[5] * pk + P22[14]
        I3uuu_3 = P13[6] * pk + P22[12]


        # D function:
        I2uudd_1 = P22[15]   #  f^2*mu^2
        I2uudd_2 = P22[16]   #  f^2*mu^4
        
        I3uuud_2 = P22[17]   #  f^3*mu^4
        I3uuud_3 = P22[18]   #  f^3*mu^6
        
        I4uuuu_2 = P22[19]   #  f^4*mu^4
        I4uuuu_3 = P22[20]   #  f^4*mu^6
        I4uuuu_4 = P22[21]   #  f^4*mu^8

            # =0 if C is kept.
        I3uuud_1_B = P22[22]  # term f^3*mu^2  I3uuud1D = I3uuud1B + I3uuud1C = 0   
        I4uuuu_1_B = P22[23]  # term f^4*mu^3  I4uuud1D = I4uuud1B + I4uuud1C = 0
        
        return (kTout, pk, Fkoverf0, Ploop_dd, Ploop_dt, Ploop_tt, 
                Pb1b2, Pb1bs2, Pb22, Pb2bs2, Pb2s2, sigma23pkl, Pb2t, Pbs2t, 
                I1udd_1, I2uud_1, I2uud_2, I3uuu_2, I3uuu_3, 
                I2uudd_1, I2uudd_2, 
                I3uuud_2, I3uuud_3, I4uuuu_2, I4uuuu_3, I4uuuu_4,
                I3uuud_1_B,I4uuuu_1_B, 
                f0, sigma2w)        
        
        # return (kTout, pk, Fkoverf0, Ploop_dd, Ploop_dt, Ploop_tt, 
        #         Pb1b2, Pb1bs2, Pb22, Pb2bs2, Pb2s2, sigma23pkl, Pb2t, Pbs2t, 
        #         I1udd_1, I2uud_1, I2uud_2, I3uuu_2, I3uuu_3, 
        #         P22[15], P22[16], 
        #         P22[17], P22[18], P22[19], P22[20], P22[21], 
        #         f0, sigma2w)
    
    TableOut    = compute_final_results(P22, P13overpkl, pk_l, Fkoverf0, sigma2w)
    TableOut_NW = compute_final_results(P22_NW, P13overpkl_NW, pk_l_NW, Fkoverf0, sigma2w_NW)
    
    return TableOut, TableOut_NW


def TableOut_interp(k):
    nobjects = 27
    Tableout = np.zeros((nobjects + 1, len(k)))
    for ii in range(nobjects):
        Tableout[ii][:] = interp(k, kTout, TableOut[1+ii])
        Tableout[27][:] = sigma2w
    return Tableout


def TableOut_NW_interp(k):
    nobjects = 27
    Tableout_NW = np.zeros((nobjects + 1, len(k)))
    for ii in range(nobjects):
        Tableout_NW[ii][:] = interp(k, kTout, TableOut_NW[1+ii])
        Tableout_NW[27][:] = sigma2w_NW
    return Tableout_NW




def PEFTs(kev, mu, NuisanParams, Table):
    '''EFT galaxy power spectrum, Eq. ~ 3.40 at arXiv: 2208.02791.
    
    Args: 
        kev: evaluation points (wave-number coordinates).
        mu: cosine angle between the wave-vector ‘\vec{k}’ and the line-of-sight direction ‘\hat{n}’.
        NuisamParams: set of nuisance parameters [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, 
                                                  alphashot0, alphashot2, PshotP] in that order.
                    b1, b2, bs2, b3nl: biasing parameters.
                    alpha0, alpha2, alpha4: EFT parameters.
                    ctilde: parameter for NL0 ∝ Kaiser power spectrum.
                    alphashot0, alphashot2, PshotP: stochastic noise parameters.
       Table: List of non-linear terms given by the wiggle or non-wiggle power spectra.
    Returns:
       EFT galaxy power spectrum in redshift space.
    '''
    
    #NuisanParams
    (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, 
                ctilde, alphashot0, alphashot2, PshotP, avir) = NuisanParams

    
    remove_DeltaP=False    #change to True for VDG
    Winfty_all=False       #change to False for VDG and no analytical marginalization
    
    #Table
    (pkl, Fkoverf0, Ploop_dd, Ploop_dt, Ploop_tt, Pb1b2, Pb1bs2, Pb22, Pb2bs2, 
         Pb2s2, sigma23pkl, Pb2t, Pbs2t, I1udd_1, I2uud_1, I2uud_2, I3uuu_2, I3uuu_3, 
         I2uudd_1D, I2uudd_2D, I3uuud_2D, I3uuud_3D, I4uuuu_2D, I4uuuu_3D, I4uuuu_4D, 
     I3uuud_1B,I4uuuu_1B,
     sigma2w) = Table
    
    fk = Fkoverf0*f0
        
    #linear power spectrum
    Pdt_L = pkl*Fkoverf0; Ptt_L = pkl*Fkoverf0**2;
        
    #one-loop power spectrum 
    Pdd = pkl + Ploop_dd; Pdt = Pdt_L + Ploop_dt; Ptt = Ptt_L + Ploop_tt;
        
        
    #biasing
    def PddXloop(b1, b2, bs2, b3nl):
        return (b1**2 * Ploop_dd + 2*b1*b2*Pb1b2 + 2*b1*bs2*Pb1bs2 + b2**2 * Pb22
                   + 2*b2*bs2*Pb2bs2 + bs2**2 *Pb2s2 + 2*b1*b3nl*sigma23pkl)
        
    def PdtXloop(b1, b2, bs2, b3nl):
        return b1*Ploop_dt + b2*Pb2t + bs2*Pbs2t + b3nl*Fkoverf0*sigma23pkl
        
    def PttXloop(b1, b2, bs2, b3nl):
        return Ploop_tt
        
    #RSD functions       
    def Af(mu, f0):
        return (f0*mu**2 * I1udd_1 + f0**2 * (mu**2 * I2uud_1 + mu**4 * I2uud_2)
                    + f0**3 * (mu**4 * I3uuu_2 +  mu**6 * I3uuu_3)) 
        
    def Df(mu, f0):
        return (f0**2 * (mu**2 * I2uudd_1D + mu**4 * I2uudd_2D) 
                    + f0**3 * (mu**2 * I3uuud_1B + mu**4 * I3uuud_2D + mu**6 * I3uuud_3D)
                    + f0**4 * (mu**2 * I4uuuu_1B + mu**4 * I4uuuu_2D + mu**6 * I4uuuu_3D + mu**8 * I4uuuu_4D))
        
        
    #Introducing bias in RSD functions, eq.~ A.32 & A.33 at arXiv: 2208.02791
    def ATNS(mu, b1):
        return b1**3 * Af(mu, f0/b1)
        
    def DRSD(mu, b1):
        return b1**4 * Df(mu, f0/b1)
        
    def GTNS(mu, b1):
        if remove_DeltaP:
            gtns= 0
        else:
            gtns=-((kev*mu*f0)**2 *sigma2w*(b1**2 * pkl + 2*b1*f0*mu**2 * Pdt_L 
                                   + f0**2 * mu**4 * Ptt_L))
        return gtns
        
        
    #One-loop SPT power spectrum in redshift space
    def PloopSPTs(mu, b1, b2, bs2, b3nl):
        return (PddXloop(b1, b2, bs2, b3nl) + 2*f0*mu**2 * PdtXloop(b1, b2, bs2, b3nl)
                    + mu**4 * f0**2 * PttXloop(b1, b2, bs2, b3nl) + ATNS(mu, b1) + DRSD(mu, b1)
                    + GTNS(mu, b1))
        
        
    #Linear Kaiser power spectrum
    def PKaiserLs(mu, b1):
        return (b1 + mu**2 * fk)**2 * pkl
        
    def PctNLOs(mu, b1, ctilde):
        return ctilde*(mu*kev*f0)**4 * sigma2w**2 * PKaiserLs(mu, b1)
    
    # EFT counterterms
    def Pcts(mu, alpha0, alpha2, alpha4):
        return (alpha0 + alpha2 * mu**2 + alpha4 * mu**4)*kev**2 * pkl
    
    #Stochastics noise
    def Pshot(mu, alphashot0, alphashot2, PshotP):
        return PshotP*(alphashot0 + alphashot2 * (kev*mu)**2)

    def Winfty(mu,avir):
        lambda2= (f0*kev*mu*avir)**2
        exp = - lambda2 * sigma2w /(1+lambda2)
        W   =np.exp(exp) / np.sqrt(1+lambda2)
        return W 


    def Wexp(mu,avir):
        lambda2= (f0*kev*mu*avir)**2
        exp = - lambda2 * sigma2w
        W   =np.exp(exp)
        return W  

    def Wlorentz(mu,avir):
        lambda2= (f0*kev*mu*avir)**2
        x2 = lambda2 * sigma2w
        W   = 1.0/(1.0+x2)
        return W 

    Damping='lor'
    
    if Damping==None: 
        W=1  # EFT if keeps DeltaP
    elif Damping=='exp':
        W=Wexp(mu,avir)
    elif Damping=='lor':
        W=Wlorentz(mu,avir)
    elif Damping=='vdg':
        W=Winfty(mu,avir)



    PK = W*PloopSPTs(mu, b1, b2, bs2, b3nl)  + Pshot(mu, alphashot0, alphashot2, PshotP)

    if Winfty_all==False:
        W = 1.0
        
    PK = PK + W * ( Pcts(mu, alpha0, alpha2, alpha4) + PctNLOs(mu, b1, ctilde) )

    return PK 




def Sigma2Total(kev, mu, Table_NW):
    '''Sigma² tot for IR-resummations, see eq.~ 3.59 at arXiv:2208.02791
    
    Args:
        kev: evaluation points (wave-number coordinates). 
        mu: cosine angle between the wave-vector ‘\vec{k}’ and the line-of-sight direction ‘\hat{n}’.
        Table_NW: List of non-linear terms given by the non-wiggle power spectra.
    Returns:
        Sigma² tot for IR-resummations.
    '''
    kT = kev; pkl_NW = Table_NW[0];
        
    kinit = 10**(-6);  kS = 0.4;                                  #integration limits
    pT = np.logspace(np.log10(kinit),np.log10(kS), num = 10**2)   #integration range
        
    PSL_NW = interp(pT, kT, pkl_NW)
    k_BAO = 1/104                                                 #BAO scale
        
    Sigma2 = 1/(6 * np.pi**2)*scipy.integrate.simps(PSL_NW*(1 - spherical_jn(0, pT/k_BAO) 
                                                + 2*spherical_jn(2, pT/k_BAO)), pT)
        
    deltaSigma2 = 1/(2 * np.pi**2)*scipy.integrate.simps(PSL_NW*spherical_jn(2, pT/k_BAO), pT)
        
    def Sigma2T(mu):
        return (1 + f0*mu**2 *(2 + f0))*Sigma2 + (f0*mu)**2 * (mu**2 - 1)* deltaSigma2
        
    return Sigma2T(mu)




def k_AP(k_obs, mu_obs, qperp, qpar):
    '''True ‘k’ coordinates.
    
    Args: where ‘_obs’ denote quantities that are observed assuming the reference (fiducial) cosmology.
        k_obs: observed wave-number.
        mu_obs: observed cosine angle between the wave-vector ‘\vec{k}’ and the line-of-sight direction ‘\hat{n}’.
        qperp, qpar: AP parameters.
    Returns:
        True wave-number ‘k_AP’.
    '''
    F = qpar/qperp
    return (k_obs/qperp)*(1 + mu_obs**2 * (1./F**2 - 1))**(0.5)




def mu_AP(mu_obs, qperp, qpar):
    '''True ‘mu’ coordinates.
    
    Args: where ‘_obs’ denote quantities that are observed assuming the reference (fiducial) cosmology.
        mu_obs: observed cosine angle between the wave-vector ‘\vec{k}’ and the line-of-sight direction ‘\hat{n}’.
        qperp, qpar: AP parameters.
    Returns:
        True ‘mu_AP’.
    '''
    F = qpar/qperp
    return (mu_obs/F) * (1 + mu_obs**2 * (1/F**2 - 1))**(-0.5)




def Hubble(Om, z_ev):
    '''Hubble parameter.
    
    Args:
        Om: Omega_b + Omega_c + Omega_nu (dimensionless matter density parameter).
        z_ev: redshift of evaluation.
    Returns:
        Hubble parameter.
    '''
    return ((Om) * (1 + z_ev)**3. + (1 - Om))**0.5




def DA(Om, z_ev):
    '''Angular-diameter distance.
    
     Args:
        Om: Omega_b + Omega_c + Omega_nu (dimensionless matter density parameter).
        z_ev: redshift of evaluation.
    Returns:
        Angular diameter distance.
    '''
    r = quad(lambda x: 1. / Hubble(Om, x), 0, z_ev)[0]
    return r / (1 + z_ev)




def Table_interp(k, kev, Table):
    '''Cubic interpolator.
    
    Args:
        k: coordinates at which to evaluate the interpolated values.
        kev: x-coordinates of the data points.
        Table: list of 1-loop contributions for the wiggle and non-wiggle
    '''
    f = interpolate.interp1d(kev, Table, kind = 'cubic', fill_value = "extrapolate")
    Tableout = f(k) 

    return Tableout




def RSDmultipoles(kev, NuisanParams, Omfid = -1, AP = False):
    '''Redshift space power spectrum multipoles.
    
    Args:
        If 'AP=True' (default: 'False') the code perform the AP test.
        If 'AP=True'. Include the fiducial Omfid after ‘NuisanParams’.
        
        kev: wave-number coordinates of evaluation.
        NuisamParams: set of nuisance parameters [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, 
                                                  alphashot2, PshotP] in that order.
                   b1, b2, bs2, b3nl: biasing parameters.
                   alpha0, alpha2, alpha4: EFT parameters.
                   ctilde: parameter for NL0 ∝ Kaiser power spectrum.
                   alphashot0, alphashot2, PshotP: stochastic noise parameters.
                   avir: a^2_vir, for VDG
    Returns:
       Redshift space power spectrum multipoles (monopole, quadrupole and hexadecapole) at 'kev'.
    '''
            
    #NuisanParams
    (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, 
                ctilde, alphashot0, alphashot2, PshotP, avir) = NuisanParams

    remove_DeltaP=False    #change to True for VDG
    Winfty_all=False       #change to False for VDG and no analytical marginalization
    
    if AP == True and Omfid == -1:
        sys.exit("Introduce the fiducial value of the dimensionless matter density parameter as ‘Omfid = value’.")
     
    if AP == True and Omfid > 0:
                
        #Om computed for any cosmology
        OmM = CosmoParam(h, omega_b, omega_cdm, omega_ncdm)[1]
        
        #qperp, qpar: AP parameters.
        qperp = DA(OmM, z_pk)/DA(Omfid, z_pk) 
        qpar = Hubble(Omfid, z_pk)/Hubble(OmM, z_pk) 
        
        
    def PIRs(kev, mu, Table, Table_NW):
        
        if AP == True:
            
            k_true = k_AP(kev, mu, qperp, qpar)
            mu_true = mu_AP(mu, qperp, qpar)
            
            Table_true = Table_interp(k_true, kev, Table)
            Table_NW_true = Table_interp(k_true, kev, Table_NW)
            
            Sigma2T = Sigma2Total(k_true, mu_true, Table_NW_true)
            
            Fkoverf0 = Table_true[1]; fk = Fkoverf0*f0
            pkl = Table_true[0]; pkl_NW = Table_NW_true[0];
            
            
            return ((b1 + fk * mu_true**2)**2 * (pkl_NW + np.exp(-k_true**2 * Sigma2T)*(pkl-pkl_NW)*(1 + k_true**2 * Sigma2T) )
                + np.exp(-k_true**2 * Sigma2T)*PEFTs(k_true, mu_true, NuisanParams, Table_true) 
                + (1 - np.exp(-k_true**2 * Sigma2T))*PEFTs(k_true, mu_true, NuisanParams, Table_NW_true)) 
            
        else:
            
            k = kev; Fkoverf0 = Table[1]; fk = Fkoverf0*f0
            pkl = Table[0]; pkl_NW = Table_NW[0];
            Sigma2T = Sigma2Total(kev, mu, Table_NW)
            
            return ((b1 + fk * mu**2)**2 * (pkl_NW + np.exp(-k**2 * Sigma2T)*(pkl-pkl_NW)*(1 + k**2 * Sigma2T) )
                + np.exp(-k**2 * Sigma2T)*PEFTs(k, mu, NuisanParams, Table) 
                + (1 - np.exp(-k**2 * Sigma2T))*PEFTs(k, mu, NuisanParams, Table_NW))     
    
    
    if AP == True:
        
        Nx = 6                                         #Points
        xGL, wGL = scipy.special.roots_legendre(Nx)    #x=cosθ and weights
        
        def ModelPkl0(Table, Table_NW):
            monop = 0;
            for ii in range(Nx):
                monop = monop + 0.5/(qperp**2 * qpar)*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)
            return monop
        
        def ModelPkl2(Table, Table_NW):    
            quadrup = 0;
            for ii in range(Nx):
                quadrup = quadrup + 5/(2*qperp**2 * qpar)*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)*eval_legendre(2, xGL[ii])
            return quadrup
        
        def ModelPkl4(Table, Table_NW):
            hexadecap = 0;
            for ii in range(Nx):
                hexadecap = hexadecap + 9/(2*qperp**2 * qpar)*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)*eval_legendre(4, xGL[ii])
            return hexadecap
    
    else:
        
        Nx = 6                                         #Points
        xGL, wGL = scipy.special.roots_legendre(Nx)    #x=cosθ and weights
        
        def ModelPkl0(Table, Table_NW):
            monop = 0;
            for ii in range(Nx):
                monop = monop + 0.5*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)
            return monop
        
        def ModelPkl2(Table, Table_NW):    
            quadrup = 0;
            for ii in range(Nx):
                quadrup = quadrup + 5/2*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)*eval_legendre(2, xGL[ii])
            return quadrup
        
        def ModelPkl4(Table, Table_NW):
            hexadecap = 0;
            for ii in range(Nx):
                hexadecap = hexadecap + 9/2*wGL[ii]*PIRs(kev, xGL[ii], Table, Table_NW)*eval_legendre(4, xGL[ii])
            return hexadecap
        
    
    Pkl0 = ModelPkl0(TableOut_interp(kev), TableOut_NW_interp(kev));
    Pkl2 = ModelPkl2(TableOut_interp(kev), TableOut_NW_interp(kev));
    Pkl4 = ModelPkl4(TableOut_interp(kev), TableOut_NW_interp(kev));
    
    #print('Redshift space power spectrum multipoles have been computed')
    #print('')
    #print('All computations have been performed successfully ')
    
    return (kev, Pkl0, Pkl2, Pkl4)




def RSDmultipoles_marginalized_const(kev, NuisanParams, Omfid = -1, AP = False, Hexa = False):
    '''Redshift space power spectrum multipoles 'const': Pℓ,const 
      (α->0, marginalizing over the EFT and stochastic parameters).
    
    Args:
        If 'AP=True' (default: 'False') the code perform the AP test.
        If 'AP=True'. Include the fiducial Omfid after ‘NuisanParams’.
        If 'Hexa = True' (default: 'False') the code includes the hexadecapole.
        
        kev: wave-number coordinates of evaluation.
        NuisamParams: set of nuisance parameters [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, 
                                                  alphashot2, PshotP] in that order.
                   b1, b2, bs2, b3nl: biasing parameters.
                   alpha0, alpha2, alpha4: EFT parameters.
                   ctilde: parameter for NL0 ∝ Kaiser power spectrum.
                   alphashot0, alphashot2, PshotP: stochastic noise parameters.
                   avir: a^2_vir for VDG
    Returns:
       Redshift space power spectrum multipoles (monopole, quadrupole and hexadecapole) at 'kev'.
    '''
    
    #NuisanParams
    (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, 
                ctilde, alphashot0, alphashot2, PshotP, avir) = NuisanParams

    
    remove_DeltaP=False    #change to True for VDG
    Winfty_all=False       #change to False for VDG and no analytical marginalization
    
    if AP == True and Omfid == -1:
        sys.exit("Introduce the fiducial value of the dimensionless matter density parameter as ‘Omfid = value’.")
     
    if AP == True and Omfid > 0:
                
        #Om computed for any cosmology
        OmM = CosmoParam(h, omega_b, omega_cdm, omega_ncdm)[1]
        
        #qperp, qpar: AP parameters.
        qperp = DA(OmM, z_pk)/DA(Omfid, z_pk) 
        qpar = Hubble(Omfid, z_pk)/Hubble(OmM, z_pk) 
        
        
    def PIRs_const(kev, mu, Table, Table_NW):
        
        #NuisanParams_const: α->0 (set to zero EFT and stochastic parameters)
        alpha0, alpha2, alpha4, alphashot0, alphashot2 = np.zeros(5)
        
        NuisanParams_const = (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, 
                              ctilde, alphashot0, alphashot2, PshotP, avir)
        
        
        if AP == True:
            
            k_true = k_AP(kev, mu, qperp, qpar)
            mu_true = mu_AP(mu, qperp, qpar)
            
            Table_true = Table_interp(k_true, kev, Table)
            Table_NW_true = Table_interp(k_true, kev, Table_NW)
            
            Sigma2T = Sigma2Total(k_true, mu_true, Table_NW_true)
            
            Fkoverf0 = Table_true[1]; fk = Fkoverf0*f0
            pkl = Table_true[0]; pkl_NW = Table_NW_true[0];
            
            
            return ((b1 + fk * mu_true**2)**2 * (pkl_NW + np.exp(-k_true**2 * Sigma2T)*(pkl-pkl_NW)*(1 + k_true**2 * Sigma2T) )
                + np.exp(-k_true**2 * Sigma2T)*PEFTs(k_true, mu_true, NuisanParams_const, Table_true) 
                + (1 - np.exp(-k_true**2 * Sigma2T))*PEFTs(k_true, mu_true, NuisanParams_const, Table_NW_true))
        
        
        else:
            
            k = kev; Fkoverf0 = Table[1]; fk = Fkoverf0*f0
            pkl = Table[0]; pkl_NW = Table_NW[0];
            Sigma2T = Sigma2Total(kev, mu, Table_NW)
            
            return ((b1 + fk * mu**2)**2 * (pkl_NW + np.exp(-k**2 * Sigma2T)*(pkl-pkl_NW)*(1 + k**2 * Sigma2T) )
                + np.exp(-k**2 * Sigma2T)*PEFTs(k, mu, NuisanParams_const, Table) 
                + (1 - np.exp(-k**2 * Sigma2T))*PEFTs(k, mu, NuisanParams_const, Table_NW))
        
        
    Nx = 6                                         #Points
    xGL, wGL = scipy.special.roots_legendre(Nx)    #x=cosθ and weights
    
    def ModelPkl0_const(Table, Table_NW):
        if AP == True:
            monop = 1/(qperp**2 * qpar) * sum(0.5*wGL[ii]*PIRs_const(kev, xGL[ii], Table, Table_NW) for ii in range(Nx))
            return monop
        else:
            monop = sum(0.5*wGL[ii]*PIRs_const(kev, xGL[ii], Table, Table_NW) for ii in range(Nx))
            return monop
        
        
    def ModelPkl2_const(Table, Table_NW):    
        if AP == True:
            quadrup = 1/(qperp**2 * qpar) * sum(5/2*wGL[ii]*PIRs_const(kev, xGL[ii], Table, Table_NW)*eval_legendre(2, xGL[ii]) for ii in range(Nx))
            return quadrup
        else:
            quadrup = sum(5/2*wGL[ii]*PIRs_const(kev, xGL[ii], Table, Table_NW)*eval_legendre(2, xGL[ii]) for ii in range(Nx))
            return quadrup
        
        
    def ModelPkl4_const(Table, Table_NW):
        if AP == True:
            hexadecap = 1/(qperp**2 * qpar) * sum(9/2*wGL[ii]*PIRs_const(kev, xGL[ii], Table, Table_NW)*eval_legendre(4, xGL[ii]) for ii in range(Nx))
            return hexadecap
        else:
            hexadecap = sum(9/2*wGL[ii]*PIRs_const(kev, xGL[ii], Table, Table_NW)*eval_legendre(4, xGL[ii]) for ii in range(Nx))
            return hexadecap
        
        
    if Hexa == False:
        Pkl0_const = ModelPkl0_const(TableOut_interp(kev), TableOut_NW_interp(kev));
        Pkl2_const = ModelPkl2_const(TableOut_interp(kev), TableOut_NW_interp(kev));
        return (Pkl0_const, Pkl2_const)
    
    else:
        Pkl0_const = ModelPkl0_const(TableOut_interp(kev), TableOut_NW_interp(kev));
        Pkl2_const = ModelPkl2_const(TableOut_interp(kev), TableOut_NW_interp(kev));
        Pkl4_const = ModelPkl4_const(TableOut_interp(kev), TableOut_NW_interp(kev));
        return (Pkl0_const, Pkl2_const, Pkl4_const)        
    
    
    

def PEFTs_derivatives(k, mu, pkl, PshotP):
    '''Derivatives of PEFTs with respect to the EFT and stochastic parameters.
    
    Args:
        k: wave-number coordinates of evaluation.
        mu: cosine angle between the wave-vector ‘\vec{k}’ and the line-of-sight direction ‘\hat{n}’.
        pkl: linear power spectrum.
        PshotP: stochastic nuisance parameter.
    Returns:
        ∂P_EFTs/∂α_i with: α_i = {alpha0, alpha2, alpha4, alphashot0, alphashot2}
    '''
    
    k2 = k**2
    k2mu2 = k2 * mu**2
    k2mu4 = k2mu2 * mu**2

    PEFTs_alpha0 = k2 * pkl
    PEFTs_alpha2 = k2mu2 * pkl 
    PEFTs_alpha4 = k2mu4 * pkl 
    PEFTs_alphashot0 = PshotP
    PEFTs_alphashot2 = k2mu2 * PshotP
    
    return (PEFTs_alpha0, PEFTs_alpha2, PEFTs_alpha4, PEFTs_alphashot0, PEFTs_alphashot2)  




def RSDmultipoles_marginalized_derivatives(kev, NuisanParams, Omfid = -1, AP = False, Hexa = False):
    '''Redshift space power spectrum multipoles 'derivatives': Pℓ,i=∂Pℓ/∂α_i 
      (derivatives with respect to the EFT and stochastic parameters).
    
    Args:
        If 'AP=True' (default: 'False') the code perform the AP test.
        If 'AP=True'. Include the fiducial Omfid after ‘NuisanParams’.
        If 'Hexa = True' (default: 'False') the code includes the hexadecapole.
        
        kev: wave-number coordinates of evaluation.
        NuisamParams: set of nuisance parameters [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, 
                                                  alphashot2, PshotP] in that order.
                   b1, b2, bs2, b3nl: biasing parameters.
                   alpha0, alpha2, alpha4: EFT parameters.
                   ctilde: parameter for NL0 ∝ Kaiser power spectrum.
                   alphashot0, alphashot2, PshotP: stochastic noise parameters.
    Returns:
       Redshift space power spectrum multipoles (monopole, quadrupole and hexadecapole) at 'kev'.
    '''
            
    #NuisanParams
    (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, 
                ctilde, alphashot0, alphashot2, PshotP, avir) = NuisanParams

    
    remove_DeltaP=False    #change to True for VDG
    
    if AP == True and Omfid == -1:
        sys.exit("Introduce the fiducial value of the dimensionless matter density parameter as ‘Omfid = value’.")
     
    if AP == True and Omfid > 0:
                
        #Om computed for any cosmology
        OmM = CosmoParam(h, omega_b, omega_cdm, omega_ncdm)[1]
        
        #qperp, qpar: AP parameters.
        qperp = DA(OmM, z_pk)/DA(Omfid, z_pk) 
        qpar = Hubble(Omfid, z_pk)/Hubble(OmM, z_pk) 
        
    
    def PIRs_derivatives(kev, mu, Table, Table_NW):
        
        if AP == True:
            
            k_true = k_AP(kev, mu, qperp, qpar)
            mu_true = mu_AP(mu, qperp, qpar)
            
            Table_true = Table_interp(k_true, kev, Table)
            Table_NW_true = Table_interp(k_true, kev, Table_NW)
            
            Sigma2T = Sigma2Total(k_true, mu_true, Table_NW_true)
            
            Fkoverf0 = Table_true[1]; fk = Fkoverf0*f0
            pkl = Table_true[0]; pkl_NW = Table_NW_true[0];
            
            #computing PEFTs_derivatives: wiggle and non-wiggle terms
            PEFTs_alpha0, PEFTs_alpha2, PEFTs_alpha4, PEFTs_alphashot0, PEFTs_alphashot2 = PEFTs_derivatives(k_true, mu_true, pkl, PshotP)
            PEFTs_alpha0_NW, PEFTs_alpha2_NW, PEFTs_alpha4_NW, PEFTs_alphashot0_NW, PEFTs_alphashot2_NW = PEFTs_derivatives(k_true, mu_true, pkl_NW, PshotP)
            
            exp_term = np.exp(-k_true**2 * Sigma2T)
            exp_term_inv = 1 - exp_term
            
            #computing PIRs_derivatives for EFT and stochastic parameters
            PIRs_alpha0 = exp_term * PEFTs_alpha0 + exp_term_inv * PEFTs_alpha0_NW
            PIRs_alpha2 = exp_term * PEFTs_alpha2 + exp_term_inv * PEFTs_alpha2_NW
            PIRs_alpha4 = exp_term * PEFTs_alpha4 + exp_term_inv * PEFTs_alpha4_NW
            PIRs_alphashot0 = exp_term * PEFTs_alphashot0 + exp_term_inv * PEFTs_alphashot0_NW
            PIRs_alphashot2 = exp_term * PEFTs_alphashot2 + exp_term_inv * PEFTs_alphashot2_NW
            
            return (PIRs_alpha0, PIRs_alpha2, PIRs_alpha4, PIRs_alphashot0, PIRs_alphashot2)
        
        
        else:
            k = kev; Fkoverf0 = Table[1]; fk = Fkoverf0*f0
            pkl = Table[0]; pkl_NW = Table_NW[0];
            
            Sigma2T = Sigma2Total(kev, mu, Table_NW)
            
            #computing PEFTs_derivatives: wiggle and non-wiggle terms
            PEFTs_alpha0, PEFTs_alpha2, PEFTs_alpha4, PEFTs_alphashot0, PEFTs_alphashot2 = PEFTs_derivatives(k, mu, pkl, PshotP)
            PEFTs_alpha0_NW, PEFTs_alpha2_NW, PEFTs_alpha4_NW, PEFTs_alphashot0_NW, PEFTs_alphashot2_NW = PEFTs_derivatives(k, mu, pkl_NW, PshotP)
            
            exp_term = np.exp(-k**2 * Sigma2T)
            exp_term_inv = 1 - exp_term
            
            #computing PIRs_derivatives for EFT and stochastic parameters
            PIRs_alpha0 = exp_term * PEFTs_alpha0 + exp_term_inv * PEFTs_alpha0_NW
            PIRs_alpha2 = exp_term * PEFTs_alpha2 + exp_term_inv * PEFTs_alpha2_NW
            PIRs_alpha4 = exp_term * PEFTs_alpha4 + exp_term_inv * PEFTs_alpha4_NW
            PIRs_alphashot0 = exp_term * PEFTs_alphashot0 + exp_term_inv * PEFTs_alphashot0_NW
            PIRs_alphashot2 = exp_term * PEFTs_alphashot2 + exp_term_inv * PEFTs_alphashot2_NW
            
            return (PIRs_alpha0, PIRs_alpha2, PIRs_alpha4, PIRs_alphashot0, PIRs_alphashot2)
        
        
    Nx = 6    
    xGL, wGL = scipy.special.roots_legendre(Nx)    #x=cosθ and weights
    
    def ModelPkl0_derivatives(Table, Table_NW):
        if AP == True:
            monop = 1/(qperp**2 * qpar) * sum(0.5*wGL[ii]*np.array(PIRs_derivatives(kev, xGL[ii], Table, Table_NW)) for ii in range(Nx))
            return monop
        
        else:
            monop = sum(0.5*wGL[ii]*np.array(PIRs_derivatives(kev, xGL[ii], Table, Table_NW)) for ii in range(Nx))
            return monop
        
    
    def ModelPkl2_derivatives(Table, Table_NW):
        if AP == True:
            quadrup = 1/(qperp**2 * qpar) * sum(5/2*wGL[ii]*np.array(PIRs_derivatives(kev, xGL[ii], Table, Table_NW))*eval_legendre(2, xGL[ii]) for ii in range(Nx))
            return quadrup 
        
        else:
            quadrup = sum(5/2*wGL[ii]*np.array(PIRs_derivatives(kev, xGL[ii], Table, Table_NW))*eval_legendre(2, xGL[ii]) for ii in range(Nx))
            return quadrup
    
    
    def ModelPkl4_derivatives(Table, Table_NW):
        if AP == True:
            hexadecap = 1/(qperp**2 * qpar) * sum(9/2*wGL[ii]*np.array(PIRs_derivatives(kev, xGL[ii], Table, Table_NW))*eval_legendre(4, xGL[ii]) for ii in range(Nx))
            return hexadecap
        
        else:
            hexadecap = sum(9/2*wGL[ii]*np.array(PIRs_derivatives(kev, xGL[ii], Table, Table_NW))*eval_legendre(4, xGL[ii]) for ii in range(Nx))
            return hexadecap  
    
    if Hexa == False:   
        Pkl0_derivatives = ModelPkl0_derivatives(TableOut_interp(kev), TableOut_NW_interp(kev));
        Pkl2_derivatives = ModelPkl2_derivatives(TableOut_interp(kev), TableOut_NW_interp(kev));
        return (Pkl0_derivatives, Pkl2_derivatives)
    
    else:
        Pkl0_derivatives = ModelPkl0_derivatives(TableOut_interp(kev), TableOut_NW_interp(kev));
        Pkl2_derivatives = ModelPkl2_derivatives(TableOut_interp(kev), TableOut_NW_interp(kev));
        Pkl4_derivatives = ModelPkl4_derivatives(TableOut_interp(kev), TableOut_NW_interp(kev));
        return (Pkl0_derivatives, Pkl2_derivatives, Pkl4_derivatives)
    
    
    
    
#Marginalization matrices

def startProduct(A, B, invCov):
    '''Computes: A @ InvCov @ B^{T}, where 'T' means transpose.
    
    Args:
         A: first vector, array of the form 1 x n
         B: second vector, array of the form 1 x n
         invCov: inverse of covariance matrix, array of the form n x n
    
    Returns:
         The result of: A @ InvCov @ B^{T}
    '''
    
    return A @ invCov @ B.T 




def compute_L0(Pl_const, Pl_data, invCov):
    '''Computes the term L0 of the marginalized Likelihood.
    
    Args:
         Pl_const: model multipoles for the constant part (Pℓ,const = Pℓ(α->0)), array of the form 1 x n
         Pl_data: data multipoles, array of the form 1 x n 
         invCov: inverse of covariance matrix, array of the form n x n
         
    Return:
         Loglikelihood for the constant part of the model multipoles 
    '''
    
    D_const = Pl_const - Pl_data
    
    L0 = -0.5 * startProduct(D_const, D_const, invCov)             #eq. 2.4 notes on marginalization
    
    return L0




def compute_L1i(Pl_i, Pl_const, Pl_data, invCov):
    '''Computes the term L1i of the marginalized Likelihood.
    
    Args:
         Pl_i: array with the derivatives of the power spectrum multipoles with respect to 
               the EFT and stochastic parameters, i.e., Pℓ,i=∂Pℓ/∂α_i , i = 1,..., ndim
               array of the form ndim x n
         Pl_const: model multipoles for the constant part (Pℓ,const = Pℓ(α->0)), array of the form 1 x n
         Pl_data: data multipoles, array of the form 1 x n
         invCov: inverse of covariance matrix, array of the form n x n
    Return:
         array for L1i
    '''
    
    D_const = Pl_const - Pl_data  
    
    ndim = len(Pl_i)
    
    #computing L1i
    L1i = np.zeros(ndim)
    
    for ii in range(ndim):
        term1 = startProduct(Pl_i[ii], D_const, invCov)
        term2 = startProduct(D_const, Pl_i[ii], invCov)
        L1i[ii] = -0.5 * (term1 + term2)
    
    return L1i




def compute_L2ij(Pl_i, invCov):
    '''Computes the term L2ij of the marginalized Likelihood.
    
    Args:
         Pl_i: array with the derivatives of the power spectrum multipoles with respect to 
               the EFT and stochastic parameters, i.e., Pℓ,i=∂Pℓ/∂α_i , i = 1,..., ndim
               array of the form ndim x n
         invCov: inverse of covariance matrix, array of the form n x n
    Return:
         array for L2ij
    '''
    
    ndim = len(Pl_i)
    
    #Computing L2ij
    L2ij = np.zeros((ndim, ndim))
    
    for ii in range (ndim):
        for jj in range (ndim):
            L2ij[ii, jj] = startProduct(Pl_i[ii], Pl_i[jj], invCov)
            
    return L2ij









###########################################################################
############### Functions for the bispectrum
##########################################################################

## Numpy interpolator, faster but only linear

def pklIR_f(k,pklIRT):
    return np.interp(k, pklIRT[0], pklIRT[1])




# This function should be located in another place, and should e called only if it was not computed before with the 
# NonLinear function. This is the slowest part of the process. 
def pklIR(inputpkT, h=0.6711, fullrange=False):

    pklnw=pknwJ_beta(inputpkT[0], inputpkT[1], h)
    kT = pklnw[0]; pkl_NW = pklnw[1]; pkl = inputpkT[1];
        
    kinit = 10**(-6);  kS = 0.4;                                  #integration limits
    pT = np.logspace(np.log10(kinit),np.log10(kS), num = 10**2)   #integration range
        
    PSL_NW = interp(pT, kT, pkl_NW)
    k_BAO = 1/104                                                 #BAO scale
        
    #Sigma2 = 1/(6 * np.pi**2)*scipy.integrate.simps(PSL_NW*(1 - spherical_jn(0, pT/k_BAO) 
    #                                            + 2*spherical_jn(2, pT/k_BAO)), pT)
        
    Sigma2 = 1/(6 * np.pi**2) * scipy.integrate.simpson(PSL_NW*(1 - spherical_jn(0, pT/k_BAO) 
                                                + 2*spherical_jn(2, pT/k_BAO)), pT)

    pklIRs = pkl_NW + np.exp(-kT**2 * Sigma2)*(pkl-pkl_NW)
    
    #kT=np.array(kT)
    #pklIRs=np.array(pklIRs)
    #array1=np.array([kT,pklIRs])
    
    kmin, kmax = 0.01, 0.5  # Replace with your desired values

    newkT = []
    newpk = []
    for i in range(len(kT)):
        if ((kT[i] >= kmin) & (kT[i] <= kmax) & (i % 2 == 0)):
            newkT.append(kT[i])
            newpk.append(pklIRs[i])
            
        
    if (fullrange==True): output = (kT,pklIRs)
    else: output = (newkT,newpk)

    output=np.array(output)

    np.savetxt('pklIR.txt', output.T, delimiter=' ') 

        
    return output



def pklIR_ini(k, pkl, pklnw, h=0.6711, k_BAO = 1.0/104.):

    kinit = 10**(-6);  kS = 0.4;
    pT = np.logspace(np.log10(kinit),np.log10(kS), num = 10**2)   #integration range        
    PSL_NW = interp(pT, k, pklnw)
        
    Sigma2 = 1/(6 * np.pi**2) * scipy.integrate.simpson(PSL_NW*(1 - spherical_jn(0, pT/k_BAO) 
                                                + 2*spherical_jn(2, pT/k_BAO)), pT)

    pklIRs = pklnw + np.exp(-k**2 * Sigma2)*(pkl-pklnw)
    

    output = (k,pklIRs)

    output=np.array(output)

    #np.savetxt('pklIR.txt', output.T, delimiter=' ') 
        
    return output



# This function should be located in another place, and should e called only if it was not computed f(k) before
def f0_function(z,OmM0):
    
    def OmM(eta):
        return 1/(1 + ((1-OmM0)/OmM0)* np.exp(3*eta) )
        
    def f1(eta):
        return 2 - 3/2 * OmM(eta)
        
    def f2(eta):
        return 3/2 * OmM(eta)
        
    etaini = -6;  #initial eta, early enough to evolve as EdS (D + \propto a)
    zfin = z;
        
    def etaofz(z):
        return np.log(1/(1 + z))
        
    etafin = etaofz(zfin); 
        
    from scipy.integrate import odeint
        
    #Differential eqs.
    def Deqs(Df, eta):
        Df, Dprime = Df
        return [Dprime, f2(eta)*Df - f1(eta)*Dprime]
        
    #eta range and initial conditions
    eta = np.linspace(etaini, etafin, 1001)   
    Df0 = np.exp(etaini)
    Df_p0 = np.exp(etaini)
        
    #solution
    Dplus, Dplusp = odeint(Deqs, [Df0,Df_p0], eta).T
    
    Dplusp_ = interp(etaofz(zfin), eta, Dplusp)
    Dplus_ = interp(etaofz(zfin), eta, Dplus)
    f0 = Dplusp_/Dplus_ 
        
    return f0




def Qij(ki, kj, xij, mui, muj, f, bisp_nuis_params):

    b1, b2, bs, c1, c2, Bshot, Pshot, avir = bisp_nuis_params

    fi=f; fj=f; fij=f;
    Z1i = b1 + fi * mui**2;
    Z1j = b1 + fj * muj**2;
    # Z1efti = Z1i - c1*(ki*mui)**2;
    # Z1eftj = Z1j - c1*(kj*muj)**2 ;
    
    kmu = ki*mui + kj*muj;
    mu2 = kmu**2 / (ki**2 + kj**2 + 2*ki*kj*xij);
    crossterm = 1.0/2.0*kmu * ( fj * muj / kj * Z1i  +  fi * mui / ki * Z1j  ) 

    advection = xij/2.0 *(ki/kj + kj/ki)
    F2 = 5.0/7.0 + 2.0/7.0 * xij**2 + advection
    G2 = 3.0/7.0 + 4.0/7.0 * xij**2 + advection
    
    Z2 = b1*F2 + fij*mu2*G2 + crossterm + b2/2.0 + bs*(xij**2 - 1.0/3.0);
    
    # Qij = 2*Z1efti*Z1eftj * Z2;
    Qij = 2 * Z2;
    
    return  Qij



def bispectrum(k1, k2, x12, mu1, phi, f, sigma2v, Sigma2, deltaSigma2, bisp_nuis_params, qpar, qperp, k_pkl_pklnw):

    b1, b2, bs, c1, c2, Bshot, Pshot, avir = bisp_nuis_params

    cosphi=np.cos(phi)
    APtransf = APtransforms(k1, k2, x12, mu1, cosphi, qpar, qperp)
    k1AP, k2AP, k3AP, x12AP, x23AP, x31AP, mu1AP, mu2AP, mu3AP,cosphi = APtransf

    Q12 = Qij(k1AP, k2AP, x12AP, mu1AP, mu2AP, f, bisp_nuis_params);
    Q13 = Qij(k1AP, k3AP, x31AP, mu1AP, mu3AP, f, bisp_nuis_params);
    Q23 = Qij(k2AP, k3AP, x23AP, mu2AP, mu3AP, f, bisp_nuis_params);


    pk1   = np.interp(k1AP, k_pkl_pklnw[0], k_pkl_pklnw[1])
    pk1nw = np.interp(k1AP, k_pkl_pklnw[0], k_pkl_pklnw[2])
    pk2   = np.interp(k2AP, k_pkl_pklnw[0], k_pkl_pklnw[1])
    pk2nw = np.interp(k2AP, k_pkl_pklnw[0], k_pkl_pklnw[2])
    pk3   = np.interp(k3AP, k_pkl_pklnw[0], k_pkl_pklnw[1])
    pk3nw = np.interp(k3AP, k_pkl_pklnw[0], k_pkl_pklnw[2])


    e1IR = (1 + f*mu1AP**2 *(2 + f))*Sigma2 + (f*mu1AP)**2 * (mu1AP**2 - 1)* deltaSigma2
    e2IR = (1 + f*mu2AP**2 *(2 + f))*Sigma2 + (f*mu2AP)**2 * (mu2AP**2 - 1)* deltaSigma2
    e3IR = (1 + f*mu3AP**2 *(2 + f))*Sigma2 + (f*mu3AP)**2 * (mu3AP**2 - 1)* deltaSigma2

    pkIR1= pk1nw + (pk1-pk1nw)*np.exp(-e1IR*k1AP**2)
    pkIR2= pk2nw + (pk2-pk2nw)*np.exp(-e2IR*k2AP**2)
    pkIR3= pk3nw + (pk3-pk3nw)*np.exp(-e3IR*k3AP**2)



    # pk1 = pklIR_f(k1AP,pk_in);
    # pk2 = pklIR_f(k2AP,pk_in);
    # pk3 = pklIR_f(k3AP,pk_in);

    f1=f; f2=f; f3=f;
    Z1_1 = b1 + f1 * mu1AP**2;
    Z1_2 = b1 + f2 * mu2AP**2;
    Z1_3 = b1 + f3 * mu3AP**2;
    Z1eft1 = Z1_1 - (c1*mu1AP**2 + c2*mu1AP**4)*k1AP**2
    Z1eft2 = Z1_2 - (c1*mu2AP**2 + c2*mu2AP**4)*k2AP**2
    Z1eft3 = Z1_3 - (c1*mu3AP**2 + c2*mu3AP**4)*k3AP**2

    B12 = Q12 * Z1eft1*pkIR1 * Z1eft2*pkIR2;
    B13 = Q13 * Z1eft1*pkIR1 * Z1eft3*pkIR3;
    B23 = Q23 * Z1eft3*pkIR2 * Z1eft3*pkIR3;

    sigma2w = 10  #cambiar por sigma2v

    l2 = (k1AP*mu1AP)**2 + (k2AP*mu2AP)**2 + (k3AP*mu3AP)**2
    l2 = 0.5 * l2 * (f*avir)**2
    Winfty = np.exp(- l2 * sigma2v /(1+l2) ) / np.sqrt((1+l2)**3) 
    Wlor = 1.0/(1.0+l2*sigma2v)

    Damping='lor'    
    if Damping==None: 
        W=1  # EFT if keeps DeltaP
    elif Damping=='lor':
        W=Wlor
    elif Damping=='vdg':
        W=Winfty

    
    
    ### Noise 
    # To match eq.3.14 of 2110.10161, one makes (1+Pshot) -> (1+Pshot)/bar-n; Bshot -> Bshot/bar-n
    shot = (b1*Bshot + 2.0*(1+Pshot)*f1*mu1AP**2)*Z1eft1*pkIR1 
    + (b1*Bshot + 2.0*(1+Pshot)*f2*mu2AP**2)*Z1eft2*pkIR2 
    + (b1*Bshot + 2.0*(1+Pshot)*f3*mu3AP**2)*Z1eft3*pkIR3  
    + (1+Pshot)**2


    bispectrum = W*(B12 + B13 + B23) + shot
    alpha      = qpar*qperp**2
    bispectrum = bispectrum / alpha**2
    
    return bispectrum






def Bisp_Sugiyama(bisp_cosmo_params, bisp_nuis_params, k_pkl_pklnw, z_pk, 
                  k1k2pairs, Omfid=-1,precision=[4,5,5]):

    OmM, h = bisp_cosmo_params

    qperp, qpar = 1, 1

    if Omfid > 0:
        qperp = DA(OmM, z_pk)/DA(Omfid, z_pk) 
        qpar  = Hubble(Omfid, z_pk)/Hubble(OmM, z_pk) 
        #Om computed for any cosmology
        #OmM = CosmoParam(h, omega_b, omega_cdm, omega_ncdm)[1]

    f = f0_function(z_pk,OmM);

    kT=k_pkl_pklnw[0]
    pklT=k_pkl_pklnw[1]
    sigma2v_, Sigma2_, deltaSigma2_ = sigmas(kT,pklT)
    # print(sigma2v_, Sigma2_, deltaSigma2_,f)


    
    # print(sigma2v_)

    #These are tables for GL pairs [phi,mu,x] [[x1,w1],[x2,w2],....]. We should compute them here
    tablesGL=tablesGL_f(precision)

    size=len(k1k2pairs)

    B000=np.zeros(size)
    B202=np.zeros(size)
    
    for ii in range(size):
        k1,k2 = k1k2pairs[ii]
        B000[ii], B202[ii] = Sugiyama_B000_B202(k1, k2, f, sigma2v_, Sigma2_, deltaSigma2_, bisp_nuis_params, qpar, qperp, tablesGL,k_pkl_pklnw)
    
    return(B000,B202)





def Sugiyama_B000_B202(k1, k2, f, sigma2v, Sigma2, deltaSigma2, bisp_nuis_params, qpar, qperp, tablesGL, k_pkl_pklnw):
    phiGL = tablesGL[0]  
    xGL = tablesGL[1]    
    muGL = tablesGL[2]  
    
    # Extract values and weights
    phi_values = phiGL[:, 0] 
    phi_weights = phiGL[:, 1] 
    
    mu_values = muGL[:, 0]  
    mu_weights = muGL[:, 1] 
    
    x_values = xGL[:, 0] 
    x_weights = xGL[:, 1]
    
    # Constants
    fourpi = 12.53667061
    normB000 = 0.5 / fourpi
    normB202 = 5.0 / 2.0 / fourpi
    
    # Create meshgrid for vectorized computation
    x_mesh, mu_mesh, phi_mesh = np.meshgrid(x_values, mu_values, phi_values, indexing='ij')
    
    # Compute bispectrum for all combinations
    bisp = bispectrum(
        k1, k2,
        x_mesh,
        mu_mesh,
        phi_mesh,
        f, sigma2v, Sigma2, deltaSigma2, 
        bisp_nuis_params, qpar, qperp, k_pkl_pklnw
    )
    
    # Multiply by phi weights and sum over phi dimension (axis=2)
    int_phi = 2 * np.sum(bisp * phi_weights, axis=2)
    
    # Compute B000 integral
    int_mu_B000  = np.sum(int_phi * mu_weights, axis=1)
    int_all_B000 = np.sum(int_mu_B000 * x_weights)
    
    # Compute B202 integral
    leg2 = 0.5 * (-1.0 + 3.0 * mu_values**2)
    int_mu_B202  = np.sum(int_phi * leg2 * mu_weights, axis=1)
    int_all_B202 = np.sum(int_mu_B202 * x_weights)
    
    B000 = int_all_B000 * normB000
    B202 = int_all_B202 * normB202
    
    return B000, B202



#These are GL pairs [[x1,w1],[x2,w2],....]. We should compute them here
def tablesGL_f(precision=[4,5,5]):

    Nphi,Nx,Nmu = precision
                                
    Pi= np.pi
    
    phi_roots, phi_weights = scipy.special.roots_legendre(Nphi) 
    phi_roots = Pi/2 * phi_roots + Pi/2;  phi_weights = Pi/2 * phi_weights
    phiGL=np.array([phi_roots,phi_weights]).T
    
    x_roots, x_weights = scipy.special.roots_legendre(Nx) 
    xGL=np.array([x_roots,x_weights]).T
    
    mu_roots, mu_weights = scipy.special.roots_legendre(Nmu) 
    muGL=np.array([mu_roots,mu_weights]).T 
    tablesGL = [phiGL,xGL,muGL]
    
    return tablesGL



def kAP(k, mu, qpar, qperp):
    return k / qperp * np.sqrt(1 + mu**2 * (-1 + (qperp**2) / (qpar**2)))

def muAP(mu, qpar, qperp):
    return (mu * qperp / qpar) / np.sqrt(1 + mu**2 * (-1 + (qperp**2) / (qpar**2)))

def APtransforms(k1, k2, x12, mu1, cosphi, qpar, qperp):
    k3 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * x12)
    mu2 = np.sqrt(1 - mu1**2) * np.sqrt(1 - x12**2) * cosphi + mu1 * x12
    mu3 = -k1 / k3 * mu1 - k2 / k3 * mu2

    k1AP = kAP(k1, mu1, qpar, qperp)
    k2AP = kAP(k2, mu2, qpar, qperp)
    k3AP = kAP(k3, mu3, qpar, qperp)

    mu1AP = muAP(mu1, qpar, qperp)
    mu2AP = muAP(mu2, qpar, qperp)
    mu3AP = muAP(mu3, qpar, qperp)

    x12AP = (k3AP**2 - k1AP**2 - k2AP**2) / (2 * k1AP * k2AP)
    x31AP = -(k1AP + k2AP*x12AP)/k3AP
    x23AP = -(k2AP + k1AP*x12AP)/k3AP

    output = np.array([k1AP, k2AP, k3AP, x12AP, x23AP, x31AP, mu1AP, mu2AP, mu3AP,cosphi])
    
    return output



def sigmas(kT,pklT):

    k_BAO = 1/104
    kS =0.4

    sigma2v_  = integrate.simps(pklT, kT) / (6 * np.pi**2)
    sigma2v_ *= 1.05  #correction due to k cut

    pklT_=pklT[kT<=0.4].copy()
    kT_=kT[kT<=0.4].copy()
    
    Sigma2_ = 1/(6 * np.pi**2)*scipy.integrate.simps(pklT_*(1 - spherical_jn(0, kT_/k_BAO) 
                                                + 2*spherical_jn(2, kT_/k_BAO)), kT_)
    deltaSigma2_ = 1/(2 * np.pi**2)*scipy.integrate.simps(pklT*spherical_jn(2, kT/k_BAO), kT)

    return sigma2v_, Sigma2_, deltaSigma2_







#### Other functions

def LinearRegression(inputxy): 
    '''Linear regression.
    
    Args:
        inputxy: data set with x- and y-coordinates.
    Returns:
        slope ‘m’ and the intercept ‘b’.
    '''
    xm = np.mean(inputxy[0])
    ym = np.mean(inputxy[1])
    Npts = len(inputxy[0])
    
    SS_xy = np.sum(inputxy[0]*inputxy[1]) - Npts*xm*ym
    SS_xx = np.sum(inputxy[0]**2) - Npts*xm**2
    m = SS_xy/SS_xx
    
    b = ym - m*xm
    return (m, b)




def Extrapolate(inputxy, outputx):
    '''Extrapolation.
    
    Args:
        inputxy: data set with x- and y-coordinates.
        outputx: x-coordinates of extrapolation.
    Returns:
        extrapolates the data set ‘inputxy’ to the range given by ‘outputx’.
    '''
    m, b = LinearRegression(inputxy)
    outxy = [(outputx[ii], m*outputx[ii]+b) for ii in range(len(outputx))]
    
    return np.array(np.transpose(outxy))




def ExtrapolateHighkLogLog(inputT, kcutmax, kmax):
    '''Extrapolation for high-k values.
    
    Args:
        inputT: k-coordinates and linear power spectrum.
        kcutmax: value of ‘k’ from which ‘inputT’ will be interpolated.
        kmax: value of ‘k’ up to which ‘inputT’ will be interpolated.
    Returns:
        extrapolation for high-k values (from ‘kcutmax’ to ‘kmax’) for a given linear power spectrum ‘ inputT’.
    '''
    cutrange = np.where(inputT[0]<= kcutmax)
    inputcutT = np.array([inputT[0][cutrange], inputT[1][cutrange]])
    listToExtT = inputcutT[0][-6:]
    tableToExtT = np.array([listToExtT, inputcutT[1][-6:]])
    delta = np.log10(listToExtT[2])-np.log10(listToExtT[1])
    lastk = np.log10(listToExtT[-1])
    
    logklist = [];
    while (lastk <= np.log10(kmax)):
        logklistT = lastk + delta;
        lastk = logklistT
        logklist.append(logklistT)
    logklist = np.array(logklist)
    
    sign = np.sign(tableToExtT[1][1])
    tableToExtT = np.log10(np.abs(tableToExtT))
    logextT = Extrapolate(tableToExtT, logklist)
    
    output = np.array([10**logextT[0], sign*10**logextT[1]])
    output = np.concatenate((inputcutT, output), axis=1)
        
    
    return output




def ExtrapolateLowkLogLog(inputT, kcutmin, kmin):
    '''Extrapolation for low-k values.
    
    Args:
        inputT: k-coordinates and linear power spectrum.
        kcutmin: value of ‘k’ from which ‘inputT’ will be interpolated.
        kmin: value of ‘k’ up to which ‘inputT’ will be interpolated.
    Returns:
        extrapolation for low-k values (from ‘kcutmin’ to ‘kmin’) for a given linear power spectrum ‘inputT’.
    '''
    cutrange = np.where(inputT[0] > kcutmin)
    inputcutT = np.array([inputT[0][cutrange], inputT[1][cutrange]])
    listToExtT = inputcutT[0][:5]
    tableToExtT = np.array([listToExtT, inputcutT[1][:5]])
    delta = np.log10(listToExtT[2])-np.log10(listToExtT[1])
    firstk = np.log10(listToExtT[0])
    
    logklist = [];
    while (firstk > np.log10(kmin)):
        logklistT = firstk - delta;
        firstk = logklistT
        logklist.append(logklistT)
    logklist = np.array(list(reversed(logklist)))
    
    sign = np.sign(tableToExtT[1][1])
    tableToExtT = np.log10(np.abs(tableToExtT))
    logextT = Extrapolate(tableToExtT, logklist)
    
    output = np.array([10**logextT[0], sign*10**logextT[1]])
    output = np.concatenate((output, inputcutT), axis=1)
        
    
    return output




def ExtrapolatekLogLog(inputT, kcutmin, kmin, kcutmax, kmax):
    '''Extrapolation at low-k and high-k.
    
    Args:
        inputT: k-coordinates and linear power spectrum.
        kcutmin, kcutmax: value of ‘k’ from which ‘inputT’ will be interpolated.
        kmin, kmax: value of ‘k’ up to which ‘inputT’ will be interpolated.
    Returns:
        combines extrapolation al low-k and high-k.
    '''
    output = ExtrapolateLowkLogLog(ExtrapolateHighkLogLog(inputT, kcutmax, kmax), kcutmin, kmin)
    
    return output




def Extrapolate_inputpkl(inputT):
    '''Extrapolation to the input linear power spectrum.
    
    Args:
        inputT: k-coordinates and linear power spectrum.
    Returns:
        extrapolates the input linear power spectrum ‘inputT’ to low-k or high-k if needed.
    '''
    kcutmin = min(inputT[0]); kmin = 10**(-5);
    kcutmax = max(inputT[0]); kmax = 200
    
    if ((kmin < kcutmin) or (kmax > kcutmax)):
        output = ExtrapolatekLogLog(inputT, kcutmin, kmin, kcutmax, kmax)
        
    else:
        output = inputT
        
    return output



def pknwJ_beta(k, PSLk, h):
    '''Routine (based on J. Hamann et. al. 2010, arXiv:1003.3999) to get the non-wiggle piece of the linear power spectrum.    
    
    Args:
        k: wave-number.
        PSLk: linear power spectrum.
        h: H0/100.
    Returns:
        non-wiggle piece of the linear power spectrum.
    '''
    #ksmin(max): k-range and Nks: points
    ksmin = 7*10**(-5)/h; ksmax = 7/h; Nks = 2**16

    #sample ln(kP_L(k)) in Nks points, k range (equidistant)
    ksT = [ksmin + ii*(ksmax-ksmin)/(Nks-1) for ii in range(Nks)]
    PSL = interp(ksT, k, PSLk)
    logkpkT = np.log(ksT*PSL)
        
    #Discrete sine transf., check documentation
    FSTtype = 1; m = int(len(ksT)/2)
    FSTlogkpkT = dst(logkpkT, type = FSTtype, norm = "ortho")
    FSTlogkpkOddT = FSTlogkpkT[::2]
    FSTlogkpkEvenT = FSTlogkpkT[1::2]
        
    #cut range (remove the harmonics around BAO peak)
    mcutmin = 120; mcutmax = 240;
        
    #Even
    xEvenTcutmin = np.linspace(1, mcutmin-2, mcutmin-2)
    xEvenTcutmax = np.linspace(mcutmax+2, len(FSTlogkpkEvenT), len(FSTlogkpkEvenT)-mcutmax-1)
    EvenTcutmin = FSTlogkpkEvenT[0:mcutmin-2] 
    EvenTcutmax = FSTlogkpkEvenT[mcutmax+1:len(FSTlogkpkEvenT)]
    xEvenTcuttedT = np.concatenate((xEvenTcutmin, xEvenTcutmax))
    nFSTlogkpkEvenTcuttedT = np.concatenate((EvenTcutmin, EvenTcutmax))


    #Odd
    xOddTcutmin = np.linspace(1, mcutmin-1, mcutmin-1)
    xOddTcutmax = np.linspace(mcutmax+1, len(FSTlogkpkEvenT), len(FSTlogkpkEvenT)-mcutmax)
    OddTcutmin = FSTlogkpkOddT[0:mcutmin-1]
    OddTcutmax = FSTlogkpkOddT[mcutmax:len(FSTlogkpkEvenT)]
    xOddTcuttedT = np.concatenate((xOddTcutmin, xOddTcutmax))
    nFSTlogkpkOddTcuttedT = np.concatenate((OddTcutmin, OddTcutmax))

    #Interpolate the FST harmonics in the BAO range
    preT, = map(np.zeros,(len(FSTlogkpkT),))
    PreEvenT = interp(np.linspace(2, mcutmax, mcutmax-1), xEvenTcuttedT, nFSTlogkpkEvenTcuttedT)
    PreOddT = interp(np.linspace(0, mcutmax-2, mcutmax-1), xOddTcuttedT, nFSTlogkpkOddTcuttedT)


    ii = np.arange(m)
    mask = (mcutmin < ii + 1) & (ii + 1 < mcutmax)
    not_mask = ~mask
    
    preT = np.empty_like(FSTlogkpkT)
    
    # Use interpolated values inside the BAO cut range
    preT[2*ii[mask]+1] = PreEvenT[ii[mask]]
    preT[2*ii[mask]]   = PreOddT[ii[mask]]
    
    # Use original values outside the BAO cut range
    preT[2*ii[not_mask]+1] = FSTlogkpkT[2*ii[not_mask]+1]
    preT[2*ii[not_mask]]   = FSTlogkpkT[2*ii[not_mask]]   
         
                
        
    #Inverse Sine transf.
    FSTofFSTlogkpkNWT = idst(preT, type = FSTtype, norm = "ortho")
    PNWT = np.exp(FSTofFSTlogkpkNWT)/ksT

    PNWk = interp(k, ksT, PNWT)
    DeltaAppf = k*(PSL[7]-PNWT[7])/PNWT[7]/ksT[7]

    irange1 = np.where((k < 1e-3))
    PNWk1 = PSLk[irange1]/(DeltaAppf[irange1] + 1)

    irange2 = np.where((1e-3 <= k) & (k <= ksT[len(ksT)-1]))
    PNWk2 = PNWk[irange2]
        
    irange3 = np.where((k > ksT[len(ksT)-1]))
    PNWk3 = PSLk[irange3]
        
    PNWkTot = np.concatenate((PNWk1, PNWk2, PNWk3))
        
    return(k, PNWkTot)


