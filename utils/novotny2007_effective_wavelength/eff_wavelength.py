import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from scipy.interpolate import interp1d
from visualization.plotter import *

eps_datas = np.loadtxt("utils/novotny2007_effective_wavelength/eV_k.dat", delimiter=",")
au_coef = interp1d(1240 / eps_datas[:,0] * 1e-9, eps_datas[:,2], kind="linear")
ag_coef = interp1d(1240 / eps_datas[:,0] * 1e-9, eps_datas[:,1], kind="linear")

# Medium
medium = {
    'air': 1.0,
    'water': 1.72,
    }

def gold_material(wavelength):
    eps_inf = au_coef(wavelength)
    wavelength_plasmonic = 138e-9
    return eps_inf, wavelength_plasmonic

def silver_material(wavelength):
    eps_inf = ag_coef(wavelength)
    wavelength_plasmonic = 135e-9
    return eps_inf, wavelength_plasmonic

# Constants 
Gamma = 0.5772156649  # Euler-Mascheroni constant
zeta = 5 / 3 + 2 * Gamma 

### Calculations
def effective_wavelength(wavelength, R, material_func, medium_func):
    eps_inf, wavelength_plasmonic = material_func(wavelength)
    eps_s = medium_func
    
    a1 = calculate_a1(eps_inf, eps_s)
    a2 = calculate_a2(eps_inf, eps_s)
    z_lambda_fun = a1 + a2*wavelength/wavelength_plasmonic    
    
    reapeted_term = 4*np.pi**2 * eps_s * R**2 / wavelength**2 * z_lambda_fun**2
    
    sqrt_term = np.sqrt((reapeted_term) / (1 + reapeted_term))
    
    wav_eff = wavelength / np.sqrt(eps_s) * sqrt_term - 4*R
    
    # plt.plot(wavelength*1e9, a1, label="a1")
    # plt.plot(wavelength*1e9, a2, label="a2")
    # plt.legend()
    # plt.show()
    
    # wav_eff = 2 * np.pi * R * (a1 + a2*wavelength/wavelength_plasmonic) - 4*R
    return wav_eff

def calculate_a1(eps_inf, eps_s):
    term1 = (1 / 3) * np.exp(zeta) * (1 + np.sqrt(3 * zeta) / 2)
    term2 = (2 * (eps_inf + eps_s * np.exp(2 * zeta) / 2) / (3 * eps_s * np.exp(zeta))) * (1 + (np.sqrt(3) / 2) * ((1+zeta) / np.sqrt(zeta)))
    a1 = term1 - term2
    return a1

def calculate_a2(eps_inf, eps_s):
    term1 = (2 * np.sqrt(eps_inf + eps_s*np.exp(2*zeta)/2)) / (3*eps_s*np.exp(zeta))
    term2 = 1 + (np.sqrt(3)/2) * ((1+zeta)/np.sqrt(zeta))
    a2 = term1 * term2
    return a2


def main():
    wavelengths = np.linspace(400e-9, 1935e-9, 100)  # from 400 nm to 2000 nm
    
    R = 5e-9
    effective_wavelengths = effective_wavelength(wavelengths, R, gold_material, medium['air'])
    
    # eps_inf, wavelength_plasmonic = gold_material(wavelengths)
    # effective_wavelengths = 2*np.pi*R * (13.74 - 0.12*(eps_inf + medium['air']*141.04)/medium['air'] - 2/np.pi + wavelengths/wavelength_plasmonic*0.12*np.sqrt(eps_inf+medium['air']*141.04)/medium['air'])

    if False:
        plt.title("The optical constant k (extinction coefficient).\nP. B. Johnson and R. W. Christy, Phys. Rev. B 6, 4370 (1972).")
        plt.plot(1240 / eps_datas[:,0], eps_datas[:,1], label="Ag")
        plt.plot(1240 / eps_datas[:,0], eps_datas[:,2], label="Au")
        plt.xlabel('Wavelength [nm]')
        plt.xlim([400, 1935])
        plt.ylabel('k')
        plt.legend()
        plt.show()
        
    if False:
        ag_fit = ag_coef(wavelengths)
        au_fit = au_coef(wavelengths)
        
        plt.title("The optical constant k (extinction coefficient).\nFit.")
        plt.plot(1240 / eps_datas[:,0], eps_datas[:,1], label="Ag")
        plt.plot(1240 / eps_datas[:,0], eps_datas[:,2], label="Au")
        plt.plot(wavelengths * 1e9, ag_fit, ":", label="Ag FIT")
        plt.plot(wavelengths * 1e9, au_fit, ":", label="Au FIT")
        plt.xlabel('Wavelength [nm]')
        plt.xlim([400, 1935])
        plt.ylabel('k')
        plt.legend()
        plt.show()

    if False:
        for R in [5, 10, 20]:
            R = R * 1e-9
            effective_wavelengths = effective_wavelength(wavelengths, R, gold_material, medium['air'])
            plt.plot(wavelengths * 1e9, effective_wavelengths * 1e9, label="R={} nm".format(R*1e9))
        plt.xlabel('Wavelength [nm]')
        plt.xlim([400, 1935])
        plt.ylabel('Effective Wavelength [nm]')
        plt.legend()
        plt.title(r"$\lambda_{eff}$ vs $\lambda$ for Gold in Air")
        plt.show()
        
        for R in [5, 10, 20]:
            R = R * 1e-9
            effective_wavelengths = effective_wavelength(wavelengths, R, silver_material, medium['air'])
            plt.plot(wavelengths * 1e9, effective_wavelengths * 1e9, label="R={} nm".format(R*1e9))
        plt.xlabel('Wavelength [nm]')
        plt.xlim([400, 1935])
        plt.ylabel('Effective Wavelength [nm]')
        plt.legend()
        plt.title(r"$\lambda_{eff}$ vs $\lambda$ for Silver in Air")
        plt.show()
        
    if True:
        fig, ax1 = plt.subplots()
        for R in [5, 10, 20]:
            R = R * 1e-9
            effective_wavelengths = effective_wavelength(wavelengths, R, silver_material, medium['air'])
            ax1.plot(wavelengths * 1e9, effective_wavelengths * 1e9, label="Ag R={} nm".format(R*1e9))
            effective_wavelengths = effective_wavelength(wavelengths, R, gold_material, medium['air'])
            ax1.plot(wavelengths * 1e9, effective_wavelengths * 1e9, ":", label="Au R={} nm".format(R*1e9))
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_xlim([400, 1935])
        ax1.set_ylabel('Effective Wavelength [nm]')
        ax1.legend()
        ax1.set_title("Effective Wavelength and Antenna Length\nvs Wavelength in Air")
        
        # Create second y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Length [nm]')
        # Set the limits for the second axis based on the first axis
        y1_min, y1_max = ax1.get_ylim()
        ax2.set_ylim(y1_min / 2.0, y1_max / 2.0)
        
        plt.show()
        
    ##################################################################################################
    # Comment:                                                                                       #
    #     Right now we can construct our half-wave dipole antenna using the effective wavelength     #
    #     by the formula: L = lambda_eff / 2.                                                        #
    #     The resonant peak should appear at wavelength lambda corelated to the effective wavelength.#
    # Scheme:                                                                                        #
    #     We choose a lambda, now we calculate lambda_eff using the above code,                      #
    #     then we set the antenna length L = lambda_eff / 2.                                         #
    ##################################################################################################
            
if __name__ == "__main__":
    main()