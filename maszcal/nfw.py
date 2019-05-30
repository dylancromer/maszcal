import numpy as np




class SimpleDeltaSigma:
    #TODO: fix shapes via broadcasting
    def __init__(self, cosmo_params, zs, rhocrit_of_z_func):
        self.cosmo_params = cosmo_params
        self.zs = zs

        self.rhocrit_of_z = rhocrit_of_z_func

    def rdel(self, mass, z, delta, mode):
        if mode == 'seljak':
            ans = (3 * mass / (4 * np.pi * delta * self.rhocrit_of_z(z)))**(1.0/3.0)

        elif mode == 'duffy':
            rho_m0 = self.cosmo_params.omega_matter
            ans = (3 * mass / (4 * np.pi * delta * rho_m0))**(1.0/3.0)

        else:
            raise ValueError('mode must be seljak or duffy')

        return ans

    def concentration(self, virial_mass, z, mode):
        if mode == 'seljak': #Seljak 2000 with hs in units
            ans = 5.72 / (1 + z) * (viral_mass / 10**14)**(-0.2)

        elif mode == 'duffy': #Duffy 2008 with hs in units
            ans = 5.09 / (1 + z)**0.71 * (virial_mass / 10**14)**(-0.081)

        elif mode == 'duffy_alt':  #Duffy 2008 with hs in units MEAN DENSITY 200
            ans = 10.14 / (1 + z)**(1.01) * (virial_mass / 2e12)**(-0.081)

        else:
            raise ValueError('mode must be seljak, duffy, or duffy_alt')

        return ans

    def mass_concentration_del_2_del_mean200(self, mdel, delta, z, EPS):
        mass = 2 * mdel
        rdels = self.rdel(mdel, z, delta, 'seljak') * (1. + z)
        ans = 0

        while np.any(np.abs(ans/mass - 1)) > EPS :
            ans = mass
            conz = self.concentration(mass, z, 'duffy') #DUFFY
            rs = self.rdel(mass, z, 200, 'duffy')/conz
            xx = rdels / rs
            assert False, 'I am in a loop'
            mass = mdel * self.m_x(conz) / self.m_x(xx)

        assert False, 'I got out of the loop'

        return ans

    def rho_s(self, c_delta, delta,z):
        return c_delta**3 * cosmo_params['rhom_0mpc'] * delta / 3. / (np.log(1.+c_delta)-c_delta/(1.+c_delta))

    def m_x(self, x):
        ans = np.log(1 + x) - x/(1+x)
        return ans

    def delta_sigma_of_mass(self, r, m_delta, delta):
        ERRTOL = 1e-6

        mass = self.mass_concentration_del_2_del_mean200(m_delta, delta, self.zs, ERRTOL)
        c_delta = self.concentration(mass, self.zs, 2) #DUFFY
        r_s = self.rdel(mass, self.zs, 200, 'duffy')/c_delta

        x = r/r_s
        x_ltone = x[x < 1]
        x_eqone = x[np.equal(x, 1.)]
        x_gtone = x[x > 1]

        fact = r_s * self.rho_s(c_delta, 200 , self.zs)

        delta_sigma_ltone = fact * (8 * np.arctanh(np.sqrt((1 - x_ltone)/(1 + x_ltone))) / (x_ltone**2 * np.sqrt(1 - x_ltone**2))
                                    + 4 * np.log(x_ltone/2) / x_ltone**2
                                    - 2 / (x_ltone**2 - 1)
                                    + 4 * np.arctanh(np.sqrt((1 - x_ltone)/(1 + x_ltone))) / ((x_ltone**2 - 1) * np.sqrt(1 - x_ltone**2)))

        delta_sigma_eqone = fact * (10/3 + 4 * np.log(0.5)) * x_eqone

        delta_sigma_gtone = fact * (8 * np.arctan(np.sqrt((x_gtone - 1)/(1 + x_gtone)))/x_gtone**2 / np.sqrt(x_gtone**2 - 1)
                                    + 4 * np.log(x_gtone/2)/x_gtone**2
                                    - 2 / (x_gtone**2 - 1)
                                    + 4 * np.arctan(np.sqrt((x_gtone - 1)/(1 + x_gtone)))/(x_gtone**2 - 1)**1.5)

        delta_sigma = np.concatenate((delta_sigma_ltone, delta_sigma_eqone, delta_sigma_gtone))/10**12 # 10**12 converts [hM_sun/Mpc^2 to hM_sun/pc^2]

        return delta_sigma
