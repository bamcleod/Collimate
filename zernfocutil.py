import numpy as np
import pylab as plt

from numpy import sin,cos

class ZernikeFringe():

    def __init__(self, resolution, nmax, pupil_diameter, obscuration_ratio):
        
        self.phases = np.zeros((nmax, resolution, resolution))
        self.dx = np.zeros((nmax, resolution, resolution))
        self.dy = np.zeros((nmax, resolution, resolution))

        size = resolution+2
        y,x = (np.indices((size,size)) - size//2) / float(resolution//2)
        r = np.sqrt(x*x+y*y)
        self.pupil_r = r[1:-1,1:-1]
        self.in_pupil = (self.pupil_r < 1) * (self.pupil_r > obscuration_ratio)

        # Build up the set of zernike functions and their derivatives
        #
        for i in range(nmax):
            zernike_coefs = np.zeros(i+1)
            zernike_coefs[i] = 1
            w = self.zerneval_fringe(x, y, zernike_coefs)
            self.phases[i] = w[1:-1,1:-1]
            self.dx[i] = (w[1:-1,2:] - w[1:-1,:-2]) / (pupil_diameter/resolution) / 2
            self.dy[i] = (w[2:,1:-1] - w[:-2,1:-1]) / (pupil_diameter/resolution) / 2
            
    def slopes_arcsec(self, coefficient_vector):

        xslopes = coefficient_vector.dot(self.dx.reshape(self.dx.shape[0], -1)).reshape(self.dx[0].shape) * 206265
        yslopes = coefficient_vector.dot(self.dy.reshape(self.dy.shape[0], -1)).reshape(self.dy[0].shape) * 205265

        return xslopes[self.in_pupil], yslopes[self.in_pupil]

    def ee80diameter(self, xslopes, yslopes):
        xcen = xslopes.mean()
        ycen = yslopes.mean()
        radial_error = np.sqrt((xslopes-xcen)**2 + (yslopes-ycen)**2)
        radial_sorted = np.sort(radial_error)
        index80 = int(0.8 * len(radial_error))
        return 2 * radial_sorted[index80]
    

    def zerneval_fringe(self, x, y, zc):

        grandtot = np.zeros(x.shape)

        n = len(zc) + 1
        r = np.sqrt(r2 := x*x + y*y)
        phi = np.arctan2(y,x)

        if (n:=n-1): grandtot += zc[0] 
        else: return grandtot
        if (n:=n-1): grandtot += zc[1] * (rcos := r * cos(phi))
        else: return grandtot
        if (n:=n-1): grandtot += zc[2] * (rsin := r * sin(phi))
        else: return grandtot
        if (n:=n-1): grandtot += zc[3] * (2 * r2 -1)
        else: return grandtot
        if (n:=n-1): grandtot += zc[4] * (r2cos2 := r2 * cos(2 * phi))
        else: return grandtot
        if (n:=n-1): grandtot += zc[5] * (r2sin2 := r2 * sin(2 * phi))
        else: return grandtot
        if (n:=n-1): grandtot += zc[6] * (t6 := 3 * r2 - 2) * rcos
        else: return grandtot
        if (n:=n-1): grandtot += zc[7] *  t6               * rsin
        else: return grandtot
        if (n:=n-1): grandtot += zc[8] * (6 * (r4:= r2 * r2) - 6 * r2 + 1)
        else: return grandtot
        if (n:=n-1): grandtot += zc[9] * (r3cos3 := (r3 := r2*r) * cos(3 * phi))
        else: return grandtot
        if (n:=n-1): grandtot += zc[10] * (r3sin3 := r3 * sin(3 * phi))
        else: return grandtot
        if (n:=n-1): grandtot += zc[11] * (t11 := 4*r2-3)* r2cos2
        else: return grandtot
        if (n:=n-1): grandtot += zc[12] *  t11          * r2sin2
        else: return grandtot
        if (n:=n-1): grandtot += zc[13] * (t13 := 10 * r4 - 12 * r2 + 3) * rcos
        else: return grandtot
        if (n:=n-1): grandtot += zc[14] *  t13                          * rsin
        else: return grandtot
        if (n:=n-1): grandtot += zc[15] * (20*(r6:=r4*r2)-30*r4+12*r2-1)
        else: return grandtot
        if (n:=n-1): grandtot += zc[16] * (r4cos4 := r4 * cos(4 * phi))
        else: return grandtot
        if (n:=n-1): grandtot += zc[17] * (r4sin4 := r4 * sin(4 * phi))
        else: return grandtot
        if (n:=n-1): grandtot += zc[18] * (t18 := (5 * r2 - 4)) * r3cos3
        else: return grandtot
        if (n:=n-1): grandtot += zc[19] *  t18                 * r3sin3
        else: return grandtot
        if (n:=n-1): grandtot += zc[20] * (t20 := 15 * r4 - 20 * r2 + 6) * r2cos2
        else: return grandtot
        if (n:=n-1): grandtot += zc[21] *  t20                          * r2sin2
        else: return grandtot
        if (n:=n-1): grandtot += zc[22] * (t22 := 35*(r6:=r3*r3)-60*r4+30*r2-4)* rcos
        else: return grandtot
        if (n:=n-1): grandtot += zc[23] *  t22                               * rsin
        else: return grandtot
        if (n:=n-1): grandtot += zc[24] * (70*(r8:=r4*r4)-140*r6+90*r4-20*r2+1)
        else: return grandtot
        if (n:=n-1): grandtot += zc[25] * (r5:=r3*r2) * cos(5 * phi)
        else: return grandtot
        if (n:=n-1): grandtot += zc[26] *  r5        * sin(5 * phi)
        else: return grandtot
        if (n:=n-1): grandtot += zc[27] * (t27 := (6 * r2 - 5) * r4) * cos(4 * phi)
        else: return grandtot
        if (n:=n-1): grandtot += zc[28] *  t27                      * sin(4 * phi)
        else: return grandtot
        if (n:=n-1): grandtot += zc[29] * (t29 := 21*r4-30*r2+10) * r3cos3
        else: return grandtot
        if (n:=n-1): grandtot += zc[30] *  t29                   * r3sin3
        else: return grandtot
        if (n:=n-1): grandtot += zc[31] * (t31 := 56*r6-105*r4+60*r2-10) * r2cos2
        else: return grandtot
        if (n:=n-1): grandtot += zc[32] *  t31                          * r2sin2
        else: return grandtot
        if (n:=n-1): grandtot += zc[33] * (t33:=126*r8-280*r6+210*r4-60*r2+5)* rcos
        else: return grandtot
        if (n:=n-1): grandtot += zc[34] *  t33                              * rsin
        else: return grandtot
        if (n:=n-1): grandtot += zc[35] * (252*(r10:=r8*r2)-630*r8+560*r6-210*r4+30*r2-1)
        else: return grandtot
        if (n:=n-1): grandtot += zc[36] * (924*r6*r6-2772*r10+3150*r8-1680*r6+420*r4-42*r2+1)
        return grandtot

if __name__=="__main__":
    nzern = 15
    zf = ZernikeFringe(64, nzern, 1.3, 0.5)


    for i in range(nzern):
        zernike_coefs = np.zeros(nzern)
        zernike_coefs[i] = 1.e-6
        xslopes, yslopes = zf.slopes_arcsec(zernike_coefs)

        plt.figure()
        plt.plot(xslopes, yslopes, '.')
        plt.gca().set_aspect('equal')
        print(i,zf.ee80diameter(xslopes,yslopes))

    plt.show()


