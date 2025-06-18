#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:25:12 2024

@author: bmcleod
"""
import numpy as np
import pylab as plt
import copy
import yaml

class zernike_vs_field():
    def __init__(self, filename, reverse=False):
        
        if filename is not None:
            self.read_zernike_field_file(filename)
            if reverse:
                self.fields *= -1
            
        
    def __sub__(self, other):
        if not np.array_equal(self.fields, other.fields):
            raise ValueError("Both operands must have the same set of fields")
        new = copy.deepcopy(self)
        for key in new.zern:
            new.coeffs[key] = self.coeffs[key] - other.coeffs[key]
        return new
        
    def read_zernike_field_file(self, filename):
        coeffs = {}
        with open(filename, "r") as file:
            lines = file.read().splitlines()
        count = 0
        for line in lines:
            count += 1
            tokens = line.split()
            if len(tokens) == 0:
                continue
            if tokens[0] == "Wavelength:": 
                wavelength = float(tokens[1])
            if tokens[0] == "Field:":
                zerns = np.array(tokens[1:]).astype(int)
            
                #print("Found Field on line", count)
                #print("Zernikes: ", zerns)
                break
        a = np.genfromtxt(lines[count:]).astype(float).T / wavelength
        #print(a)
        fields = a[0]
        for i in range(len(zerns)):
            coeffs[zerns[i]] = a[i+1]
        self.zern = zerns
        self.fields = fields
        self.coeffs = coeffs
        return
    
    def concat(self, new):
        self.fields = np.concatenate((self.fields, new.fields))
        argsort = np.argsort(self.fields)
        for i in range(len(self.zern)):
            izern = self.zern[i]
            self.coeffs[izern] = np.concatenate((self.coeffs[izern], new.coeffs[izern]))[argsort]
        self.fields = self.fields[argsort]

    def plot(self, title=None):
        #
        plt.figure()
        for i in range(len(self.zern)):
            izern = self.zern[i]
            plt.plot(self.fields, self.coeffs[izern], label=str(izern))
        plt.title(title)
        plt.legend()

class Aberrations():
    def __init__(self):
        self.displacements = ["aligned", "transx", "tilty"]
        
        # term: [list of polynomial terms to use for fitting zernikes]
        # e.g. for 4, focus, fit even terms up to 4
        # 5: astigmatism, which must go to zero on axis, and be even, use terms 2 and 4
        # 7: coma, must be an odd function
        # 9: spherical aberration is even
        # 10: trefoil is odd
        self.powerdict={
        4: [0,2,4],
        5: [2,4],
        7: [1,3,5],
        9: [0,2,4],
        10: [1,3]
        }
        
        # term, cos term, angular order
        # e.g. 6 is the sin astigmatism term, so it points to 5 the cosine 
        #                 version, and has angular order of 2
        self.allterms=[
            [4,   4, 0],
            [5,   5, 2],
            [6,   5, 2],
            [7,   7, 1],
            [8,   7, 1],
            [9,   9, 0],
            [10, 10, 3],
            [11, 10, 3]
        ]
        self.maxcoeff = 5
    
        
    def read_zernike_data(self, folder):
        results = {}
        for disp in self.displacements:
            for axis in ['x', 'y']:
                results[f"{disp}_{axis}"] = zernike_vs_field(folder+f"{disp}+{axis}.txt")
                minus=zernike_vs_field(folder+f"{disp}-{axis}.txt", reverse=True)
                results[f"{disp}_{axis}"].concat(minus)
                
                if disp!="aligned":
                    results[f"{disp}_{axis}_diff"] = results[f"{disp}_{axis}"] - results[f"aligned_{axis}"]
                    results[f"{disp}_{axis}_diff"].plot(title=f"Displacement: {disp} seen along {axis} axis, static removed")
                else:
                    results[f"{disp}_{axis}"].plot(title=f"Aligned seen along {axis} axis")
        self.results = results

    def fit_static_terms(self):
        fld = self.results["aligned_x"].fields
        t = self.results["aligned_x"].coeffs
        self.cos = {}
        self.sin = {}
        self.order = {}
        self.coefftable = {}
        
        for zterm,parent,order in self.allterms:    
            powers=self.powerdict[parent]
            term=parent
            if term not in t: continue
            nterms=len(powers)
            ndata=len(fld)
            a=np.zeros((nterms,ndata))
            for i,p in enumerate(powers):
                a[i]=(fld)**p
        
            x,resid,rank,s=np.linalg.lstsq(a.transpose(),t[term],rcond=None)
        
            coeffs=np.zeros((self.maxcoeff+1))
            coeffs[powers] = x
            self.cos[zterm] = 1 if zterm==parent else 0
            self.sin[zterm] = 0 if zterm==parent else 1
            self.order[zterm] = order
            self.coefftable[zterm] = coeffs.copy()
            #cs="\t".join("%.8g" % item for item in coeffs)
            #print ("%d\t%d\t%d\t%d\t%s" % (zterm, 
            #                               self.cos[zterm],
            #                               self.sin[zterm],
            #                               self.order[zterm],
            #                               cs))

    
    def eval_coeffs(self, fldx, fldy):
        """
        Compute the static zernike aberrations for an arbitrary field point

        Returns
        -------
        Dictionary of zernike terms.  The value has the same shape as fldx

        """
        # Convert to polar coords
        ang=np.arctan2(fldy,fldx)
        r=np.sqrt(fldx*fldx+fldy*fldy)
        results = {}
        for zterm,coeffs in self.coefftable.items():
            c = self.cos[zterm]
            s = self.sin[zterm]
            order = self.order[zterm]
        
            tot = 0.0 * fldx
            n=0
            for coeff in coeffs:
               tot += coeff * r**n
               n+=1
            if order!=0:
                tot*=(c*np.cos(ang*order) + s*np.sin(ang*order))
            results[zterm] = tot
        return results

        
    def plot_contributions(self, fields, fit_terms, title, data):
        deg = len(fit_terms)
        plt.figure()
        plt.plot()
        plt.title(title)
        total = np.zeros(len(fields))

        for iterm in range(deg):
            term = fit_terms[-(iterm+1)] * fields**iterm
            total += term
            plt.plot(fields, term, label=str(iterm))
        plt.plot(fields, data, '.',label="Data")

        plt.plot(fields, total, label="Total")
        
        plt.legend()      
        
          
    def compute_collimation_response_matrix(self):
        """
        Compute the response matrix to vertex tilts and translations of the secondary.
        Resulting 2x2 numpy array is saved as self.response_matrix.
        The units are as follows:
            astig_tilty: microns of astigmatism zernike fringe per mm of field per degree of tilt
            astig_transx: microns of astigmatism zernike fringe per mm of field per mm of translation
            coma_tilty: microns of coma zernike fringe per degree of tilt
            coma_trans: microns of coma zernike fringe per mm of translation

        Returns
        -------
        2x2 numpy array
            [[astig_tilty, astig_transx], [coma_tilty, coma_transx]]

        """
        astig_tilty = np.polyfit(self.results["tilty_x_diff"].fields, self.results["tilty_x_diff"].coeffs[5],1)[0]
        astig_transx = np.polyfit(self.results["transx_x_diff"].fields, self.results["transx_x_diff"].coeffs[5],1)[0]
        
        coma_tilty = self.results["tilty_x_diff"].coeffs[7].mean()
        coma_transx = self.results["transx_x_diff"].coeffs[7].mean()
        A = np.array([[astig_tilty, astig_transx], [coma_tilty, coma_transx]])
        self.response_matrix = A
        return self.response_matrix

if __name__=="__main__":
             
    aber = Aberrations()
    aber.read_zernike_data("/home/bmcleod/src/CollTAOS/Taos-pokes/")
    
    aber.fit_static_terms()
    
    xflds = np.array([0, 60, -60, 0, 0, 30, 30, -30, -30])
    yflds = np.array([0,  0,  0, 60, -60, 30, -30, 30, -30])
    
    static = aber.eval_coeffs(xflds, yflds)
    print(static)
    
    for zterm,zvalue in static.items():
        plt.figure()
        plt.plot(xflds, yflds, '.')
        for x,y,z in zip(xflds, yflds, zvalue):
            print(zterm, x, y, z)
            plt.text(x,y,f"{z:.2f}")
        plt.title(f"Zernike {zterm}")
        
    #%%
    
    # Coma neutral point:
    print (aber.results["tilty_x_diff"].coeffs[7].mean() / aber.results["transx_x_diff"].coeffs[7].mean() * 57.3 )
    

    
    print (astig_tilty / astig_transx * 57.3 )
    

    
    
    with open("taos2.yaml", "w") as f:
        print("TAOS2:", file=f)
        print(f"    astig_tilty: {astig_tilty:.5f}     # microns of astigmatism per mm of field per degree of tilt", file=f)
        print(f"    astig_transx: {astig_transx:.5f}    # microns of astigmatism per mm of field per mm of translation", file=f)
        print(f"    coma_tilty: {coma_tilty:.5f}       # microns of coma per degree of tilt", file=f)
        print(f"    coma_transx: {coma_transx:.5f}     # microns of coma per mm of translation", file=f)
    
    

