import astropy.io.fits as fits
import numpy as np
from pyds9 import DS9
from io import StringIO
import subprocess
from astropy.table import Table
from collections import defaultdict
import re
import pylab as plt
import yaml
import os
from pathlib import Path
import zernfocutil

#%%

class Collimate():
    def __init__(self, filename, configname):
        self.config = self.read_config(filename, configname)
        conf = self.config
        self.reconstructor = np.array([[conf['astig_tilty'], conf['astig_transx']],
                                       [conf['coma_tilty'], conf['coma_transx']]])
        self.boxsize = conf['boxsize']
        
        self.defocfit_starting_values = """defoc_scale         	{defoc_scale}
teldia              	{teldia}
obscureratio        	{obscureratio}
seeing              	1.5              
pixsize             	{pixsize}
nimage              	2                   
bestfocus           	{bestfocus}
waveradius          	25
image[0].intens     	{image0_intens:.1f}             
image[0].background 	0                   
image[0].filename   {image0_filename}        
image[0].focus      	{image0_focus}
image[1].intens     	{image1_intens:.1f}   
image[0].background 0                       
image[1].filename   	{image1_filename}      
image[1].focus      	{image1_focus}
nzern               	9                   
zernwave            	1e-6
"""

        self.defocfit_params_to_vary = """image[0].intens	       	10
image[0].x0         	0.1
image[0].y0         	0.1
image[1].intens	       	10
image[1].x0         	0.1
image[1].y0         	0.1
bestfocus      1
seeing			0.01
zern[4]                 0.001                   
zern[5]                 0.001                   
zern[6]                 0.001                   
zern[7]                 0.001
zern[8]			0.001
defoc_scale		0.01
"""


    def read_config(self, filename, configname):
        with open(filename, 'r') as f:            
            config = yaml.safe_load(f)
        return config[configname]

    def determine_tilt_and_translation(self, astigx, astigy, comax, comay, suppress_printing=False):
        tilty, transx = np.linalg.lstsq(self.reconstructor, np.array([astigx, comax]),rcond=None)[0]
        tiltx, transy = np.linalg.lstsq(self.reconstructor, np.array([astigy, comay]),rcond=None)[0]
        tiltx = -(tiltx)  # To match the Zemax standard sign convention
        if not suppress_printing:
            print(f"Translation X: {transx:.3f} mm")
            print(f"Translation Y: {transy:.3f} mm")
            print(f"Tilt about Y:  {tilty: .3f} deg")
            print(f"Tilt about X:  {tiltx: .3f} deg")

        return transx, transy, tiltx, tilty

    def set_xy(self, xy_locations):
        self.xy_locations = xy_locations
        
    def extract_boxes_from_image(self, image_name):
        xy = self.xy_locations
        boxsize = self.boxsize
        boxes = []
        image = fits.getdata(image_name)
        for x,y in xy:
            xmin = int(x - boxsize//2)
            xmax = xmin + boxsize
            ymin = int(y - boxsize//2)
            ymax = ymin + boxsize
            box = image[ymin:ymax, xmin:xmax]
            boxes.append(box)
        return np.array(boxes)


    
    def fit_images(self, boxes_m, boxes_p, focus_m, focus_p):
        xy_locations = self.xy_locations
        
        imnum = 0
        results = []
        
        for box_m, box_p in zip(boxes_m, boxes_p):
    
            print(f"Processing {imnum}")
            
            image0_filename = f"m{imnum:02d}.fits"
            image1_filename = f"p{imnum:02d}.fits"
            
            starting_values = self.defocfit_starting_values.format(
                defoc_scale = self.config['defoc_scale'],
                teldia = self.config['teldia'],
                obscureratio = self.config['obscureratio'],
                pixsize = self.config['pix_size_arcsec'],
                image0_focus = focus_m,
                image1_focus = focus_p,
                bestfocus = (focus_m + focus_p) / 2,
                imnum = imnum, 
                image0_filename = image0_filename,
                image1_filename = image1_filename,
                image0_intens = box_m.sum(),
                image1_intens = box_p.sum())

            output_fits = f"pm_{imnum:02d}.fits"  # eg pm_00.fits
            output_out = f"pm_{imnum:02d}.out"    # e.g. pm_00.out
            output_stderr = f"pm{imnum:02d}.stderr"
    
            fits.writeto(image0_filename, box_m, overwrite=True)
            fits.writeto(image1_filename, box_p, overwrite=True)
    
            params = self.defocfit_params_to_vary.splitlines(keepends=True)
            with open("pm.params", "w") as file:
                for param in params:
                    p,val = param.split()
                    if ('fixed' in self.config) and (p in self.config['fixed']):
                        continue
                    file.write(param)
    
            # Save the input parameters
            with open("pm.in", 'w') as file:
                file.write(starting_values)
    
            # Fit the images
            
            
            defocfit = os.path.join(Path(__file__).parent,'Defocfit/defocfit')
            result = subprocess.run([defocfit, "pm.in", "pm.params", output_fits],
                                    stdout = subprocess.PIPE, text=True, stderr=subprocess.PIPE)
    
            with open(output_stderr, 'w') as file:
                file.write(result.stderr)
    
    
            # Save the fitted parameters
            with open(output_out, 'w') as file:
                file.write(result.stdout)
    

            str = result.stdout
            str += f"xpos\t{xy_locations[imnum][0]}\n"
            str += f"ypos\t{xy_locations[imnum][1]}\n"
            
            results.append(str)
            
            imnum += 1
            
        results_table = process_multiline_strings(results)
        
        columns_to_keep = ["xpos", "ypos", "zern_4", "bestfocus",
                       "zern_5", "zern_6", "zern_7", "zern_8"]
        
        new_columns = {col: results_table[col].astype(float) for col in columns_to_keep}

        
        return Table(new_columns)
    
    def fit_aberrations(self, aberration_table):
        #%%  Fit aberrations
        #plt.close('all')
        
        #t = Table.read("/home/bmcleod/src/CollTAOS/pm.tab", format="ascii")
        t = aberration_table
        
        pixel_size_mm = self.config['pix_size_mm']
        x0 = self.config['center_pix_x']
        y0 = self.config['center_pix_y']
        
    
        thetax = (t['xpos'] - x0) * pixel_size_mm
        thetay = (t['ypos'] - y0) * pixel_size_mm
        one = np.ones(len(thetax))
        zero = np.zeros(len(thetax))
        zern_4 = t["zern_4"]
        zern_5 = t["zern_5"]
        zern_6 = t["zern_6"]
        zern_7 = t["zern_7"]
        bestfocus = t["bestfocus"]
        
        #Astigmatism
        A4 = np.vstack([thetax, -thetay, one, zero])
        A5 = np.vstack([thetay,  thetax, zero, one])
        
        # Solve for a static quadratic term
        A4 = np.vstack([thetax, -thetay, one, zero, thetax**2-thetay**2])
        A5 = np.vstack([thetay,  thetax, zero, one, 2*thetax*thetay])
        
        A45 = np.hstack([A4, A5])
        z45 = np.hstack([zern_4, zern_5])
        
        fit45 = np.linalg.lstsq(A45.T, z45, rcond=None)

        print("Astigmatism fit:")
        print(f"  Constant terms: {fit45[0][2]:.2f}, {fit45[0][3]:.2f}")
        print(f"  Linear terms @ 60mm: {fit45[0][0]*60:.2f}, {fit45[0][1]*60:.2f}")
        print(f"  Quadratic term @ 60mm: {fit45[0][4]*60*60:.2f}")
        
        model45 = fit45[0].dot(A45)
        residuals = z45 - model45
        z4_model = model45[:len(zern_4)]
        z5_model = model45[len(zern_4):]
        z4_res = zern_4 - z4_model
        z5_res = zern_5 - z5_model
        
        plotastig_multiple(thetax, thetay, 
                   [zern_4, z4_model, z4_res], 
                   [zern_5, z5_model, z5_res])

        
        #Focus
        A3 = np.vstack([thetax, thetay, one, thetax*thetax+thetay*thetay])
        fit3 = np.linalg.lstsq(A3.T, bestfocus, rcond=None)
        model3 = fit3[0].dot(A3)
        res3 = bestfocus - model3
        print("Focus fit")
        print(f"  Constant: {fit3[0][2]:.2f}")
        print(f"  Linear @ 60mm: {fit3[0][0]*60:.2f} {fit3[0][1]*60:.2f}")
        print(f"  Quadratic @ 60mm: {fit3[0][3]*60*60:.2f}")

        #Coma
        A6 = np.vstack([thetax, one])
        fit6 = np.linalg.lstsq(A6.T, zern_6, rcond=None)
        z6_model = fit6[0].dot(A6)
        z6_res = zern_6 - z6_model
        plt.figure()
        plt.plot(thetax, zern_6,'.')
        plt.plot(thetax, z6_model)
        plt.xlabel("Image X position [mm]")
        plt.title("X coma Zernike Fringe amplitude")
        plt.ylabel("Microns")
        
        A7 = np.vstack([thetay, one])
        fit7 = np.linalg.lstsq(A7.T, zern_7, rcond=None)
        z7_model = fit7[0].dot(A7)
        z7_res = zern_7 - z7_model
        plt.figure()
        plt.plot(thetay, zern_7,'.')
        plt.plot(thetay, z7_model)
        plt.xlabel("Image Y position [mm]")
        plt.ylabel("Microns")
        
        plt.title("Y coma Zernike Fringe amplitude")
        
        print("X coma fit")
        print(f"  Constant: {fit6[0][1]:.2f}")
        print(f"  Linear @ 60mm: {fit6[0][0]*60:.2f}")

        print("Y coma fit")
        print(f"  Constant: {fit7[0][1]:.2f}")
        print(f"  Linear @ 60mm: {fit7[0][0]*60:.2f}")

        self.comax = fit6[0][1]
        self.comay = fit7[0][1]
        
        self.astigx = fit45[0][0]
        self.astigy = fit45[0][1]
        
        self.fit45 = fit45
        self.fit6 = fit6
        self.fit7 = fit7
        
        
        
        size = 10
        field_array_y, field_array_x = (np.indices((size,size))/ (size//2) - 1) * 60
        ee80 = np.zeros((size,size))
        
        zernike_terms = self.compute_aberrations_in_field(field_array_x, field_array_y)
        
    def compute_aberrations_in_field(self, fldx, fldy, if_collimated=False):
        one = np.ones(fldx.shape)
        zero = np.zeros(fldx.shape)
        
        if not if_collimated:
            A7 = np.array([fldy, one])
            A6 = np.array([fldx, one])
            A4 = np.array([fldx, -fldy, one, zero, fldx**2-fldy**2])
            A5 = np.array([fldy,  fldx, zero, one, 2*fldx*fldy])
            
        else:
            A7 = np.array([fldy, zero])
            A6 = np.array([fldx, zero])
            A4 = np.array([zero, -zero, one, zero, fldx**2-fldy**2])
            A5 = np.array([zero,  zero, zero, one, 2*fldx*fldy])
            
        shape7=A7.shape
        print(shape7)
        z7 = self.fit7[0].dot(A7.reshape((shape7[0],-1))).reshape(shape7[1:])
    
        shape6=A6.shape
        z6 = self.fit6[0].dot(A6.reshape((shape6[0],-1))).reshape(shape6[1:])
        
        A45 = np.hstack([A4, A5])
        shape45=A45.shape
        model45 = self.fit45[0].dot(A45.reshape((shape45[0],-1))).reshape(shape45[1:])
        z4 = model45[:len(fldx)]
        z5 = model45[len(fldx):]
        
        return np.array([zero, zero, zero, zero, z4, z5, z6, z7])

    def make_image_quality_maps(self, size=50):

        field_array_y, field_array_x = (np.indices((size,size))/ (size//2) - 1) * self.config['max_field']
        
        for if_collimated in (True, False):
        
            zernike_terms = self.compute_aberrations_in_field(field_array_x, field_array_y, if_collimated=if_collimated)
            
        
            xslopes_list = [[[] for i in range(field_array_x.shape[1])] for j in range(field_array_x.shape[0])]
            yslopes_list = [[[] for i in range(field_array_x.shape[1])] for j in range(field_array_x.shape[0])]
            ee80 = np.zeros(field_array_x.shape)
            zf = zernfocutil.ZernikeFringe(64, 8, 1.3, 0.5)
            for iy in range(field_array_x.shape[0]):
                for ix in range(field_array_x.shape[1]):
                    
                    xslopes, yslopes = zf.slopes_arcsec((zernike_terms[:,iy,ix])*1e-6)
                    xslopes_list[iy][ix] = xslopes
                    yslopes_list[iy][ix] = yslopes
                    ee80[iy,ix] = zf.ee80diameter(xslopes,yslopes)
                    
            print(f"If collimated: {if_collimated} On-axis aberrations: {zernike_terms[:,size//2,size//2]}")     
            plt.figure()
            plt.imshow(ee80, extent=(field_array_x[0][0],field_array_x[-1][-1],field_array_y[0][0], field_array_y[-1][-1]))
            plt.colorbar()
            if (if_collimated):
                plt.title("Site2 EE80 diameter (arcsec) prediction after collimation")
            else:
                plt.title("Site2 EE80 diameter (arcsec) current")
            plt.xlabel("Detector X position [mm]")
            plt.ylabel("Detector Y position [mm]")
            plt.show()
        

    def show_stamp_layout_from_model_images(self, ds9):
        #Make stamp layout from model images
        model_p = []
        model_m = []
        
        for i in range(len(self.xy_locations)):
            f = fits.open(f"pm_{i:02d}.fits")
            model_m.append(f[1].data)
            model_p.append(f[4].data)
        
        labels, stamp_layout_m = make_stamp_layout(model_m, self.xy_locations, 30)
        ds9.set("frame 3")
        ds9.set_np2arr(stamp_layout_m)
        ds9.set("region",labels)
        
        labels, stamp_layout_p = make_stamp_layout(model_p, self.xy_locations, 30)
        ds9.set("frame 4")
        ds9.set_np2arr(stamp_layout_p)
        ds9.set("region",labels)
    


def plot_coma(zernike_data):
    
    xpos = zernike_data['xpos']
    ypos = zernike_data['ypos']
    zern_6 = zernike_data['zern_6']
    zern_7 = zernike_data['zern_7']
    
    plt.figure()
    plt.quiver(xpos, ypos, zern_6, zern_7)
    plt.gca().set_aspect('equal')
    plt.title("Coma")
    
    
def save_star_locations(filename, xy_locations):
    np.savetxt(filename, xy_locations)

def load_star_locations(filename):
    return np.loadtxt(filename)

def get_xy_from_ds9(ds9):
    regions_xy = ds9.get("regions -format xy")

    xy = np.atleast_2d(np.loadtxt(StringIO(regions_xy)))
    return xy

        
def process_multiline_strings(strings):
    data = defaultdict(list)

    for string in strings:
        for line in string.strip().splitlines():
            if not line.startswith("#"):  # Discard lines starting with '#'
                fields = line.split("\t")
                if len(fields) == 2:  # Ensure we have exactly 2 fields
                    s, value = fields
                    s = s.strip()
                    col_name = re.sub(r"\[(\d+)\]", r"_\1", s)  # Change [n] to _n
                    data[col_name].append(value)

    # Convert to Astropy Table
    table = Table(data)
    return table

#%%

def make_stamp_layout(boxes, xy_locations, magnification):
    boxsizey,boxsizex = boxes[0].shape
    normalize = 100000
    x_max = -1e9
    x_min = 1e9
    y_max = -1e9
    y_min = 1e9
    for x,y in xy_locations:
        x_new = int(x / magnification)
        if x_new < x_min: x_min = x_new
        if x_new + boxsizex > x_max: x_max = x_new + boxsizex
        y_new = int(y / magnification)
        if y_new < y_min: y_min = y_new
        if y_new + boxsizey > y_max: y_max = y_new + boxsizey

    xsize = x_max - x_min
    ysize = y_max - y_min

    new_image = np.zeros((ysize, xsize))
    boxnum = 0
    ds9_region_string = ""
    for box,(x,y) in zip(boxes, xy_locations):
        x0 = int(x / magnification) - x_min
        x1 = x0 + boxsizex
        y0 = int(y / magnification) - y_min
        y1 = y0 + boxsizey
        total_counts = box.sum()
        new_image[y0:y1,x0:x1] += box / total_counts * normalize
        
        new_image[y0,x0:x1] = -100
        new_image[y1-1,x0:x1] = -100
        new_image[y0:y1,x0] = -100
        new_image[y0:y1,x1-1] = -100
        
        ds9_region_string = ds9_region_string + f"# text({x0+5},{y0+5}) text = {{ {boxnum} }}\n"
        boxnum += 1
        
    return ds9_region_string, new_image




#%%
def quiver_astig(axis, xpos, ypos, zern_4, zern_5, color='black', label="Data", xlabelpos = 0.9, ylabelpos=1.05 ):
    astig_angle = np.arctan2(zern_5, zern_4) / 2
    astig_mag = np.sqrt(zern_5 * zern_5 + zern_4 * zern_4)

    q = axis.quiver(xpos, ypos,
               np.cos(astig_angle) * astig_mag,
               np.sin(astig_angle) * astig_mag,
               headaxislength = 0, headlength=0, pivot='middle', color=color,
               scale=1, scale_units='inches')
    axis.quiverkey(q, X=xlabelpos, Y=ylabelpos, U=1, label=label)
               
    return q
    
def plotastig(xpos, ypos, zern_4, zern5, title="Astigmatism"):
    plt.figure()
    q = quiver_astig(xpos, ypos, zern_4, zern_5)
    plt.quiverkey(q, 0.1, 1.1, 1.0 , "1 micron zernike fringe")
    plt.title(title)
    
def plotastig_multiple(xpos, ypos, z4list, z5list, title="Astigmatism"):
    fix, ax = plt.subplots(figsize=(5,5))
    q = quiver_astig(ax, xpos, ypos, z4list[0], z5list[0], color='blue', xlabelpos=0.3,label='Measured')
    q = quiver_astig(ax, xpos-1, ypos-1, z4list[1], z5list[1],color='green', xlabelpos = 0.5, label='Model')
    q = quiver_astig(ax, xpos+1, ypos+1, z4list[2], z5list[2],color='red', xlabelpos = 0.7, label='Resid')

    plt.title(title)
    

#%%

if __name__=="__main__":
        
    boxsize = 64
    ds9 = DS9()
    
    folder = "/data/piper0/bmcleod/2024-11-TAOS2/2024September_fits/site2/240928/fseq51 (final)/"
    
    
    xy_locations = get_xy_from_ds9(ds9)
    print("Got these positions: ", xy_locations)
    
    boxes_m = extract_boxes_from_image(folder + "mosaic_2072.fits", xy_locations)
    boxes_p = extract_boxes_from_image(folder + "mosaic_2096.fits", xy_locations)
    
    labels, stamp_layout_m = make_stamp_layout(boxes_m, xy_locations, 30)
    ds9.set("frame 3")
    ds9.set_np2arr(stamp_layout_m)
    ds9.set("region",labels)
    
    labels, stamp_layout_p = make_stamp_layout(boxes_p, xy_locations, 30)
    ds9.set("frame 4")
    ds9.set_np2arr(stamp_layout_p)
    ds9.set("region",labels)
    #%%
    #results = fit_images(boxes_m, boxes_p)
    
    results = []
    for i in range(14):
        filename = f"pm_{i:02d}.out"
        with open(filename, 'r') as file:
            str = file.read()
            str += f"xpos\t{xy_locations[i][0]}\n"
            str += f"ypos\t{xy_locations[i][1]}\n"
            
            results.append(str)
            
    
    results_table = process_multiline_strings(results)
    
    
    
    columns_to_keep = ["xpos", "ypos", "zern_4", "bestfocus",
                       "zern_5", "zern_6", "zern_7", "zern_8"]
    
    filtered_table = results_table[columns_to_keep]
    
    print(filtered_table)
    
    filtered_table.write('pm.tab', format='ascii', overwrite=True)
    
    xsize = 9600
    ysize = 9216
    x0 = xsize // 2
    y0 = ysize // 2
    
    t = filtered_table
    
    xpos = t["xpos"].astype(float) - x0
    ypos = t["ypos"].astype(float) - y0
    zern_4 = t["zern_4"].astype(float)
    zern_5 = t["zern_5"].astype(float)
    zern_6 = t["zern_6"].astype(float)
    zern_7 = t["zern_7"].astype(float)
    bestfocus = t["bestfocus"].astype(float)
    
    #############
    # Coma X
    #############
    
    A = np.vstack([
        np.ones(len(xpos)),
        xpos])
    
    fit_6 = np.linalg.lstsq(A.T, zern_6, rcond = None)[0]
    print(fit_6)
    model_6 = fit_6.dot(A)
    plt.figure()
    plt.plot(xpos, zern_6)
    plt.plot(xpos, model_6)
    plt.title("Coma X")
    
    #############
    # Coma Y
    #############
    A = np.vstack([
        np.ones(len(ypos)),
        ypos])
    
    fit_7 = np.linalg.lstsq(A.T, zern_7, rcond = None)[0]
    print(fit_7)
    model_7 = fit_7.dot(A)
    
    plt.figure()
    plt.plot(ypos, zern_7)
    plt.plot(ypos, model_7)
    plt.title("Coma Y")
    
    #############
    # Focus
    #############
    
    A = np.vstack([
        np.ones(len(ypos)),
        xpos, ypos])
    
    fit_3 = np.linalg.lstsq(A.T, bestfocus, rcond = None)[0]
    print(fit_3)
    model_3 = fit_3.dot(A)
    resid = bestfocus - model_3
    plt.figure()
    plt.plot(ypos, bestfocus)
    plt.plot(ypos, model_3)
    plt.title("Focus")
    print(bestfocus, model_3, resid)
    
    #%%

    dist = np.sqrt((ypos-3000)**2 + (xpos)**2)
    plt.figure()
    plt.plot(dist, astig_mag, ".")
    plt.title("Astig magnitude")
    
    plt.show()
    
    #%%  Fit aberrations
    #plt.close('all')
    t = Table.read("/home/bmcleod/src/CollTAOS/pm.tab", format="ascii")
    
    pixel_size_mm = 0.015
    xsize = 9600
    ysize = 9216
    x0 = xsize // 2
    y0 = ysize // 2
    
    thetax = (t['xpos'] - x0) * pixel_size_mm
    thetay = (t['ypos'] - y0) * pixel_size_mm
    one = np.ones(len(thetax))
    zero = np.zeros(len(thetax))
    zern_4 = t["zern_4"].astype(float)
    zern_5 = t["zern_5"].astype(float)
    zern_6 = t["zern_6"].astype(float)
    zern_7 = t["zern_7"].astype(float)
    bestfocus = t["bestfocus"].astype(float)
    
    #Astigmatism
    A4 = np.vstack([thetax, -thetay, one, zero])
    A5 = np.vstack([thetay,  thetax, zero, one])
    
    # Solve for a static quadratic term
    A4 = np.vstack([thetax, -thetay, one, zero, thetax**2-thetay**2])
    A5 = np.vstack([thetay,  thetax, zero, one, 2*thetax*thetay])
    
    A45 = np.hstack([A4, A5])
    z45 = np.hstack([zern_4, zern_5])
    
    fit45 = np.linalg.lstsq(A45.T, z45, rcond=None)
    model45 = fit45[0].dot(A45)
    residuals = z45 - model45
    z4_model = model45[:len(zern_4)]
    z5_model = model45[len(zern_4):]
    z4_res = zern_4 - z4_model
    z5_res = zern_5 - z5_model
    
    #plotastig(thetax, thetay, z4_model, z5_model, title = "Modeled astig")
    #plotastig(thetax, thetay, zern_4, zern_5, title="Measured astig")
    #plotastig(thetax, thetay, z4_res, z5_res, title="Residual astig")
    
    #Focus
    A3 = np.vstack([thetax, thetay, one])
    fit3 = np.linalg.lstsq(A3.T, bestfocus, rcond=None)
    model3 = fit3[0].dot(A3)
    res3 = bestfocus - model3
    
    #Coma
    A6 = np.vstack([thetax, one])
    fit6 = np.linalg.lstsq(A6.T, zern_6, rcond=None)
    z6_model = fit6[0].dot(A6)
    z6_res = zern_6 - z6_model
    plt.figure()
    plt.plot(thetax, zern_6,'.')
    plt.plot(thetax, z6_model)
    plt.xlabel("Image X position [mm]")
    plt.title("X coma Zernike Fringe amplitude")
    plt.ylabel("Microns")
    
    A7 = np.vstack([thetay, one])
    fit7 = np.linalg.lstsq(A7.T, zern_7, rcond=None)
    z7_model = fit7[0].dot(A7)
    z7_res = zern_7 - z7_model
    plt.figure()
    plt.plot(thetay, zern_7,'.')
    plt.plot(thetay, z7_model)
    plt.xlabel("Image Y position [mm]")
    plt.ylabel("Microns")
    
    plt.title("Y coma Zernike Fringe amplitude")
    
    print("X coma fit", fit6[0])
    print("Y coma fit", fit7[0])
    
    comax = fit6[0][1]
    comay = fit7[0][1]
    
    astigx = fit45[0][0]
    astigy = fit45[0][1]
    
    
    
    size = 10
    field_array_y, field_array_x = (np.indices((size,size))/ (size//2) - 1) * 60
    ee80 = np.zeros((size,size))
    
    zernike_terms = compute_aberrations_in_field(field_array_x, field_array_y)
    
    
    #%%
    
    plotastig_multiple(thetax, thetay, 
                       [zern_4, z4_model, z4_res], 
                       [zern_5, z5_model, z5_res])
    #%%
        
    coll = Collimate("taos2.yaml", "TAOS2")
    
    print (coll.determine_tilt_and_translation(astigx, astigy, comax, comay))
    #%%
    
    
    plt.close('all')
    size = 50
    field_array_y, field_array_x = (np.indices((size,size))/ (size//2) - 1) * 60
    
    for if_collimated in (True, False):
    
        zernike_terms = compute_aberrations_in_field(field_array_x, field_array_y, if_collimated=if_collimated)
        zernike_terms[3] = 1.5
    
        xslopes_list = [[[] for i in range(field_array_x.shape[1])] for j in range(field_array_x.shape[0])]
        yslopes_list = [[[] for i in range(field_array_x.shape[1])] for j in range(field_array_x.shape[0])]
        ee80 = np.zeros(field_array_x.shape)
        zf = zernfocutil.ZernikeFringe(64, 8, 1.3, 0.5)
        for iy in range(field_array_x.shape[0]):
            for ix in range(field_array_x.shape[1]):
                
                xslopes, yslopes = zf.slopes_arcsec((zernike_terms[:,iy,ix])*1e-6)
                xslopes_list[iy][ix] = xslopes
                yslopes_list[iy][ix] = yslopes
                ee80[iy,ix] = zf.ee80diameter(xslopes,yslopes)
                
        print(f"If collimated: {if_collimated} On-axis aberrations: {zernike_terms[:,5,5]}")     
        plt.figure()
        plt.imshow(ee80, extent=(field_array_x[0][0],field_array_x[-1][-1],field_array_y[0][0], field_array_y[-1][-1]))
        plt.colorbar()
        if (if_collimated):
            plt.title("Site2 EE80 diameter (arcsec) prediction after collimation")
        else:
            plt.title("Site2 EE80 diameter (arcsec) current")
        plt.xlabel("Detector X position [mm]")
        plt.ylabel("Detector Y position [mm]")
        plt.show()
    
    
    #%%
    mm_per_arcsec = 0.015 / 0.6
    magnification = 30
    field_array_x = thetax
    field_array_y = thetay
    zernike_terms = compute_aberrations_in_field(field_array_x, field_array_y, if_collimated=if_collimated)
    defoc_amount = 0.2 * 12
    for defocus in [-defoc_amount, defoc_amount]:
        zernike_terms[3] = defocus 
    
        plt.figure()
        zf = zernfocutil.ZernikeFringe(64, 8, 1.3, 0.5)
        
        for i in range(len(thetax)):
            fldx = thetax[i]
            fldy = thetay[i]
            zern = zernike_terms[:,i]
            with np.printoptions(precision=2,suppress=True):
                print (f"Star {i:2d} Field {fldx:5.1f}, {fldy:5.1f} ,{zern}")
            xslopes, yslopes = zf.slopes_arcsec(zern*1e-6)
    
            #ee80[iy,ix] = zf.ee80diameter(xslopes,yslopes)
            
            xmag = xslopes * mm_per_arcsec * magnification + fldx
            ymag = yslopes * mm_per_arcsec * magnification + fldy
            plt.plot(xmag, ymag,',')
        plt.gca().set_aspect('equal')
    plt.show()
    
