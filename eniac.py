#!/usr/bin/env python3
# Peter Lynch's ENIAC weather model "emulator" converted to python3 for ATMO 455
# Created 25 March 2022 by Sam Gardner <stgardner4@tamu.edu>

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime as dt
from os import path
from pathlib import Path

###########################################################
###########             ENIAC               ###############
###########################################################
###     Solve the Barotropic Vorticity Equation         ###
###        d/dt (Del^2(z)) = J(h*Del^2(z)+f,z)  (1)     ###
###   where h = g*m^2/f with map factor m.              ###
###   The formulation is the same as used by            ###
###   Charney, Fjortoft and von Neumann (1950) in       ###
###   the ENIAC integrations (Tellus, 2, 237-254).      ###
###   An alternative formulation of the BVE using       ###
###   the stream function is  used if the indicator     ###
###               StreamFunction = 1                    ###
###   It is formally identical to (1) above, but now    ###
###   z stands for the streamfunction and h has a       ###
###   different definition: h = m^2.                    ###
###   The initial data are as used in tho original      ###
###   integrations in 1950. The boundary conditions     ###
###   are formulated in the same way.                   ###
###########################################################
###   The Barotropic Vorticity Equation is approximated ###
###   by centered finite differences. The domain has    ###
###   M * N points.                                     ###
###   The boundary conditions are as described          ###
###   in the Tellus paper: z is given on the boundary   ###
###   and vorticity (xi) is given at inflow points.     ###
###   The time scheme is the leapfrog method.           ###
###   (the first step is a forward step).               ###
###   The quantity to be stepped forward is             ###
###                xi = Del^2(z)                        ###
###   Thus, after each time-step, it is necessary       ###
###   to solve a Poisson equation to get z.             ###
###   This is done by the Fourier Transform method,     ###
###   as in the Tellus paper. (For a description of     ###
###   the method, see Numerical Recipes Sec. 19.4.)     ###
###   A variety of initial conditions may be specified  ###
###   determined by the parameter Ncase (see below).    ###
###   The data were downloaded from the NCEP/NCAR       ###
###   Reanalysis site. See, for example,                ###
###          http://nomad3.ncep.noaa.gov/               ###
###                 cgi-bin/ftp2u_6p_r1.sh              ###
###   They were converted from GRIB to ASCII and        ###
###   saved in files with the name YYYYMMDDHH.asc       ###
###   where YYYYMMDDHH is the date and time of the      ###
###   analysis. Thus, for example, for Ncase=1,         ###
###   the file is 1949010503.asc.                       ###
###   The re-analysis is on a 2.5 x 2.5 degree grid.    ###
###   The data were recovered on a domain covering      ###
###   the northern hemisphere. The fields were          ###
###   interpolated to the 19 x 16 polar stereographic   ###
###   grid used by Charney et al, and the results       ###
###   saved in files with the names YYYYMMDDHH.z00      ###
###   The four cases in question are (all in 1949):     ###
###     (1) 0300 Z, January 5th, 1949                   ###
###         with verification analysis Jan 6            ###
###     (2) 0300 Z, January 30th, 1949                  ###
###         with verification analysis Jan 31           ###
###     (3) 0300 Z, January 31st, 1949                  ###
###         with verification analysis Feb 1            ###
###     (4) 0300 Z, February 13th, 1949                 ###
###         with verification analysis Feb 14           ###
###   Note: Charney et al. used the analyses valid      ###
###   at 0300 Z. These were not immediately available   ###
###   from the re-analysis site.                        ###
###   Thanks to Chi-Fan Shih for assistance in          ###
###   acquiring these 0300Z fields.                     ###
### Ncase    Analysis file        Verification file     ###
###  1    Case1-1949010503.z00   Case1-1949010603.z00   ###
###  2    Case2-1949013003.z00   Case2-1949013103.z00   ###
###  3    Case2-1949013103.z00   Case2-1949020103.z00   ###
###  4    Case2-1949002133.z00   Case2-1949021403.z00   ###
###########################################################
### Date: February, 2007                                ###
### Author: Peter Lynch, UCD Met & Climate Centre       ###
###                      University College Dublin      ###
### Email:  Peter.Lynch@ucd.ie                          ###
###########################################################

## Parameters
colormap = "rainbow" # Set matplotlib colormap (expected: string name of matplotlib colormap, recommended "rainbow", "plasma", or "viridis")
Ncase = 1 # Forecast case number (expected: 1, 2, 3, or 4)
DAYLEN = 1 # Forecast length in days (expected: integer)
DtHours = 1 # Time step in hours (expected: integer)
StreamFunction = 0 # Indicator for psi-form of equation (expected: 0 or 1)
Corners = False # Whether or not to record lat/lon of the outer and inner domain (expected: bool)
PlotAll = False # Whether or not to make debug plots for map factor, coriolis parameter, and h factor (expected: bool)
PrintStep = False # Whether or not to print tracking information during the main loop run. (expected: bool)
PlotSteps = False # Whether or not to plot the hieght field at each time step, significantly increases runtime (expected: bool)
PlotEnergyEnstrophy = True # Whether or not to plot the integrals of energy and enstrophy (expected: bool)

## Set the PS grid parameters for 4 cases.
NUMx = np.array([19, 19, 19, 19]) # 19 points in the x-direction
NUMy = np.array([16, 16, 16, 16]) # 16 points in the y-direction
Xpol = np.array([10, 10, 10, 10]) # x-coordinate of the North Pole
Ypol = np.array([14, 12, 14, 14]) # y-coordinate of the North Pole
DELs = np.array([736000, 736000, 736000, 736000]) # 736 km grid step at the North Pole
Cang = np.array([-90, -70, -35, -85]) # Angle between date-line and positive y-axis

## Select the parameters for the forecast (Ncase)
Ncase = Ncase - 1 
M = NUMx[Ncase] # Points in the x direction
N = NUMy[Ncase] # Points in the y direction
Xp = Xpol[Ncase] # Coordinates of the North Pole
Yp = Ypol[Ncase] # Coordinates of the North Pole
Ds = DELs[Ncase] # Grid step at North Pole (meters)
Centerangle = Cang[Ncase] # Center angle of map

MN = M*N # Total number of grid points

## Define the spatial grid (not used in this form)
# x = np.arange(0, M)*Ds
# y = np.arange(0, N)*Ds

## Define the (X,Y) grid (for plotting)
(X, Y) = np.meshgrid(range(1, M+1), range(1, N+1))
X = X.transpose()
Y = Y.transpose()

## Define the time variable
daylen = DAYLEN
seclen = daylen*24*60*60
Dt = DtHours*60*60
nt = int(seclen/Dt)

t = [_*Dt for _ in range(0, int(nt)+1)]
time = [int(_/(60*60)) for _ in t]
nspd = (24*60*60)/Dt

## Print out information on space and time grid
print("Ncase = "+str(Ncase+1))
print("Grid size, M="+str(M)+" N="+str(N))
print("Total grid points, M*N="+str(MN))
print("Timesteps per day, nspd="+str(nspd))
print("Stream function, "+str(StreamFunction))

## Define some geophysical constants (SI units)
a = (4*10**7)/(2*np.pi) #m, radius of the earth
grav = 9.80665 # m/s/s, gravity
Omega = 2*np.pi/(24*60*60) #radians/sec, angular velocity of the earth
f0 = 2*Omega*np.sin(np.pi/4) #dimensionless, mean value of the coriolis parameter

## Compute latitude and longitude on the polar stereographic grid.
LON = np.zeros((M, N))
LAT = np.zeros((M, N))
LONDEG = np.zeros((M, N))
LATDEG = np.zeros((M, N))
for ny in range(1, N+1):
    for nx in range(1, M+1):
        xx = (nx-Xp)*Ds
        yy = (ny-Yp)*Ds
        rr = (xx**2+yy**2)**(0.5)
        theta = np.arctan2(yy,xx)
        lambduh = theta + np.deg2rad(90+Centerangle) # editor's/Sam's note: you cant name a variable "lambda" in python
        if lambduh > np.pi:
            lambduh = lambduh - 2*np.pi
        phi = 2*((np.pi/4)-np.arctan(rr/(2*a)))
        LON[nx-1, ny-1] = lambduh # Longitude (radians)
        LAT[nx-1, ny-1] = phi # Latitude (radians)
        lamd = np.rad2deg(lambduh)
        phid = np.rad2deg(phi)
        LONDEG[nx-1, ny-1] = lamd # Longitude (degrees)
        LATDEG[nx-1, ny-1] = phid # Latitude (degrees)

## Reord the latitude and longitude of the corners of the outer domain and of the inner domain (for checking purposes)
if Corners:
    latSW = LATDEG[0, 0]
    lonSW = LONDEG[0, 0]
    latSE = LATDEG[-1, 0]
    lonSE = LONDEG[-1, 0]
    latNE = LATDEG[-1, -1]
    lonNE = LONDEG[-1, -1]
    latNW = LATDEG[0, -1]
    lonNW = LONDEG[0, -1]
    print("Corner latitudes and longitudes, OUTER DOMAIN")
    print("Lat and long of SW corner  "+str(latSW)+" "+str(lonSW))
    print("Lat and long of SE corner  "+str(latSE)+" "+str(lonSE))
    print("Lat and long of NE corner  "+str(latNE)+" "+str(lonNE))
    print("Lat and long of NW corner  "+str(latNW)+" "+str(lonNW))
    latSW = LATDEG[2, 1]
    lonSW = LONDEG[2, 1]
    latSE = LATDEG[-3, 1]
    lonSE = LONDEG[-3, 1]
    latNE = LATDEG[-3, -3]
    lonNE = LONDEG[-3, -3]
    latNW = LATDEG[2, -3]
    lonNW = LONDEG[2, -3]
    print("Corner latitudes and longitudes, INNER DOMAIN")
    print("Lat and long of SW inner  "+str(latSW)+" "+str(lonSW))
    print("Lat and long of SE inner  "+str(latSE)+" "+str(lonSE))
    print("Lat and long of NE inner  "+str(latNE)+" "+str(lonNE))
    print("Lat and long of NW inner  "+str(latNW)+" "+str(lonNW))

## Compute coriolis parameter and map factor and parameter h = g*m^2/f used in the BVE (for psi-equation, h = m^2 is used)
MAP = np.zeros((M, N))
FCOR = np.zeros((M, N))
h = np.zeros((M, N))
for ny in range(0, N):
    for nx in range(0, M):
        lambduh = LON[nx, ny]
        phi = LAT[nx, ny]
        map = 2 / (1+np.sin(phi))
        f = 2*Omega*np.sin(phi)
        MAP[nx, ny] = map
        FCOR[nx, ny] = f
        if StreamFunction == 0:
            h[nx, ny] = (grav * map**2)/f 
        elif StreamFunction == 1:
            h[nx, ny] = map**2

px = 1/plt.rcParams["figure.dpi"]
basePath = path.realpath(path.dirname(__file__))
Path(path.join(basePath, "output", "steps")).mkdir(parents=True, exist_ok=True)
if PlotAll:
    debugFig = plt.figure()
    debugFig.set_size_inches(852*px, 480*px)
    debugSpec = GridSpec(2, 2, figure=debugFig)
    mapFactorAx = debugFig.add_subplot(debugSpec[0,0])
    mapFactorAx.contourf(X, Y, MAP, cmap=colormap)
    mapFactorAx.contour(X, Y, MAP, colors="black", linewidths=0.75)
    mapFactorAx.set_title("MAP FACTOR")
    mapFactorAx.set_aspect(1)
    coriAx = debugFig.add_subplot(debugSpec[0,1])
    coriAx.contourf(X, Y, FCOR, cmap=colormap)
    coriAx.contour(X, Y, FCOR, colors="black", linewidths=0.75)
    coriAx.set_title("f CORIOLIS")
    coriAx.set_aspect(1)
    factorHAx = debugFig.add_subplot(debugSpec[1, :])
    factorHAx.contourf(X, Y, h, cmap=colormap)
    factorHAx.contour(X, Y, h, colors="black", linewidths=0.75)
    factorHAx.set_title("FACTOR h")
    factorHAx.set_aspect(1)
    debugFig.savefig("output/00_debug.png")

## Compute sine matricies for the poisson solver
## Coefficients for x-transformation
SM = np.zeros((M-2, M-2))
for m1 in range(1, M-1):
    for m2 in range(1, M-1):
        SM[m1-1, m2-1] = np.sin(np.pi*m1*m2/(M-1))

## Coefficients for y-transformation
SN = np.zeros((N-2, N-2))
for n1 in range(1, (N-1)):
    for n2 in range(1, (N-1)):
        SN[n1-1, n2-1] = np.sin(np.pi*n1*n2/(N-1))

## Eigenvalues of Laplacian operator
EIGEN = np.zeros((M-2, N-2))
for mm in range(1, (M-1)):
    for nn in range(1, (N-1)):
        eigen = (np.sin(np.pi*mm/(2*(M-1))))**2 +(np.sin(np.pi*nn/(2*(N-1))))**2
        EIGEN[mm-1, nn-1] = (-4/(Ds**2)) * eigen

## Read and plot the initial and verification height data
## Define the inital field sizes
z0 = np.zeros((M, N)) # Initial height
z24 = np.zeros((M, N)) # Verifying Analysis
xi0 = np.zeros((M, N)) # Initial Laplacian of height
eta0 = np.zeros((M, N)) # Initial absolute vorticity

## File name tag
Case = ["Case1-", "Case2-", "Case3-", "Case4-"]

## Initial analysis
YMDH1 = ["1949010503", "1949013003", "1949013103", "1949021303"]

## Verifying analysis
YMDH2 = ["1949010603", "1949013103", "1949020103", "1949021403"]

## Initial and verification analysis on PS grid
File1 = Case[Ncase] + YMDH1[Ncase] + ".z00.txt"
File2 = Case[Ncase] + YMDH2[Ncase] + ".z00.txt"

## Read in the analyses on the PS grid
z0 = np.loadtxt(File1)
z24 = np.loadtxt(File2)

## Plot the analysis fields
initialDataFig = plt.figure()
initialDataFig.set_size_inches(1024*px, 1024*px)
initialDataAx = initialDataFig.gca()
zcontours=range(4500, 6001, 50)
initialDataAx.contourf(X, Y, z0, levels=zcontours, cmap=colormap, vmin=4500, vmax=6000)
initialDataAx.contour(X, Y, z0, levels=zcontours, colors="black", linewidths=0.75)
initialDataAx.set_aspect(1)
initialDataAx.set_title("Initial Data")
initialDataFig.savefig("output/01_initial_data.png")

finalAnalysisFig = plt.figure()
finalAnalysisFig.set_size_inches(1024*px, 1024*px)
finalAnalysisAx = finalAnalysisFig.gca()
finalAnalysisAx.contourf(X, Y, z24, levels=zcontours, cmap=colormap, vmin=4500, vmax=6000)
finalAnalysisAx.contour(X, Y, z24, levels=zcontours, colors="black", linewidths=0.75)
finalAnalysisAx.set_aspect(1)
finalAnalysisAx.set_title("Final Analysis")
finalAnalysisFig.savefig("output/02_final_analysis.png")

## Plot inner region: omit one row at bottom and two rows on the other three sides (as plotted by Charney et al.)
rx = slice(2, M-2)
ry = slice(1, N-2)
initVerChangeFig = plt.figure()
initVerChangeFig.set_size_inches(1280*px, 720*px)
initVerChangeSpec = GridSpec(2, 2, figure=initVerChangeFig)
initialAnalysisAx = initVerChangeFig.add_subplot(initVerChangeSpec[0,0])
initialAnalysisAx.contour(X[rx, ry], Y[rx, ry], z0[rx, ry], levels=zcontours, cmap=colormap, vmin=4500, vmax=6000, linewidths=0.75)
initialAnalysisAx.set_aspect(1)
initialAnalysisAx.set_title("Initial Analysis")
verifyingAnalysisAx = initVerChangeFig.add_subplot(initVerChangeSpec[0, 1])
verifyingAnalysisAx.contour(X[rx, ry], Y[rx, ry], z24[rx, ry], levels=zcontours, cmap=colormap, vmin=4500, vmax=6000, linewidths=0.75)
verifyingAnalysisAx.set_aspect(1)
verifyingAnalysisAx.set_title("Verifying Analysis")
changeAx = initVerChangeFig.add_subplot(initVerChangeSpec[1, :])
changeAx.contour(X[rx, ry], Y[rx, ry], (z24[rx, ry] - z0[rx, ry]), levels=21, cmap=colormap, linewidths=0.75)
changeAx.set_aspect(1)
changeAx.set_title("Analyzed Change")
initVerChangeFig.savefig("output/03_inner_region_initial_verifying_change.png")

## Plot the analysis on the inner region omit one row at bottom and two rows on the other three sides
initHeightInnerFig = plt.figure()
initHeightInnerFig.set_size_inches(1024*px, 1024*px)
initHeightInnerAx = initHeightInnerFig.gca()
initHeightInnerAx.contour(X[rx, ry], Y[rx, ry], z0[rx, ry], levels=zcontours, cmap="inferno")
ry = slice(1, N-4)
initHeightInnerAx.contour(X[rx, ry], Y[rx, ry], LONDEG[rx, ry], levels=np.linspace(-180, 180, 37), colors="blue", linestyles="solid")
initHeightInnerAx.contour(X[rx, ry], Y[rx, ry], LATDEG[rx, ry], levels=np.linspace(0, 90, 10), colors="blue", linestyles="solid")
initHeightInnerAx.set_aspect(1)
initHeightInnerAx.set_title("INITIAL HEIGHT (inner area)")
initHeightInnerFig.savefig("output/04_initial_height_inner_area.png")

## Define working arrays to have correct size
ddxz = np.zeros((M,N)) # x derivative of z
ddyz = np.zeros((M,N)) # y derivative of z
gradzsq = np.zeros((M,N)) # squared gradiant of z
d2dx2z = np.zeros((M,N)) # second x derivative of z
d2dy2z = np.zeros((M,N)) # second y derivative of z
xi = np.zeros((M,N)) # Laplacian of z
eta = np.zeros((M,N)) # absolute vorticity
ddxeta = np.zeros((M,N)) # x derivative of eta
ddyeta = np.zeros((M,N)) # y derivative of eta
Jacobi = np.zeros((M,N)) # Jacobian J(eta,z)
ddtz = np.zeros((M,N)) # Tendency of z
ddtxi = np.zeros((M,N)) # Tendency of xi
zdot = np.zeros((M-2,N-2)) # Interior values of ddtz
xidot = np.zeros((M-2,N-2)) # Interior values of ddtxi
ZDOT = np.zeros((M-2,N-2)) # Fourier transform of zdot
XIDOT = np.zeros((M-2,N-2)) # Fourier transform of xidot
E = np.zeros(nt+1) # Total energy
S = np.zeros(nt+1) # Total enstrophy

## Install the initial conditions for the forecast
if StreamFunction == 1:
    z0 = (grav * z0)/f0
z = z0.copy()

## Compute the Laplacian of the height/psi
# Second x-derivative of z
d2dx2z[1:M-1,:] = (z[2:M,:]+z[0:M-2,:]-2*z[1:M-1,:])/(Ds**2)
# Second y-derivative of z
d2dy2z[:,1:N-1] = (z[:,2:N]+z[:,0:N-2]-2*z[:,1:N-1])/(Ds**2)


## Laplacian of height (or vorticity)
xi0[1:M-1,1:N-1] = d2dx2z[1:M-1,1:N-1]+d2dy2z[1:M-1,1:N-1]

## Extend xi0 to boundaries (for Jacobian).
xi0[0,:] = 2*xi0[1,:]-xi0[2,:] # West
xi0[M-1,:] = 2*xi0[M-2,:]-xi0[M-3,:] # East
xi0[:,0] = 2*xi0[:,1]-xi0[:,2] # South
xi0[:,N-1] = 2*xi0[:,N-2]-xi0[:,N-3] # North

## Absolute vorticity
eta0 = np.multiply(h, xi0) + FCOR

## Compute statistics of initial fields
zmin = z0.min()
zave = z0.mean()
zmax = z0.max()
print(" z0: min, ave, max "+str(zmin)+" "+str(zave)+" "+str(zmax))

ximin = xi0.min()
xiave = xi0.mean()
ximax = xi0.max()
print(" xi0: min, ave, max "+str(ximin)+" "+str(xiave)+" "+str(ximax))

etamin = eta0.min()
etaave = eta0.mean()
etamax = eta0.max()
print(" eta0: min, ave, max "+str(etamin)+" "+str(etaave)+" "+str(etamax))

## Plot the initial height, Laplacian and Vorticity
initHgtLapVortFig = plt.figure()
initHgtLapVortFig.set_size_inches(1024*px, 1024*px)
initHgtLapVortSpec = GridSpec(2, 2, figure=initHgtLapVortFig)
initHgtAx = initHgtLapVortFig.add_subplot(initHgtLapVortSpec[0,0])
initHgtAx.contourf(X, Y, z0, levels=21, cmap=colormap)
initHgtAx.contour(X, Y, z0, levels=21, colors="black", linewidths=0.75)
initHgtAx.set_aspect(1)
initHgtAx.set_title("INITIAL HEIGHT/STREAMFUNCTION")
initLapAx = initHgtLapVortFig.add_subplot(initHgtLapVortSpec[0,1])
initLapAx.contourf(X, Y, xi0, levels=21, cmap=colormap)
initLapAx.contour(X, Y, xi0, levels=21, colors="black", linewidths=0.75)
initLapAx.set_aspect(1)
initLapAx.set_title("INITIAL LAPLACIAN OF HEIGHT/PSI")
initVortAx = initHgtLapVortFig.add_subplot(initHgtLapVortSpec[1,0])
initVortAx.contourf(X, Y, eta0, levels=21, cmap=colormap)
initVortAx.contour(X, Y, eta0, levels=21, colors="black", linewidths=0.75)
initVortAx.set_aspect(1)
initVortAx.set_title("INITIAL ABSOLUTE VORTICITY")
initMapFacAx = initHgtLapVortFig.add_subplot(initHgtLapVortSpec[1,1])
initMapFacAx.contourf(X, Y, h, cmap=colormap)
initMapFacAx.contour(X, Y, h, colors="black", linewidths=0.75)
initMapFacAx.set_aspect(1)
initMapFacAx.set_title("FACTOR h")
initHgtLapVortFig.savefig("output/05_initial_hgtstreamfunc_laplacian_vorticity.png")

#-----------------------
## MAIN LOOP
#-----------------------
startTime = dt.utcnow()

## Integrate BVE in time 
# Time-stepping is by leapfrog method (first step is forward) 
# Define xi = Del^2(z) The BVE is 
# (d/dt)xi = J(h*xi+f, z)
#
# We approximate the time derivative by
# (xi(n+1)-ni(n-1))/(2*Dt)
#
# The Jacobian term J(eta,z) is approximated by centered space differences
#
# Then the new value of xi at the new time (n+1)*Dt is:
# xi(n+1) = xi(n-1) + 2*Dt*J(n)
#
# When we have ddtxi, we have to solve a Poisson equation to get ddtz.
# Then both xi and z are stepped forward to (n+1)*Dt and the cycle is repeated.

## Start of time-stepping loop
for n in range(1, nt+1):
    ## Print out tracking information
    if PrintStep:
        print(" Step number, time(h), time(d) "+str(n)+" "+str(n*(Dt/3600))+" "+str(n*(Dt/(3600*24))))
    
    ## First time through loop
    if n == 1:
        littledt = Dt/2 # First step is forward ----- Editor's/Sam's note: cannot name variable dt or will conflict with datetime import, choosing "littledt"
        znm1 = z0.copy() # Copy initial height field
        xinm1 = xi0.copy() # Copy initial vorticity field
   
    ## Compute the derivatives, Laplacian and Jacobian.

    ## x-derivative of z
    ddxz[1:M-1,:] = (z[2:M,:]-z[0:M-2,:])/(2*Ds)

    ## y-derivative of z
    ddyz[:,1:N-1] = (z[:,2:N]-z[:,0:N-2])/(2*Ds)

    ## Square of the gradient of z
    gradzsq = np.square(ddxz)+np.square(ddyz)

    ## Second x-derivative of z
    d2dx2z[1:M-1,:] = (z[2:M,:]+z[0:M-2,:]-2*z[1:M-1,:])/(Ds**2)

    ## Second y-derivative of z
    d2dy2z[:,1:N-1] = (z[:,2:N]+z[:,0:N-2]-2*z[:,1:N-1])/(Ds**2)

    ## Laplacian of height 
    ## xi = d2dx2z + d2dy2z
    xi[1:M-1,1:N-1] = d2dx2z[1:M-1,1:N-1]+d2dy2z[1:M-1,1:N-1]

    ## Extend xi to boundaries (to compute Jacobian)
    ## (First time step only; xi from BCs after that)
    if n == 1:
        xi[0,:] = 2*xi[1,:]-xi[2,:] # West
        xi[M-1,:] = 2*xi[M-2,:]-xi[M-3,:] # East
        xi[:,0] = 2*xi[:,1]-xi[:,2] # South
        xi[:,N-1] = 2*xi[:,N-2]-xi[:,N-3] # North
    
    ## Absolute vorticity
    eta = np.multiply(h, xi) + FCOR

    ## x-derivative of eta
    ddxeta[1:M-1,:] = (eta[2:M,:]-eta[0:M-2,:])/(2*Ds)

    ## y-derivative of eta
    ddyeta[:,1:N-1] = (eta[:,2:N]-eta[:,0:N-2])/(2*Ds)

    ## Compute the Jacobian J(eta, z)
    Jacobi = np.multiply(ddxeta, ddyz) - np.multiply(ddyeta, ddxz)
    
    ## Calculate the energy and enstrophy integrals
    E[n-1] = 0.5 * np.sum(gradzsq)
    S[n-1] = 0.5 * np.sum(np.square(xi))

    ## Solve the Poisson Equation del^2(ddtz) = ddtxi with homogeneous boundary conditions:
    # z is constant on the boundaries, so ddtz vanishes.
    #
    # Note: Fourier transform of xidot denoted XIDOT
    # Fourier transform of zdot denoted ZDOT. Forward Fourier transform:
    # XIDOT = SM*xidot*SN
    # Inverse transform:
    # zdot = (4/((M-1)*(N-1)))*SM*ZDOT*SN
    
    ## Tendency values in interior
    xidot = Jacobi[1:M-1, 1:N-1]

    ## Compute the transform of the solution
    XIDOT = np.matmul(np.matmul(SM, xidot), SN)

    ## Convert transform of d(xi)/dt to transform of d(z)/dt
    ZDOT = np.divide(XIDOT, EIGEN)

    ## Computer inverse transform to get the height tendency
    zdot = (4/((M-1)*(N-1)))*np.matmul(np.matmul(SM, ZDOT), SN)
    ddtz[1:M-1, 1:N-1] = zdot # Insert inner values
    ddtxi[1:M-1, 1:N-1] = xidot # Insert inner values

    ## Compute ddtxi on the boundaries. If fluid is entering through the boundary, we set ddtxi to zero.
    # If fluid is leaving the region, we extrapolate ddtxi linearly from the interior. Charney, et al., Eqn (21)

    ## Western boundary
    sigW = np.where(np.sign(ddyz[0,:]) > 0, 1, 0)
    ddtxi[0,:] = np.multiply(sigW, (2*ddtxi[1,:]-ddtxi[2,:]))

    ## Eastern boundary
    sigE = np.where(np.sign(ddyz[M-1,:]) < 0, 1, 0)
    ddtxi[M-1,:] = np.multiply(sigE, (2*ddtxi[M-2,:]-ddtxi[M-3,:]))

    ## Sourthern boundary
    sigS = np.where(np.sign(ddxz[:,0]) < 0, 1, 0)
    ddtxi[:,0] = np.multiply(sigS, (2*ddtxi[:,1]-ddtxi[:,2]))

    ## Northern boundary
    sigN = np.where(np.sign(ddxz[:,N-1]) > 0, 1, 0)
    ddtxi[:,N-1] = np.multiply(sigN, (2*ddtxi[:,N-2]-ddtxi[:,N-3]))

    ## Step forward one time-step (leapfrog scheme)
    xinp1 = xinm1+(2*littledt)*ddtxi
    znp1 = znm1+(2*littledt)*ddtz
    
    ## Move the old values into arrays znm1 and xinm1
    ## Move the new values into arrays z and xi
    znm1 = z.copy()
    xinm1 = xi.copy()
    z = znp1.copy()
    xi = xinp1.copy()


    ## Save the fields at quarterpoints of the integration
    if n == (nt/4):
        zq1 = z.copy()
    elif n == (nt/2):
        zq2 = z.copy()
    elif n == (3*nt/4):
        zq3 = z.copy()
    elif n == nt:
        zq4 = z.copy()

    ## Plot the height field at each time step (if required)
    if PlotSteps:
        hrFig = plt.figure()
        hrFig.set_size_inches(1024*px, 1024*px)
        hrAx = hrFig.gca()
        hrAx.contourf(X, Y, z, levels=200, cmap=colormap)
        hrAx.set_aspect(1)
        hrAx.set_title("tau "+str(n*(Dt/3600))+" hours")
        hrFig.savefig("output/steps/frame"+str(n)+".png")
        plt.close(hrFig)

    ## Restore the timestep (after the first step)
    littledt = Dt

## End of the time-stepping loop
endTime = dt.utcnow()
elapsedTime = endTime - startTime
print(" Elapsed time: "+str(elapsedTime))

## Calculate the energy and enstrophy
ddxz[1:M-1,:] = (z[2:M,:]-z[0:M-2,:])/(2*Ds)
ddyz[:,1:N-1] = (z[:,2:N]-z[:,0:N-2])/(2*Ds)
gradzsq = np.square(ddxz)+np.square(ddyz)
E[nt] = 0.5 * np.sum(gradzsq)
S[nt] = 0.5 * np.sum(np.square(xi))

## Plot the energy and enstrophy integrals
if PlotEnergyEnstrophy:
    energyFig = plt.figure()
    energyFig.set_size_inches(1024*px, 1024*px)
    energyAx = energyFig.gca()
    energyAx.plot(time, E)
    energyAx.set_xlim([0, time[nt]])
    energyAx.set_ylim([0, 2*E[0]])
    energyAx.set_title("Total Energy")
    energyFig.savefig("output/06_energy.png")
    enstrophyFig = plt.figure()
    enstrophyFig.set_size_inches(1024*px, 1024*px)
    enstrophyAx = enstrophyFig.gca()
    enstrophyAx.plot(time, S)
    enstrophyAx.set_xlim([0, time[nt]])
    enstrophyAx.set_ylim([0, 2*S[0]])
    enstrophyAx.set_title("Total Enstrophy")
    enstrophyFig.savefig("output/07_enstrophy.png")

## Convert back from psi to z if necessary
if StreamFunction == 1:
    z0 = z0 * (f0/grav)
    z = z * (f0/grav)
    xi = xi * (f0/grav)
    zq1 = zq1 * (f0/grav)
    zq2 = zq2 * (f0/grav)
    zq3 = zq3 * (f0/grav)
    zq4 = zq4 * (f0/grav)

## Plot the final height field
finalHgtFig = plt.figure()
finalHgtFig.set_size_inches(1024*px, 1024*px)
finalHgtAx = finalHgtFig.gca()
finalHgtAx.contourf(X, Y, z, levels=zcontours, cmap=colormap, vmin=4500, vmax=6000)
finalHgtAx.contour(X, Y, z, levels=zcontours, colors="black", linewidths=0.75)
finalHgtAx.set_aspect(1)
finalHgtAx.set_title("FORECAST HEIGHT FIELD")
finalHgtFig.savefig("output/08_final_height.png")

## Plot the final height, laplacian and vorticity
threePanelFig = plt.figure()
threePanelFig.set_size_inches(1280*px, 720*px)
threePanelSpec = GridSpec(2, 2, figure=threePanelFig)
threePanelHgtAx = threePanelFig.add_subplot(threePanelSpec[0,0])
threePanelHgtAx.contourf(X, Y, z, levels=zcontours, cmap=colormap, vmin=4500, vmax=6000)
threePanelHgtAx.contour(X, Y, z, levels=zcontours, colors="black", linewidths=0.75)
threePanelHgtAx.set_aspect(1)
threePanelHgtAx.set_title("FORECAST GEOPOTENTIAL HEIGHT")
threePanelLapAx = threePanelFig.add_subplot(threePanelSpec[0,1])
threePanelLapAx.contourf(X, Y, xi, levels=21, cmap=colormap)
threePanelLapAx.contour(X, Y, xi, levels=21, colors="black", linewidths=0.75)
threePanelLapAx.set_aspect(1)
threePanelLapAx.set_title("FORECAST LAPLACIAN OF HEIGHT")
threePanelVortAx = threePanelFig.add_subplot(threePanelSpec[1,:])
threePanelVortAx.contourf(X, Y, eta, levels=21, cmap=colormap)
threePanelVortAx.contour(X, Y, eta, levels=21, colors="black", linewidths=0.75)
threePanelVortAx.set_aspect(1)
threePanelVortAx.set_title("FORECAST ABSOLUTE VORTICITY")
threePanelFig.savefig("output/09_forecast_geopotential_laplacian_vorticity.png")

## Plot the heights at check-points
rx = slice(0, M) #  Editor's/Sam's note: I am not sure why this is necessary, and actually this seems a 
ry = slice(0, N) #  little inefficient, but I'm sticking to the original matlab script for historical purposes
checkPointFig = plt.figure()
checkPointFig.set_size_inches(1024*px, 1024*px)
checkPointSpec = GridSpec(2, 2, figure=checkPointFig)
firstQuarterAx = checkPointFig.add_subplot(checkPointSpec[0,0])
firstQuarterAx.contour(X[rx,ry], Y[rx,ry], zq1[rx,ry], levels=zcontours, cmap=colormap, linewidths=0.75)
firstQuarterAx.set_aspect(1)
firstQuarterAx.set_title("Z (6 hours)") # Editor's/Sam's note: if the daylen or dthours parameters change, these titles will be invalid, but again, historical purposes... 
secondQuarterAx = checkPointFig.add_subplot(checkPointSpec[0,1])
secondQuarterAx.contour(X[rx,ry], Y[rx,ry], zq2[rx,ry], levels=zcontours, cmap=colormap, linewidths=0.75)
secondQuarterAx.set_aspect(1)
secondQuarterAx.set_title("Z (12 hours)")
thirdQuarterAx = checkPointFig.add_subplot(checkPointSpec[1,0])
thirdQuarterAx.contour(X[rx,ry], Y[rx,ry], zq3[rx,ry], levels=zcontours, cmap=colormap, linewidths=0.75)
thirdQuarterAx.set_aspect(1)
thirdQuarterAx.set_title("Z (18 hours)")
fourthQuarterAx = checkPointFig.add_subplot(checkPointSpec[1,1])
fourthQuarterAx.contour(X[rx,ry], Y[rx,ry], zq4[rx,ry], levels=zcontours, cmap=colormap, linewidths=0.75)
fourthQuarterAx.set_aspect(1)
fourthQuarterAx.set_title("Z (24 hours)")
checkPointFig.savefig("output/10_heights_at_checkpoints.png")

## Plot inner region: omit one row at bottom and two rows on the other three sides. These figures are in the same form as Figs 2 to 5 in Charney et al. (1950)
rx = slice(2, M-2)
ry = slice(1, N-2)
charneyFig = plt.figure()
charneyFig.set_size_inches(1024*px, 1024*px)
charneySpec = GridSpec(2, 2, figure=charneyFig)
charneyInitAx = charneyFig.add_subplot(charneySpec[0,0])
charneyInitAx.contour(X[rx,ry], Y[rx,ry], z0[rx,ry], levels=zcontours, colors="blue")
charneyInitAx.set_aspect(1)
charneyInitAx.set_title("(A) INITIAL ANALYSIS")
charneyInitAx.xaxis.set_visible(False)
charneyInitAx.yaxis.set_visible(False)
charneyVerifAx = charneyFig.add_subplot(charneySpec[0,1])
charneyVerifAx.contour(X[rx,ry], Y[rx,ry], z24[rx,ry], levels=zcontours, colors="blue")
charneyVerifAx.set_aspect(1)
charneyVerifAx.set_title("(B) VERIFYING ANALYSIS")
charneyVerifAx.xaxis.set_visible(False)
charneyVerifAx.yaxis.set_visible(False)
charneyChangeAx = charneyFig.add_subplot(charneySpec[1,0])
charneyChangeAx.contour(X[rx,ry], Y[rx,ry], (z24[rx,ry]-z0[rx,ry]), levels=11, colors="blue")
charneyChangeAx.contour(X[rx,ry], Y[rx,ry], (z[rx,ry]-z0[rx,ry]), levels=11, colors="red", linestyles="dashed")
charneyChangeAx.set_aspect(1)
charneyChangeAx.set_title("(C) INITIAL ANALYSIS")
charneyChangeAx.xaxis.set_visible(False)
charneyChangeAx.yaxis.set_visible(False)
charneyFcstAx = charneyFig.add_subplot(charneySpec[1,1])
charneyFcstAx.contour(X[rx,ry], Y[rx,ry], z[rx,ry], levels=zcontours, colors="blue")
charneyFcstAx.set_aspect(1)
charneyFcstAx.set_title("(D) FORECAST HEIGHT")
charneyFcstAx.xaxis.set_visible(False)
charneyFcstAx.yaxis.set_visible(False)
charneyFig.savefig("output/11_NEW-Case-N.png")
charneyFig.savefig("output/NEW-Case-N.eps")

## Calculate the Forecast & Persistence BIAS
sum0 = 0
sum1 = 0
sum2 = 0
sum3 = 0
rx = range(2, M-2)
ry = range(1, N-2)
for nx in rx:
    for ny in ry:
        sum0 = sum0 + 1
        sum1 = sum1 + z[nx,ny] - z24[nx,ny]
        sum2 = sum2 + z0[nx,ny] - z24[nx,ny]
        sum3 = sum3 + z[nx,ny] - z0[nx,ny]
rms1 = sum1/sum0
rms2 = sum2/sum0
rms3 = sum3/sum0
print(" Bias of 24h forecast: "+str(rms1))
print(" Bias of persistence: "+str(rms2))
print(" Mean 24h forecast change: "+str(rms3))

## Calculate the Forecast & Persistence RMS Errors
sum0 = 0
sum1 = 0
sum2 = 0
sum3 = 0
rx = range(2, M-2)
ry = range(1, N-2)
for nx in rx:
    for ny in ry:
        sum0 = sum0 + 1
        sum1 = sum1 + (z[nx,ny] - z24[nx,ny])**2
        sum2 = sum2 + (z0[nx,ny] - z24[nx,ny])**2
        sum3 = sum3 + (z[nx,ny] - z0[nx,ny])**2
rms1 = np.sqrt(sum1/sum0)
rms2 = np.sqrt(sum2/sum0)
rms3 = np.sqrt(sum3/sum0)
print(" RMS error of 24h forecast: "+str(rms1))
print(" RMS error of persistence: "+str(rms2))
print(" RMS 24h forecast change: "+str(rms3))

## Compute the S1 scores for the Forecast and for a persistence forecast
ddxz[1:M-1,:] = (z[2:M,:]-z[0:M-2,:])/(2*Ds)
ddyz[:,1:N-1] = (z[:,2:N]-z[:,0:N-2])/(2*Ds)
gradzsq = np.square(ddxz)+np.square(ddyz)
sgradz = np.sum(np.sqrt(gradzsq))

ddxz[1:M-1,:] = (z0[2:M,:]-z0[0:M-2,:])/(2*Ds)
ddyz[:,1:N-1] = (z0[:,2:N]-z0[:,0:N-2])/(2*Ds)
gradzsq = np.square(ddxz)+np.square(ddyz)
sgradz0 = np.sum(np.sqrt(gradzsq))

ddxz[1:M-1,:] = (z24[2:M,:]-z24[0:M-2,:])/(2*Ds)
ddyz[:,1:N-1] = (z24[:,2:N]-z24[:,0:N-2])/(2*Ds)
gradzsq = np.square(ddxz)+np.square(ddyz)
sgradz24 = np.sum(np.sqrt(gradzsq))

ddxz[1:M-1,:] = ((z[2:M,:]-z[0:M-2,:])/(2*Ds))-((z24[2:M,:]-z24[0:M-2,:])/(2*Ds))
ddyz[:,1:N-1] = ((z[:,2:N]-z[:,0:N-2])/(2*Ds))-((z24[:,2:N]-z24[:,0:N-2])/(2*Ds))
gradzsq = np.square(ddxz)+np.square(ddyz)
sgradzmz24 = np.sum(np.sqrt(gradzsq))
S1fcst = (sgradzmz24/max(sgradz, sgradz24)) * 100.0

ddxz[1:M-1,:] = ((z0[2:M,:]-z0[0:M-2,:])/(2*Ds))-((z24[2:M,:]-z24[0:M-2,:])/(2*Ds))
ddyz[:,1:N-1] = ((z0[:,2:N]-z0[:,0:N-2])/(2*Ds))-((z24[:,2:N]-z24[:,0:N-2])/(2*Ds))
gradzsq = np.square(ddxz)+np.square(ddyz)
sgradz0mz24 = np.sum(np.sqrt(gradzsq))
S1Pers = (sgradz0mz24/max(sgradz0, sgradz24)) * 100.0

print(" S1 score for 24h forecast: "+str(S1fcst))
print(" S1 score for persistence: "+str(S1Pers))

print("Done! Figures in output/")
# Thanks and gig'em!
