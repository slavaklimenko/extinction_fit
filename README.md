# extinction_fit
The code provides fitring to dust extinction using SDSS and UV/IR photometric dta for quasars with foreground absorptions 
===============================
Create a table of objects using

sdss search - save spectral fits files to sdss folder and their names into the table
https://skyserver.sdss.org/dr18/VisualTools/explore/summary?ra=11:27:21.09&dec=24:24:17.14

2mass search - save J,H,K fluxes in the table
https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd

wise search - save W1,W2 fluxes in the table
https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd

galex search - save NUV fluxes in the table
https://galex.stsci.edu/gr6/

V band galactic extinction -  save Av value in the table
Search Av using the NED database/Galactic extinction
https://ned.ipac.caltech.edu/byname?objname=WISEA+J112721.09%2B242417.1&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1

Run the code
