;+
;Name:
;      Compareflux
;Purpose:
;      Compare flux of dark subtracted frame and final reduced frame
;Explanation:
;      The dark subtracted and reduced frames are read in. A
;      quadrilateral aperature, determined by the edges file, is
;      applied to the dark subtracted frame and resulting flux is
;      summed over. The reduced frame is summed over wavelength, then
;      the y axis. Lastly, the flux is not summed over the entire x
;      range, but rather within certain limits, set by the keywork
;      pixFromMax. After flux has been integrated over wavelength and
;      the y axis, the maximum flux element in the resulting 1D along
;      the x axis is found and the lower and upper bounds of
;      integration are the index of the maximum flux element minus and
;      plus, respectively, the value pixFromMax, or the lower or
;      higher edge of the array, if applicable. The integration over
;      the x axis between these two limits is taken to be the reduced
;      flux. The total flux in the dark frame and the ratio of the
;      flux from the dark frame and the reduced frame is printed. 
;
;Calling sequence
;      dark_sub_name - full location of desired dark subtracted frame
;
;      reduced_name - full location of desired fully reduced frame
;
;      edge_name - full location of file with x and y positions of
;                  edges desired for quadrilateral mask for dark
;                  subtracted frame. Must be ordered in clockwise or
;                  counter-clockwise direction.
;
;                  Example file:
;                  x_1   y_1
;                  x_2   y_2
;                  x_3   y_3
;                  x_4   y_4
;
;       pixFromMax - the distance from the maximum flux element in the
;                    1D array along the x axis (flux has been
;                    integrated over wavelength and the y axis) to set
;                    the lower and upper bounds of integration along
;                    the x axis. If the edge of the array is surpassed
;                    then that edge is used as the respective upper or
;                    lower bound of integration.
;
;Output
;       The total flux in the square aperture of the reduced frame is
;       printed and the ratio of that formentioned flux and that found
;       in the reduced frame is also printed.


pro compareFlux, dark_sub_name, reduced_name, edges_name

pixFromMax=6

;; Read in dark subtracted frame
darkSub = readfits(dark_sub_name)
reduced = readfits(reduced_name)
openr,1,edges_name
edges = fltarr(2,4)
readf,1,edges
free_lun,1
X1dex = [0,2,4,6]
X2dex = [2,4,6,0]
Y1dex = [1,3,5,7]
Y2dex = [3,5,7,1]
darkTot = 0.0

apeture = darkSub * 0.0
for i=0, 2047 do begin
   for j=0, 2047 do begin
      X1 = edges[X1dex] - i
      X2 = edges[X2dex] - i
      Y1 = edges[Y1dex] - j
      Y2 = edges[Y2dex] - j
      dp = X1*X2 + Y1*Y2
      cp = X1*Y2 - Y1*X2
      theta = Atan(cp,dp)
      if (Abs(Total(theta)) GT !pi) then darkTot += darkSub[j,i]
   endfor
endfor

sum_lambda = total(reduced,1)
sum_y = total(sum_lambda,1)
junk = max(sum_y,maxDex)
lowDex = maxDex - pixFromMax
highDex = maxDex + pixFromMax
if (lowDex LT 0) then lowDex = 0
if (highDex GT 18) then highDex = 18
redTot = total(sum_y[lowDex:highDex])

print, 'Flux counts from dark subtracted frame:'
print, darkTot
print, 'Flux counts from reduced frame:'
print, redTot
print, redTot/darkTot
end
