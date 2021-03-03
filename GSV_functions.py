from scipy.interpolate import interp1d


## Function that computes the gridded, stratified variogram of a fine-scale field.
##    It's a numba function, so it does not accept masked array. To avoid masked pixels interfering with the  
##       computation of the GSV, we must input the array 'fine_field' where masked pixels are set to 999 (or
##       another value, indicated in the argument 'masked_value').
#----------------------------------------------------------------
@njit
def GSV(fine_field, grid_size=1, nstrata=3, transform_power=0.5, variogram_power=1, masked_value=999): 
    
    ## Function that "shifts" a field by i pixels horizontally and j pixels vertically:
    def lagging(f_padded, gs, i, j):
        ny, nx = f_padded.shape[0]-2*gs, f_padded.shape[1]-2*gs
        f_lagged = np.zeros((ny,nx))
        for x in range(nx):
            for y in range(ny):
                f_lagged[y,x] = f_padded[y+j, x+i]
        return f_lagged
    

    ## We eventually transform the field to reduce the differences between high and low values:
    f = np.power(fine_field, transform_power)
    
    ## 'fill_value' is the value in 'fine_field' which indicates masked data (Numba doesn't like masked arrays):
    msk_value = math.floor(np.power(masked_value, transform_power))      # we use math.floor to prevent non-matching due to numeric approximations
        
    ny, nx = f.shape
    gs = grid_size

    f_padded = np.zeros((ny+2*gs, nx+2*gs), dtype=np.float64)   
    f_padded[gs:-gs, gs:-gs] = f

    f = f.reshape(ny*nx)
    gsv = np.zeros((nstrata, 2*gs+1, 2*gs+1))
    for i in range(2*gs+1):
        for j in range(2*gs+1):
            if i < gs: continue               # only one side is needed
            if i == gs and j >= gs: continue  # center pixel and the pixels below are not needed

            f_lagged = lagging(f_padded, gs, i, j).reshape(ny*nx)
            
            ## We remove the pairs with at least one zero, or with at least one masked value:
            both_non0 = np.logical_and(np.logical_and(f != 0, f_lagged != 0), np.logical_and(f < msk_value, f_lagged < msk_value))
            X = f[both_non0]
            Y = f_lagged[both_non0]
            
            ## Stratification:
            intervals = np.zeros((nstrata+1), dtype=np.float64)
            intervals[-1] = 1000.
            for s in range(1,nstrata):
                intervals[s] = np.percentile(X, q=s/nstrata*100)
                
            for s in range(nstrata):
                keep = np.logical_and(X >= intervals[s], X < intervals[s+1])                
                ## There are very rare occasions where 'fine_field' has only a few different values, and thus some stratification bounds in the vector 'intervals' are equal.
                ##       These cases make the code bug, since for some 's' the vector 'keep' is empty. When we run into these cases, to avoid the code bugging we just
                ##       raise a custom exception that will be catched by the calibration function, so that the gsv is not computed (the date is discarded, as if there was not enough wet pixels)
                if np.sum(keep) == 0:                                                  
                    raise GSV_stratification_error()
                gsv[s,j,i] = np.mean(np.abs(X[keep] - Y[keep])**variogram_power)

    return(gsv)




## Custom exception:
class GSV_stratification_error(ValueError): pass    




## Function similar to the function GSV(), although it does apply stratification (hence 'GV' instead of 'GSV'), and it uses a gs fixed to 3 (hence 'GV3')
#----------------------------------------------------------------
@njit
def GV3(fine_field, transform_power=1, variogram_power=1, masked_value=999): 
    
    gs = 3

    ## Function that "shifts" a field by i pixels horizontally and j pixels vertically:
    def lagging(f_padded, gs, i, j):
        ny, nx = f_padded.shape[0]-2*gs, f_padded.shape[1]-2*gs
        f_lagged = np.zeros((ny,nx))
        for x in range(nx):
            for y in range(ny):
                f_lagged[y,x] = f_padded[y+j, x+i]
        return f_lagged

    ## We eventually transform the field to reduce the differences between high and low values:
    f = np.power(fine_field, transform_power)
        
    ## 'fill_value' is the value in 'fine_field' which indicates masked data (Numba doesn't like masked arrays):
    msk_value = math.floor(np.power(masked_value, transform_power))      # we use math.floor to prevent non-matching due to numeric approximations

    ny, nx = f.shape

    f_padded = np.zeros((ny+2*gs, nx+2*gs), dtype=np.float64)   
    f_padded[gs:-gs, gs:-gs] = f

    f = f.reshape(ny*nx)
    gv3 = np.zeros((2*gs+1, 2*gs+1))
    for i in range(2*gs+1):
        for j in range(2*gs+1):
            if i < gs: continue               # only one side is needed
            if i == gs and j >= gs: continue  # center pixel and the pixels below are not needed
            
            ## Specific to GV3: we don't compute the variogram for the pixels we don't need:
            if (j == 0 or j == 6) and i >=4: continue
            if i ==6 and j != 3: continue 
            if i == 3 and j == 2: continue
            if i == 4 and j == 3: continue

            f_lagged = lagging(f_padded, gs, i, j).reshape(ny*nx)
            
            ## We remove the pairs with at least one zero, or with at least one masked value:
            both_non0 = np.logical_and(np.logical_and(f != 0, f_lagged != 0), np.logical_and(f < msk_value, f_lagged < msk_value))
            X = f[both_non0]
            Y = f_lagged[both_non0]
            
            gv3[j,i] = np.mean(np.abs(X - Y)**variogram_power)

    return(gv3)




## Function that computes the directionnal variogram (variogram value as a funcction of angle and distance) from a gridded variogram with gs=3
#----------------------------------------------------------------
def directional_variogram3(gridded_variogram_3):
    
    gv3 = np.array(gridded_variogram_3)
    
    vario = []
    angle = []
    dist = []
    
    angle.append(90.)
    vario.append(gv3[[1,0],[3,3]])  # This is actually -90, but it's equivalent
    dist.append(np.array([2,3]))

    angle.append(np.rad2deg(np.arctan(2)))
    vario.append(gv3[[5],[4]])
    dist.append(np.array([1])*np.sqrt(5))

    angle.append(45.)
    vario.append(gv3[[4,5],[4,5]])
    dist.append(np.array([1,2])*np.sqrt(2))

    angle.append(np.rad2deg(np.arctan(0.5)))
    vario.append(gv3[[4],[5]])
    dist.append(np.array([1])*np.sqrt(5))

    angle.append(0.)
    vario.append(gv3[[3,3],[5,6]])
    dist.append(np.array([2,3]))

    angle.append(-np.rad2deg(np.arctan(0.5)))
    vario.append(gv3[[2],[5]])
    dist.append(np.array([1])*np.sqrt(5))

    angle.append(-45.)
    vario.append(gv3[[2,1],[4,5]])
    dist.append(np.array([1,2])*np.sqrt(2))

    angle.append(-np.rad2deg(np.arctan(2)))
    vario.append(gv3[[1],[4]])
    dist.append(np.array([1])*np.sqrt(5))
    
    return(angle, dist, vario)




## Function that interpolates a directionnal variogram to the distance D=sqrt(5). Returns the variogram as a function of the angle only.
#----------------------------------------------------------------
def interpolated_variogram3(direc_variogram3):
    
    angle, dist, vario = direc_variogram3
    
    vario_at_lag = np.zeros(8)
    for i in range(8):
        if vario[i].size == 1:
            vario_at_lag[i] = vario[i]
        else :
            vario_at_lag[i] = interp1d(x=dist[i], y=vario[i], kind='linear')(np.sqrt(5))

    return(vario_at_lag)




## Function that computes the indices ADI, ASI, SVI from a fine0scale fields
#----------------------------------------------------------------
def VariogramBased_indices(fine_field, transform_power=0.5, variogram_power=1):
    
    if ma.is_masked(fine_field):
        raise TypeError("The mask array 'fine_field' must be filled with 999 before used as input of 'VariogramBased_indices()'")
    
    gridded_variogram_3 = GV3(fine_field, transform_power=transform_power, variogram_power=variogram_power, masked_value=999)
    (angle, dist, vario) = directional_variogram3(gridded_variogram_3)
    vario_at_lag = interpolated_variogram3((angle, dist, vario))
    
    ## ADI (Anisotropy Direction Index):
    ADI = angle[np.argmin(vario_at_lag)]
    
    ## ASI (Anisotropy Strength Index):
    ASI = np.max(vario_at_lag) / np.min(vario_at_lag)
    
    ## SVI (Small-scale Variability Index):
    SVI = np.min(vario_at_lag)
    
    return (ADI, ASI, SVI)