## LIST OF FUNCTIONS:
##---------------------
##    remove_dry_dates()
##    np_apply_along_axis(), np_mean(), np_std()
##    get_closest_cell_mat()
##    aggregate_fine2coarse()
##    regrid_coarse2fine()
##    fun_matchIndex()
##    fun_first_occ_of_dup()
##    moving_mean()
##    compute_relDiff_C_av()
##    get_clim_bootstrap()
##    coord_equallySpaced()
##    
##    
##    
##   
#test

import numpy as np
from numba import njit



## Function that modifies the array 'Nfine_fields', and eventually 'Ncoarse_fields', 'Npred_fields' (list of arrays ) and 'Ndown_fields',
##    by removing the dates where the original field ('Nfine_fields') has less than X% wet pixels (X=10% by default)
#---------------------------------------------------------------
def remove_dry_dates(Nfine_fields, Ncoarse_fields=None, Npred_fields=None, Ndown_fields=None, thresh_wetPix=0.1):

    ndays, nya, nxa = Nfine_fields.shape
    id_keep = np.full(ndays, True)
    for n in range(ndays):
        if np.sum(Nfine_fields[n,:,:] > 0) < (nya * nxa * thresh_wetPix):
            id_keep[n] = False
            
    Nfine_fields = Nfine_fields[id_keep,:,:]
    new_ndays = Nfine_fields.shape[0]
    out_tutple = (new_ndays, Nfine_fields)
    
    if Ncoarse_fields is not None:
        if Ncoarse_fields.shape[0] != ndays: 
            raise TypeError("Number of dates not equal in 'Nfine_fields' and 'Ncoarse_fields'")
        Ncoarse_fields = Ncoarse_fields[id_keep,:,:]
        out_tutple = out_tutple + (Ncoarse_fields, )
        
    if Npred_fields is not None: 
        if Npred_fields[0].shape[0] != ndays: 
            raise TypeError("Number of dates not equal in 'Nfine_fields' and 'Npred_fields'")
        Npred_fields = [Npred_fields[i][id_keep,:,:] for i in range(len(Npred_fields))]   # (list of arrays)
        out_tutple = out_tutple + (Npred_fields, )

    if Ndown_fields is not None:
        if Ndown_fields.shape[0] != ndays: 
            raise TypeError("Number of dates not equal in 'Nfine_fields' and 'Ndown_fields'")
        Ndown_fields = Ndown_fields[id_keep,:,:,:]     # (additional dimension: members)
        out_tutple = out_tutple + (Ndown_fields, )

    return out_tutple




## Small functions that permits the computations of mean() and std() along axis in Numba:
#---------------------------------------------------------------
@njit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result

@njit
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)


@njit
def np_std(array, axis):
    return np_apply_along_axis(np.std, axis, array)




## Function that generates the array 'closest_cell', used in various other functions. 
##    It describes the "match" between the fine and the coarse scale grid. 
##    It has the same shape as a fine-scale field (nya,nxa), and it gives for each fine-scale 
##      pixel the index of the nearest coarse-scale pixel, with the coarse-scale field flattened.
#---------------------------------------------------------------
@njit
def get_closest_cell_mat(lons_fcst, lats_fcst, lons_anal, lats_anal):

    nyf, nxf = lats_fcst.size, lons_fcst.size
    nya, nxa = lats_anal.size, lons_anal.size
    lats_fcst_vec = np.repeat(lats_fcst, repeats=nxf)
    lons_fcst_vec = np.repeat(lons_fcst, repeats=nyf).reshape(nxf,nyf).transpose().copy().reshape(nyf*nxf)
    ## (to avoid the use of np.meshgrid, which is not supported by numba)
    
    closest_cell = np.zeros((nya,nxa), dtype=np.int32)
    for i in range(nxa):
        d_lon = np.square(lons_fcst_vec - lons_anal[i])
        for j in range(nya):
            d_lat = np.square(lats_fcst_vec - lats_anal[j])
            closest_cell[j,i] = np.argmin(d_lat + d_lon)

    return closest_cell




## Function that aggregates a series of fine-scale fields into the coarse scale using spatial averaging. 
##    Masked pixels must be provided in 'mask_field' (optionnal). These pixels will be ignored in the computation of the mean.
#---------------------------------------------------------------
   
## Wrapper function (to add an optionnal mask, and reshape Nfine_fields which must be 2d for Numba:
def aggregate_fine2coarse(Nfine_fields, latLon_analFcst, mask_field=None):
    ndays, nya, nxa = Nfine_fields.shape
    lats_anal, lons_anal, lats_fcst, lons_fcst = latLon_analFcst
    if mask_field is None:
        mask_field = np.full((lats_anal.size, lons_anal.size), False)
    
    Ncoarse_fields = ma.array(aggregate_fine2coarse_njit(Nfine_fields.reshape(ndays,nya*nxa), lons_fcst, lats_fcst, lons_anal, lats_anal, mask_field), mask=False)
    Ncoarse_fields.mask[Ncoarse_fields == 999.] = True
    
    return Ncoarse_fields
    
## Wrapped function (in Numba):    
@njit
def aggregate_fine2coarse_njit(Nfine_fields_2d, lons_fcst, lats_fcst, lons_anal, lats_anal, mask_field):
    if Nfine_fields_2d.ndim != 2:
        raise TypeError("'Nfine_fields_2d' must be reshaped to 2D")

    ndays = Nfine_fields_2d.shape[0]
    nyf, nxf = lats_fcst.size, lons_fcst.size
    nya, nxa = lats_anal.size, lons_anal.size
    closest_cell_1d = get_closest_cell_mat(lons_fcst, lats_fcst, lons_anal, lats_anal).reshape(nya*nxa)
    mask_field_1d = mask_field.reshape(nya*nxa)
    
    ## Aggregation:
    Ncoarse_fields_2d = np.full((ndays, nyf*nxf), 999.)
    for k in range(nyf*nxf):
        id_fine = np.where(closest_cell_1d == k)
        if np.all(mask_field_1d[id_fine]):
            continue
        for n in range(ndays):
            Ncoarse_fields_2d[n,k] = np.mean(Nfine_fields_2d[n,:][id_fine][np.invert(mask_field_1d[id_fine])])

    return Ncoarse_fields_2d.reshape(ndays,nyf,nxf)




## Function that regrids a series of coarse-scale fields into the fine scale by just duplicating the coarse-scale pixels. 
##    Masked pixels must be provided in 'mask_field' (optionnal). 
#---------------------------------------------------------------

## Wrapper function (to add an optionnal mask, and reshape Ncoarse_fields which must be 2d for Numba:
def regrid_coarse2fine(Ncoarse_fields, latLon_analFcst, mask_coarsefield=None):
    ndays, nyf, nxf = Ncoarse_fields.shape
    lats_anal, lons_anal, lats_fcst, lons_fcst = latLon_analFcst
    if mask_coarsefield is None:
        mask_coarsefield = np.full((lats_fcst.size, lons_fcst.size), False)
    
    Nfine_fields = ma.array(regrid_coarse2fine_njit(Ncoarse_fields.reshape(ndays,nyf*nxf), lons_fcst, lats_fcst, lons_anal, lats_anal, mask_coarsefield), mask=False)
    Nfine_fields.mask[Nfine_fields == 999.] = True
    
    return Nfine_fields

## Wrapped function (in Numba):    
@njit
def regrid_coarse2fine_njit(Ncoarse_fields_2d, lons_fcst, lats_fcst, lons_anal, lats_anal, mask_coarsefield):
    
    ndays = Ncoarse_fields_2d.shape[0]
    nyf, nxf = lats_fcst.size, lons_fcst.size
    nya, nxa = lats_anal.size, lons_anal.size
    closest_cell_1d = get_closest_cell_mat(lons_fcst, lats_fcst, lons_anal, lats_anal).reshape(nya*nxa)
    mask_coarsefield_1d = mask_coarsefield.reshape(nyf*nxf)

    ## Regridding:
    Nfine_fields_2d = np.full((ndays, nya*nxa), 999.)
    for k in range(nyf*nxf):
        if mask_coarsefield_1d[k]:
            continue
        id_fine = np.where(closest_cell_1d == k)[0]
        for n in range(ndays):
            Nfine_fields_2d[n,:][id_fine] = Ncoarse_fields_2d[n,k]
    
    return Nfine_fields_2d.reshape(ndays,nya,nxa)




## Function that generates the spatial moving mean of a field, with window L (L=1 corresponds to a 3x3 pixels average)
##    Masked data (provided via the optional 'mask_field') are ignored in the moving means.
#---------------------------------------------------------------

## Wrapper for moving_mean_njit (to add a mask)
def moving_mean(field, L=1, mask_field=None):
    ny, nx = field.shape
    if mask_field is None:
        mask_field = np.full((ny,nx), False)
    
    field_mm = ma.array(moving_mean_njit(field, L, mask_field), mask=False)
    field_mm.mask[mask_field] = True
    
    return field_mm
    
## Wrapped function (in Numba):
@njit
def moving_mean_njit(field, L, mask_field):
    ny, nx = field.shape
    field_1d = field.reshape(ny*nx)
    mask_field_1d = mask_field.reshape(ny*nx)
    id_field1d = np.arange(ny*nx).reshape(ny,nx)
    field_mm = np.full((ny,nx), 999.)
    for j in range(ny):
        for i in range(nx):
            a = (j-L) if (j-L) >= 0 else 0
            b = (j+L+1) if (j+L+1) <= ny else ny
            c = (i-L) if (i-L) >= 0 else 0
            d = (i+L+1) if (i+L+1) <= nx else nx
            id_toAverage = id_field1d[a:b,c:d].ravel()
            if np.all(mask_field_1d[id_toAverage]):
                continue
            else:
                id_toAverage_withMask = id_toAverage[np.invert(mask_field_1d[id_toAverage])]
            field_mm[j,i] = np.mean(field_1d[id_toAverage_withMask])
    return field_mm




# ## Function that returns a masked vector with same size as Y, and contains
# ##   the position in X of each element of Y. If an element of Y is not in X, the
# ##   returned value is masked.
# ##   Exemple:
# ##       X = np.array([3,5,7,1,9,8,12,11])
# ##       Y = np.array([2,1,5,10])
# ##       id_match = fun_matchIndex(X, Y)
# ##       id_match
# ##       >>> masked_array(data=[--, 3, 1, --])
# #---------------------------------------------------------------
# def fun_matchIndex(X, Y):
#     index = np.argsort(X)
#     sorted_X = X[index]
#     sorted_index = np.searchsorted(sorted_X, Y)
#     Yindex = np.take(index, sorted_index, mode="clip")
#     mask = X[Yindex] != Y
#     id_match = np.ma.array(Yindex, mask=mask)
#     return id_match



## Function that returns a masked vector with same size as Y, and contains the position in X 
##   of each element of Y. If an element of Y is not in X, the returned value is masked.
##   If an element of Y is present several times in X, the first position of that element in X is returned.
##   
##   Exemple 1 (with numbers):
##       X = np.array([3,5,7,1,9,8,5,11])
##       Y = np.array([2,1,5,10])
##       id_match = fun_matchIndex(X, Y)
##       id_match
##       >>> masked_array(data=[--, 3, 1, --])
##   
##   Exemple 2 (with strings):
##       X = np.array(['bv5','bv1','bv2','bv1','bv4'])
##       Y = np.array(['bv1','bv3'])
##       id_match = fun_matchIndex(X, Y)
##       id_match
##       >>> masked_array(data=[--, 3, 1, --])
#---------------------------------------------------------------
def fun_matchIndex(X, Y, decimal_round=None):
    
    if decimal_round is not None:
        if X.dtype.type is np.str_ or Y.dtype.type is np.str_:
            raise TypeError("fun_matchIndex() cannot round strings, remove argument 'decimal_round'")
        X = np.round(X, decimal_round)
        Y = np.round(Y, decimal_round)
    
    ## X may contain duplicates, and we want the function to return the position of the first occurrence. Thus, we "erase" the non-first occurrences of the duplicates by setting their value to a value that we are sure is not in Y:
    first_occ_of_dup = fun_first_occ_of_dup(X)
    Xnodup = X.copy()
    try:
        Xnodup[first_occ_of_dup.mask == False] = -99999
    except:
        Xnodup[first_occ_of_dup.mask == False] = '-99999'
    index = np.argsort(Xnodup)
    sorted_Xnodup = Xnodup[index]
    sorted_index = np.searchsorted(sorted_Xnodup, Y, side='left')
    Yindex = np.take(index, sorted_index, mode="clip")
    mask = Xnodup[Yindex] != Y
    id_match = np.ma.array(Yindex, mask=mask)
    return id_match



## Function that returns a masked vector with same size as X, and contains the position in X
##   of the first occurence of each element, unless (i) that element is unique (no duplicates),
##   or (ii) it's the first occurrence of the duplicates. In these cases, the returned value is masked.
##   
##   Exemple:
##       X = np.array([3,3,7,1,9,8,9,11])
##       first_occ = fun_first_occ_of_duplicates(X)
##       first_occ
##       >>> masked_array(data=[--,0,--,--,--,--,4,--]) 
def fun_first_occ_of_dup(X):
    first_occ_of_dup = ma.array(np.zeros_like(X), mask=True)
    for i in range(X.size):
        if np.where(X == X[i])[0].size > 1 and i != np.where(X == X[i])[0][0]:
            first_occ_of_dup[i] = np.where(X == X[i])[0][0]
    return first_occ_of_dup

        
    
    

## Function that computes the field of the relative difference between the climatology at pixel i,j and its average over a square window of size L_oro.
##    To introduce some uncertainty, the climatology at pixel i,j is not the average of all fields in 'Nfine_fields', but of a random selection of them, 
##      with B the number of fields that is randomly selected. If the climatology at pixel i,j is equal to zero, we set relDiff_C_av = 0.
#----------------------------------------------------------------
def compute_relDiff_C_av(Nfine_fields, mask_field, B=100, L_oro=2, seed=np.random.randint(1,100000,1)[0]):
    
    np.random.seed(seed)
    
    clim_rdmDraw = np.mean(Nfine_fields[np.random.choice(np.arange(Nfine_fields.shape[0]), size=B, replace=True),:,:], axis=0)
    clim_av = moving_mean(clim_rdmDraw, L=L_oro, mask_field=mask_field)
    ## To deal with the cases where clim_rdmDraw = 0:
    id_zeros = np.where(np.logical_or(clim_rdmDraw == 0, clim_av == 0.))
    clim_av[id_zeros] = 1.         # could be any value (it's just to avoid dividing by zero), as these pixels will be set to 0.
    relDiff_C_av = ma.filled((clim_rdmDraw - clim_av)/clim_av, fill_value=0.)
    relDiff_C_av[id_zeros] = 0.
    
    return(relDiff_C_av)


    
        
## Function that generates a bootstrap sample of the climatology of a series of fine-scale fields.
##    It is useful for estimating the uncertainty of the climatology.
#---------------------------------------------------------------
def get_clim_bootstrap(Nfine_fields, B, seed=np.random.randint(1,100000,1)[0]):
    np.random.seed(seed)
    ndays, nya, nxa = Nfine_fields.shape
    
    clim_bootstrap = ma.array(np.zeros((B,nya,nxa)), mask=True)
    for b in range(B):
        id_rdmBoot = np.random.choice(np.arange(ndays), size=ndays, replace=True)
        clim_bootstrap[b,:,:] = np.mean(Nfine_fields[id_rdmBoot,:,:], axis=0)

    return clim_bootstrap




## Function that reads a vector of coordinates (lat or lon), and if the coordinates are not exactly equally spaced, it returns
##    a new vectors of coordinates that are equally spaced, and which difference with the original coordinates is minimal.
#----------------------------------------------------------------
def coord_equallySpaced(vec_coord):
    
    def coord_diff(start_and_resol, vec_coord):
        start, resol = start_and_resol[0], start_and_resol[1]
        N = vec_coord.size
        return np.mean(np.square(vec_coord - np.arange(start, start + (N-1)*resol+1e-10, resol)))
    
    if np.unique(np.diff(vec_coord)).size == 1:
        return vec_coord
    
    par_opt = minimize(coord_diff, [vec_coord[0], np.unique(np.diff(vec_coord))[0]], vec_coord).x
    vec_coord_new = np.arange(par_opt[0], par_opt[0]+(vec_coord.size-1)*par_opt[1]+1e-10, par_opt[1])

    return vec_coord_new
