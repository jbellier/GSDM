## LIST OF FUNCTIONS:
##---------------------
##    GSDM_njit()
##    GSDM_lgD_njit()   -> models and parameters can be specified for each fine-scale pixel


@njit
def GSDM_njit(modelE, modelS, parE, parS, coarse_field, closest_cell, ASP, ADP, SVP, relDiff_C_av, mask_field, rs_pred3_spVar='no_pred3', N_iter=10, seed=np.random.randint(1,100000,1)[0]):

    ## MODIFICATIONS IN THIS NEW VERSION OF GSDM_njit:        -> NewOro5
    ##     - The relative difference (relDiff_C_av) of the clim at i,j and the averagged clim over a L=2 window is now given in argument, in place of clim and clim_std.
    ##     - The orographic adjustment is done at the end of each Gibbs sampling iteration, rather than being included in E.
    
    np.random.seed(seed)
        
    nya, nxa = closest_cell.shape
    nyf, nxf = coarse_field.shape
    
    ## Check for correct inputs
    if modelE != 'E10' and modelE != 'E30' and modelE != 'E21' and modelE != 'E32' and modelE != 'E20o' and modelE != 'E31o' and modelE != 'E40o' and modelE != 'E42o':
        raise TypeError("'ModelE' in incorrect")
    if modelS != 'S10' and modelS != 'S20' and modelS != 'S21' and modelS != 'S31':
        raise TypeError("'ModelS' in incorrect")
    if (modelE == 'E10' and len(parE) != 1) or ((modelE == 'E21' or modelE == 'E20o') and len(parE) != 2) or ((modelE == 'E30' or modelE == 'E32' or modelE == 'E31o') and len(parE) != 3) or ((modelE == 'E40o' or modelE == 'E42o') and len(parE) != 4):
        raise TypeError('The number of parameters in "parE" is not compatible with "modelE"')
    if (modelS == 'S10' and len(parS) != 1) or ((modelS == 'S20' or modelS == 'S21') and len(parS) != 2) or (modelS == 'S31' and len(parS) != 3):
        raise TypeError('The number of parameters in "parS" is not compatible with "modelS"')
    if ASP.shape != (nya,nxa) or ADP.shape != (nya,nxa) or SVP.shape != (nya,nxa):
        raise TypeError('The predictor arrays have a wrong shape')
    if rs_pred3_spVar != 'positive' and rs_pred3_spVar != 'negative' and rs_pred3_spVar != 'no_pred3':
        raise TypeError("'rs_pred3_spVar' should be either 'positive', 'negative' or 'no_pred3'")
    if (modelS == 'S21' or modelS == 'S31') and rs_pred3_spVar == 'no_pred3':
        raise TypeError("For models 'S21' and 'S31' the argument 'rs_pred3_spVar' must be either 'positive' or 'negative'")
        
    coarse_field_1d = coarse_field.reshape(nyf*nxf)
    closest_cell_1d = closest_cell.reshape(nya*nxa)
    mask_field_1d = mask_field.reshape(nya*nxa)

    R = np.zeros((nya,nxa), dtype=np.float64)               # Creates empty downscaled field (will be updated 'N_iter' times)
        
    ## Initialization (with arrays flatten to permit advanced indexing in Numba): ---------------------------
    R_1d = R.reshape(nya*nxa)
    for jj in range(nyf):
        for ii in range(nxf):
            k = jj*nxf + ii
            if coarse_field_1d[k] == 0.:
                continue
            pix = np.nonzero(closest_cell_1d == k)
            R_1d[pix] = coarse_field_1d[k]
    R = R_1d.reshape(nya,nxa)

    ## Gibbs sampling: ---------------------------
    for n_it in range(N_iter):
        for i in range(nxa):
            for j in range(nya):
                
                if coarse_field_1d[closest_cell[j,i]] == 0.:   # No need to do Gibbs sampling if coarse-scale pixel is dry
                    continue
                
                if (i == 0 and j == 0):                            # Lower left corner
                    Aver = R[1,0]
                    Ahor = R[0,1]
                    Adown = R[1,1]
                    Aup = R[1,1]
                    Aav = 1/3 * (R[1,0] + R[0,1] + R[1,1])

                elif (i == (nxa-1) and j == 0):                    # Lower right corner
                    Aver = R[1,nxa-1]
                    Ahor = R[0,nxa-2]
                    Adown = R[1,nxa-2]
                    Aup = R[1,nxa-2]
                    Aav = 1/3 * (R[1,nxa-1] + R[0,nxa-2] + R[1,nxa-2])

                elif (i == 0 and j == (nya-1)):                    # Upper left corner
                    Aver = R[(nya-2),0]
                    Ahor = R[(nya-1),1]
                    Adown = R[(nya-2),1]
                    Aup = R[(nya-2),1]
                    Aav = 1/3 * (R[(nya-2),0] + R[(nya-1),1] + R[(nya-2),1])

                elif (i == (nxa-1) and j == (nya-1)):              # Upper right corner
                    Aver = R[(nya-2),(nxa-1)]
                    Ahor = R[(nya-1),(nxa-2)]
                    Adown = R[(nya-2),(nxa-2)]
                    Aup = R[(nya-2),(nxa-2)]
                    Aav = 1/3 * (R[(nya-2),(nxa-1)] + R[(nya-1),(nxa-2)] + R[(nya-2),(nxa-2)])

                elif (i > 0 and i < (nxa-1) and j == 0):           # Lower row
                    Aver = R[1,i]
                    Ahor = 1/2 * (R[0,i-1] + R[0,i+1])
                    Adown = R[1,i-1]
                    Aup = R[1,i+1]
                    Aav = 1/5 * (R[0,i-1] + R[0,i+1] + R[1,i-1] + R[1,i] + R[1,i+1])

                elif (i > 0 and i < (nxa-1) and j == (nya-1)):     # Upper row
                    Aver = R[(nya-2),i]
                    Ahor = 1/2 * (R[(nya-1),i-1] + R[(nya-1),i+1])
                    Adown = R[(nya-2),i+1]
                    Aup = R[(nya-2),i-1]
                    Aav = 1/5 * (R[(nya-1),i-1] + R[(nya-1),i+1] + R[(nya-2),i-1] + R[(nya-2),i] + R[(nya-2),i+1])

                elif (i == 0 and j > 0 and j < (nya-1)):           # Left column
                    Aver = 1/2 * (R[j-1,0] + R[j+1,0])
                    Ahor = R[j,1]
                    Adown = R[j-1,1]
                    Aup = R[j+1,1]
                    Aav = 1/5 * (R[j-1,0] + R[j+1,0] + R[j-1,1] + R[j,1] + R[j+1,1])

                elif (i == (nxa-1) and j > 0 and j < (nya-1)):     # Right column
                    Aver = 1/2 * (R[j-1,(nxa-1)] + R[j+1,(nxa-1)])
                    Ahor = R[j,(nxa-2)]
                    Adown = R[j+1,(nxa-2)]
                    Aup = R[j-1,(nxa-2)]
                    Aav = 1/5 * (R[j-1,(nxa-1)] + R[j+1,(nxa-1)] + R[j-1,(nxa-2)] + R[j,(nxa-2)] + R[j+1,(nxa-2)])

                else:                                              # REGULAR CASE
                    Aver = 1/2 * (R[j-1,i] + R[j+1,i])
                    Ahor = 1/2 * (R[j,i-1] + R[j,i+1])
                    Aup = 1/2 * (R[j-1,i-1] + R[j+1,i+1])
                    Adown = 1/2 * (R[j+1,i-1] + R[j-1,i+1])
                    Aav = 1/4 * (Aver + Ahor + Aup + Adown)
               
            
                ## Expectation:
                #---------------
                if modelE == 'E10':
                    E = Aav + parE[0]/2*(Aver+Ahor-Adown-Aup)
                elif modelE == 'E30':
                    E = Aav + parE[0]/2*(Aver+Ahor-Adown-Aup) + parE[1]*(Aup-Adown) + parE[2]*(Aver-Ahor)
                elif modelE == 'E21':
                    E = Aav + parE[0]/2*(Aver+Ahor- Adown-Aup) + parE[1]*np.cos(np.deg2rad(2*(ADP[j,i] - 45)))*(Aup-Adown) + parE[1]*np.cos(np.deg2rad(2*(ADP[j,i] - 90)))*(Aver-Ahor)               
                elif modelE == 'E32':
                    E = Aav + parE[0]/2*(Aver+Ahor- Adown-Aup) + (parE[1] + parE[2]*ASP[j,i])*np.cos(np.deg2rad(2*(ADP[j,i] - 45)))*(Aup-Adown) + (parE[1] + parE[2]*ASP[j,i])*np.cos(np.deg2rad(2*(ADP[j,i] - 90)))*(Aver-Ahor)               
                elif modelE == 'E20o':
                    E = Aav + parE[0]/2*(Aver+Ahor-Adown-Aup)
                elif modelE == 'E40o':
                    E = Aav + parE[0]/2*(Aver+Ahor-Adown-Aup) + parE[1]*(Aup-Adown) + parE[2]*(Aver-Ahor)
                elif modelE == 'E31o':
                    E = Aav + parE[0]/2*(Aver+Ahor- Adown-Aup) + parE[1]*np.cos(np.deg2rad(2*(ADP[j,i] - 45)))*(Aup-Adown) + parE[1]*np.cos(np.deg2rad(2*(ADP[j,i] - 90)))*(Aver-Ahor)               
                elif modelE == 'E42o':
                    E = Aav + parE[0]/2*(Aver+Ahor- Adown-Aup) + (parE[1] + parE[2]*ASP[j,i])*np.cos(np.deg2rad(2*(ADP[j,i] - 45)))*(Aup-Adown) + (parE[1] + parE[2]*ASP[j,i])*np.cos(np.deg2rad(2*(ADP[j,i] - 90)))*(Aver-Ahor)               
                    
                ## Correction to make sure we don't have a negative expectation:
                if E < 0.1: E = 0.1
                    
                    
                ## Standard deviation:
                #---------------
                if modelS == 'S10':
                    SD = parS[0]
                elif modelS == 'S20':
                    SD = parS[0] + parS[1]*E
                elif modelS == 'S21':
                    if rs_pred3_spVar == 'negative':    # Negative relationship between the predictor SVP and the spatial variability
                        SD = parS[0] * np.exp(-parS[1]/parS[0] * SVP[j,i])
                    else:                               # Positive relationship
                        SD = parS[0] + parS[1]*SVP[j,i]
                elif modelS == 'S31':
                    if rs_pred3_spVar == 'negative':    # Negative relationship between the predictor SVP and the spatial variability
                        SD = parS[0] * np.exp(-parS[2]/parS[0] * SVP[j,i]) + parS[1]*E
                    else:                               # Positive relationship
                        SD = parS[0] + parS[2]*SVP[j,i] + parS[1]*E    


                ## We put a safety to avoid potential overflow when SD is squared, which can happen if parameters or/or predictor values are extreme (we cannot handle overflow with try/except in numba):
                if SD > 1e9:
                    SD = 1e9
                
                ## Parameters of the lognormal distribution (as a function of E and SD):
                mu = -1/2 * np.log(SD**2/E**4 + 1/E**2)
                sigma = np.sqrt(2*(np.log(E) - mu))

                ## Random sampling:
                R[j,i] = lognormal(mu, sigma)
                
                if np.isnan(R[j,i]):   # Unrealistically high values. To avoid 'Nan', we put random values but high enough so that it still gets penalized
                    R[j,i] = randint(1000,2000,1)[0]
                    
                    
        ## Orographic adjustment:
        if modelE == 'E20o' or modelE == 'E40o' or modelE == 'E31o' or modelE == 'E42o':
            for i in range(nxa):
                for j in range(nya):
                    if modelE == 'E20o':
                        P_enh = parE[1]*relDiff_C_av[j,i]
                    elif modelE == 'E40o':
                        P_enh = parE[3]*relDiff_C_av[j,i]
                    elif modelE == 'E31o':
                        P_enh = parE[2]*relDiff_C_av[j,i]
                    elif modelE == 'E42o':
                        P_enh = parE[3]*relDiff_C_av[j,i]
                    R[j,i] = R[j,i]*(1 + P_enh)

                
        ## Multiplicative adjustment to adjust the volume over the coarse cell (with arrays flatten):
        R_1d = R.reshape(nya*nxa)
        for jj in range(nyf):
            for ii in range(nxf):
                k = jj*nxf + ii
                if coarse_field_1d[k] == 0.:
                    continue
                pix = np.nonzero(np.logical_and(closest_cell_1d == k, np.invert(mask_field_1d)))  # permits to ignore masked pixels in the coarse-scale average
                coeff = coarse_field_1d[k] / np.mean(R_1d[pix])
                R_1d[pix] = coeff * R_1d[pix]
        R = R_1d.reshape(nya,nxa)
        
    return(R)




## This function has the same core as GSDM_njit, but instead of accepting a single set of models and parameters that is used for all pixels,  
##    it takes a list of models and parameters, as well as a field ('field_parDomain') that specifies for each pixel which model/parameter to use.
#----------------------------------------------------------------
@njit
def GSDM_lgD_njit(DmodelE, DmodelS, DparE, DparS, Drs_pred3_spVar, field_parDomain, coarse_field, closest_cell, ASP, ADP, SVP, relDiff_C_av, mask_field, maskOro_field, N_iter=10, seed=np.random.randint(1,100000,1)[0]):

    ## MODIFICATIONS IN THIS NEW VERSION OF GSDM_lgD_njit:        -> NewOro5
    ##     - maskOro_field is passed as an additional argument. However it's not used in this current version of GSDM_lgD_njit, so it coould be removed
    ##     - The relative difference (relDiff_C_av) of the clim at i,j and the averagged clim over a L=2 window is now given in argument, in place of clim and clim_std.
    ##     - The orographic adjustment is done at the end of each Gibbs sampling iteration, rather than being included in E
    ##     - We return P_enh_arr in addition to R
    
    np.random.seed(seed)
        
    nya, nxa = closest_cell.shape
    nyf, nxf = coarse_field.shape
    
    ## Check for correct inputs
    if ASP.shape != (nya,nxa) or ADP.shape != (nya,nxa) or SVP.shape != (nya,nxa):
        raise TypeError('The predictor arrays have a wrong shape')
    for d in range(DmodelE.size):
        modelE = DmodelE[d]
        modelS = DmodelS[d]
        parE = DparE[d,:]
        parS = DparS[d,:]
        rs_pred3_spVar = Drs_pred3_spVar[d] 
        if modelE != 'E10' and modelE != 'E30' and modelE != 'E21' and modelE != 'E32' and modelE != 'E20o' and modelE != 'E31o' and modelE != 'E40o' and modelE != 'E42o':
            raise TypeError("'ModelE' in incorrect")
        if modelS != 'S10' and modelS != 'S20' and modelS != 'S21' and modelS != 'S31':
            raise TypeError("'ModelS' in incorrect")
#         if (modelE == 'E10' and len(parE) != 1) or ((modelE == 'E21' or modelE == 'E20o') and len(parE) != 2) or ((modelE == 'E30' or modelE == 'E32' or modelE == 'E31o') and len(parE) != 3) or ((modelE == 'E40o' or modelE == 'E42o') and len(parE) != 4):
#             raise TypeError('The number of parameters in "parE" is not compatible with "modelE"')
#         if (modelS == 'S10' and len(parS) != 1) or ((modelS == 'S20' or modelS == 'S21') and len(parS) != 2) or (modelS == 'S31' and len(parS) != 3):
#             raise TypeError('The number of parameters in "parS" is not compatible with "modelS"')
        if rs_pred3_spVar != 'positive' and rs_pred3_spVar != 'negative' and rs_pred3_spVar != 'no_pred3':
            raise TypeError("'rs_pred3_spVar' should be either 'positive', 'negative' or 'no_pred3'")
        if modelS == 'S31' and rs_pred3_spVar == 'no_pred3':
            raise TypeError("For model 'S31' the argument 'rs_pred3_spVar' must be 'positive' or 'negative'")
        
    coarse_field_1d = coarse_field.reshape(nyf*nxf)
    closest_cell_1d = closest_cell.reshape(nya*nxa)
    mask_field_1d = mask_field.reshape(nya*nxa)
    field_parDomain_1d = field_parDomain.reshape(nyf*nxf)    
    
    
    R = np.zeros((nya,nxa), dtype=np.float64)               # Creates empty downscaled field (will be updated 'N_iter' times)
    P_enh_arr = np.zeros((nya,nxa), dtype=np.float64)                   
        
    ## Initialization (with arrays flatten to permit advanced indexing in Numba): ---------------------------
    R_1d = R.reshape(nya*nxa)
    for jj in range(nyf):
        for ii in range(nxf):
            k = jj*nxf + ii
            if coarse_field_1d[k] == 0.:
                continue
            pix = np.nonzero(closest_cell_1d == k)
            R_1d[pix] = coarse_field_1d[k]
    R = R_1d.reshape(nya,nxa)

    ## Gibbs sampling: ---------------------------
    for n_it in range(N_iter):
        for i in range(nxa):
            for j in range(nya):
                
                if coarse_field_1d[closest_cell[j,i]] == 0.:   # No need to do Gibbs sampling if coarse-scale pixel is dry
                    continue
                    
                id_dom = field_parDomain_1d[closest_cell[j,i]]
                modelE = DmodelE[id_dom]
                modelS = DmodelS[id_dom]
                parE = DparE[id_dom,:]
                parS = DparS[id_dom,:]
                rs_pred3_spVar = Drs_pred3_spVar[id_dom]
                
                if (i == 0 and j == 0):                            # Lower left corner
                    Aver = R[1,0]
                    Ahor = R[0,1]
                    Adown = R[1,1]
                    Aup = R[1,1]
                    Aav = 1/3 * (R[1,0] + R[0,1] + R[1,1])
                    
                elif (i == (nxa-1) and j == 0):                    # Lower right corner
                    Aver = R[1,nxa-1]
                    Ahor = R[0,nxa-2]
                    Adown = R[1,nxa-2]
                    Aup = R[1,nxa-2]
                    Aav = 1/3 * (R[1,nxa-1] + R[0,nxa-2] + R[1,nxa-2])

                elif (i == 0 and j == (nya-1)):                    # Upper left corner
                    Aver = R[(nya-2),0]
                    Ahor = R[(nya-1),1]
                    Adown = R[(nya-2),1]
                    Aup = R[(nya-2),1]
                    Aav = 1/3 * (R[(nya-2),0] + R[(nya-1),1] + R[(nya-2),1])

                elif (i == (nxa-1) and j == (nya-1)):              # Upper right corner
                    Aver = R[(nya-2),(nxa-1)]
                    Ahor = R[(nya-1),(nxa-2)]
                    Adown = R[(nya-2),(nxa-2)]
                    Aup = R[(nya-2),(nxa-2)]
                    Aav = 1/3 * (R[(nya-2),(nxa-1)] + R[(nya-1),(nxa-2)] + R[(nya-2),(nxa-2)])

                elif (i > 0 and i < (nxa-1) and j == 0):           # Lower row
                    Aver = R[1,i]
                    Ahor = 1/2 * (R[0,i-1] + R[0,i+1])
                    Adown = R[1,i-1]
                    Aup = R[1,i+1]
                    Aav = 1/5 * (R[0,i-1] + R[0,i+1] + R[1,i-1] + R[1,i] + R[1,i+1])

                elif (i > 0 and i < (nxa-1) and j == (nya-1)):     # Upper row
                    Aver = R[(nya-2),i]
                    Ahor = 1/2 * (R[(nya-1),i-1] + R[(nya-1),i+1])
                    Adown = R[(nya-2),i+1]
                    Aup = R[(nya-2),i-1]
                    Aav = 1/5 * (R[(nya-1),i-1] + R[(nya-1),i+1] + R[(nya-2),i-1] + R[(nya-2),i] + R[(nya-2),i+1])

                elif (i == 0 and j > 0 and j < (nya-1)):           # Left column
                    Aver = 1/2 * (R[j-1,0] + R[j+1,0])
                    Ahor = R[j,1]
                    Adown = R[j-1,1]
                    Aup = R[j+1,1]
                    Aav = 1/5 * (R[j-1,0] + R[j+1,0] + R[j-1,1] + R[j,1] + R[j+1,1])

                elif (i == (nxa-1) and j > 0 and j < (nya-1)):     # Right column
                    Aver = 1/2 * (R[j-1,(nxa-1)] + R[j+1,(nxa-1)])
                    Ahor = R[j,(nxa-2)]
                    Adown = R[j+1,(nxa-2)]
                    Aup = R[j-1,(nxa-2)]
                    Aav = 1/5 * (R[j-1,(nxa-1)] + R[j+1,(nxa-1)] + R[j-1,(nxa-2)] + R[j,(nxa-2)] + R[j+1,(nxa-2)])

                else:                                              # REGULAR CASE
                    Aver = 1/2 * (R[j-1,i] + R[j+1,i])
                    Ahor = 1/2 * (R[j,i-1] + R[j,i+1])
                    Aup = 1/2 * (R[j-1,i-1] + R[j+1,i+1])
                    Adown = 1/2 * (R[j+1,i-1] + R[j-1,i+1])
                    Aav = 1/4 * (Aver + Ahor + Aup + Adown)
                
                
                ## Expectation:
                #---------------
                if modelE == 'E10':
                    E = Aav + parE[0]/2*(Aver+Ahor-Adown-Aup)
                elif modelE == 'E30':
                    E = Aav + parE[0]/2*(Aver+Ahor-Adown-Aup) + parE[1]*(Aup-Adown) + parE[2]*(Aver-Ahor)
                elif modelE == 'E21':
                    E = Aav + parE[0]/2*(Aver+Ahor- Adown-Aup) + parE[1]*np.cos(np.deg2rad(2*(ADP[j,i] - 45)))*(Aup-Adown) + parE[1]*np.cos(np.deg2rad(2*(ADP[j,i] - 90)))*(Aver-Ahor)               
                elif modelE == 'E32':
                    E = Aav + parE[0]/2*(Aver+Ahor- Adown-Aup) + (parE[1] + parE[2]*ASP[j,i])*np.cos(np.deg2rad(2*(ADP[j,i] - 45)))*(Aup-Adown) + (parE[1] + parE[2]*ASP[j,i])*np.cos(np.deg2rad(2*(ADP[j,i] - 90)))*(Aver-Ahor)               
                elif modelE == 'E20o':
                    E = Aav + parE[0]/2*(Aver+Ahor-Adown-Aup)
                elif modelE == 'E40o':
                    E = Aav + parE[0]/2*(Aver+Ahor-Adown-Aup) + parE[1]*(Aup-Adown) + parE[2]*(Aver-Ahor)
                elif modelE == 'E31o':
                    E = Aav + parE[0]/2*(Aver+Ahor- Adown-Aup) + parE[1]*np.cos(np.deg2rad(2*(ADP[j,i] - 45)))*(Aup-Adown) + parE[1]*np.cos(np.deg2rad(2*(ADP[j,i] - 90)))*(Aver-Ahor)               
                elif modelE == 'E42o':
                    E = Aav + parE[0]/2*(Aver+Ahor- Adown-Aup) + (parE[1] + parE[2]*ASP[j,i])*np.cos(np.deg2rad(2*(ADP[j,i] - 45)))*(Aup-Adown) + (parE[1] + parE[2]*ASP[j,i])*np.cos(np.deg2rad(2*(ADP[j,i] - 90)))*(Aver-Ahor)               
                
                ## Correction to make sure we don't have a negative expectation:
                if E < 0.1: E = 0.1
                    
                    
                ## Standard deviation:
                #---------------
                if modelS == 'S10':
                    SD = parS[0]
                elif modelS == 'S20':
                    SD = parS[0] + parS[1]*E
                elif modelS == 'S31':
                    if rs_pred3_spVar == 'negative':    # Negative relationship between the predictor SVP and the spatial variability
                        SD = parS[0] * np.exp(-parS[2]/parS[0] * SVP[j,i]) + parS[1]*E
                    else:                               # Positive relationship
                        SD = parS[0] + parS[2]*SVP[j,i] + parS[1]*E    
                
                ## We put a safety to avoid potential overflow when SD is squared, which can happen if parameters or/or predictor values are extreme (we cannot handle overflow with try/except in numba):
                if SD > 1e9:
                    SD = 1e9

                ## Parameters of the lognormal distribution (as a function of E and SD):
                mu = -1/2 * np.log(SD**2/E**4 + 1/E**2)
                sigma = np.sqrt(2*(np.log(E) - mu))

                ## Random sampling:
                R[j,i] = lognormal(mu, sigma)
                
                if np.isnan(R[j,i]):   # Unrealistically high values. To avoid 'Nan', we put random values but high enough so that it still gets penalized
                    R[j,i] = randint(1000,2000,1)[0]
                    
                    
        ## Orographic adjustment:
        ##         (COULD BE SPED UP BY ONLY LOOPING OVER PIXELS WHERE modelE IS OROGRAPHIC)
        for i in range(nxa):
            for j in range(nya):
                id_dom = field_parDomain_1d[closest_cell[j,i]]
                modelE = DmodelE[id_dom]
                parE = DparE[id_dom,:]
                
                if modelE == 'E20o':
                    P_enh_arr[j,i] = parE[1]*relDiff_C_av[j,i]
                elif modelE == 'E40o':
                    P_enh_arr[j,i] = parE[3]*relDiff_C_av[j,i]
                elif modelE == 'E31o':
                    P_enh_arr[j,i] = parE[2]*relDiff_C_av[j,i]
                elif modelE == 'E42o':
                    P_enh_arr[j,i] = parE[3]*relDiff_C_av[j,i]
                else:
                    P_enh_arr[j,i] = 0.
                R[j,i] = R[j,i]*(1 + P_enh_arr[j,i])
                
                    
        ## Multiplicative adjustment to adjust the volume over the coarse cell (with arrays flatten):
        R_1d = R.reshape(nya*nxa)
        for jj in range(nyf):
            for ii in range(nxf):
                k = jj*nxf + ii
                if coarse_field_1d[k] == 0.:
                    continue
                pix = np.nonzero(np.logical_and(closest_cell_1d == k, np.invert(mask_field_1d)))  # permits to ignore masked pixels in the coarse-scale average
                coeff = coarse_field_1d[k] / np.mean(R_1d[pix])
                R_1d[pix] = coeff * R_1d[pix]
        R = R_1d.reshape(nya,nxa)
        
    return(R, P_enh_arr)