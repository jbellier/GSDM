from scipy.optimize import minimize, minimize_scalar


## LIST OF FUNCTIONS:
##---------------------
##    calib_GSDM_mixTextureClim()
##    calib_GSDM_mixTextureClim_stepByStep()
##    CF_mixTextureClim()
##    CF_mixTextureClim_parOro_only()
##    get_default_par_start()
##    get_parent_models()
##    
##    



## Function that performs the calibration of the GSDM.
##    Cost function: quantifies both the texture and climatology, with weights to be defined.
##    Optimizer: constrained Nelder-Mead optimization (i.e., with parameter bounds).
##    Remark about 'fullNamePred': it is just a list of 3 strings that tells which predictors are used, but its content is for printing purpose and does not influence the calibration (the correct predictors are already put in 'Npred_fields')
#----------------------------------------------------------------
def calib_GSDM_mixTextureClim(modelE, modelS, Nfine_fields, Npred_fields, rs_pred3_spVar, latLon_analFcst, Ncoarse_fields=None, mask_field=None, par_start=None, par_optim=None, par_CF=None, par_GSV=None, par_GSDM=None, par_Penh=None, fullNamePred=['unknown','unknown','unknown'], verbose=False):
        
    ## MODIFICATIONS IN THIS NEW VERSION OF calib_GSDM_mixTextureClim:        -> NewOro5
    ##     - We have removed the computation of 'clim' and 'clim_std', as we now give 'Nfine_fields' as input of CF_mixTextureClim.
    ##     - For orographic models we start with a value that is optimal regarding the function CF_mixTextureClim_parOro_only (1d optim, with other GSDM parameters unuchanged)
    ##     - par_Penh is passed as an additionnal argument (and it's passed to CF_mixTextureClim) 

    ndays, nya, nxa = Nfine_fields.shape
    lats_anal, lons_anal, lats_fcst, lons_fcst = latLon_analFcst
    nyf, nxf = lats_fcst.size, lons_fcst.size
    
    if par_optim is None: 
        par_optim = dict(thresh_wetPix=0.1, nb_fun_eval=100, get_detailedCF=True)
    thresh_wetPix, nb_fun_eval, get_detailedCF = par_optim['thresh_wetPix'], par_optim['nb_fun_eval'], par_optim['get_detailedCF']
    
    if par_GSV is None: 
        par_GSV = dict(grid_size=1, nstrata=3, transform_power=0.5, variogram_power=1)
    grid_size, nstrata, transform_power, variogram_power = par_GSV['grid_size'], par_GSV['nstrata'], par_GSV['transform_power'], par_GSV['variogram_power']

    if par_CF is None:
        par_CF = dict(w_txtr_clim=[0.5,0.5], N_draws=1, seed=123)
    w_txtr_clim, N_draws, seed = par_CF['w_txtr_clim'], par_CF['N_draws'], par_CF['seed']

    if par_start is None:
        par_start = np.append(*get_default_par_start(modelE, modelS))

    if mask_field is None:
        mask_field = np.zeros((nya,nxa), dtype=bool)
    
    ## Initial parameters values and bounds:
    #-------------------------------------------------
    if modelE == 'E10':
        parE_bnds = [(0.,1.2)]
    elif modelE == 'E30':
        parE_bnds = [(0.,1.2), (-1.,1.), (-1.,1.)]
    elif modelE == 'E21':
        parE_bnds = [(0.,1.2), (0.,1.)]
    elif modelE == 'E32':
        parE_bnds = [(0.,1.2), (0.,1.), (0.,0.1)]
    elif modelE == 'E20o':
        parE_bnds = [(0.,1.2), (0.,4.)]
    elif modelE == 'E40o':
        parE_bnds = [(0.,1.2), (-1.,1.), (-1.,1.), (0.,5.)]
    elif modelE == 'E31o':
        parE_bnds = [(0.,1.2), (0.,1.), (0.,5.)]
    elif modelE == 'E42o':
        parE_bnds = [(0.,1.2), (0.,1.), (0.,0.1), (0.,5.)]
        
    if modelS == 'S10':
        parS_bnds = [(0.0001,10.)]
    elif modelS == 'S20':
        parS_bnds = [(0.0001,10.), (0.,1.)]
    elif modelS == 'S21':
        parS_bnds = [(0.0001,10.), (0.,2.)]
    elif modelS == 'S31':       # Same bounds for S31 and S31r
        parS_bnds = [(0.0001,10.), (0.,1.), (0.,2.)]
        
    par_bnds = parE_bnds + parS_bnds
    
    if Ncoarse_fields is None:
        ## Aggregate the fine-scale fields to the coarse scale:
        Ncoarse_fields = aggregate_fine2coarse(Nfine_fields, latLon_analFcst, mask_field=mask_field)
    else:
        assert Ncoarse_fields.shape == (ndays, nyf, nxf)
        
    ## Compute 'closest_cell' (matrix that describes the "match" between the analysis and fcst grids):
    closest_cell = get_closest_cell_mat(lons_fcst, lats_fcst, lons_anal, lats_anal)
    
    ## We first compute the GSV of all the analysis fields where there is enough wet pixels
    ##     i.e., % wet pixels > thresh_wetPix, in both the analysis field and the coarse-scale field (with useGEFS=True it is possible that the analysis field has enough wet pixels, but the GEFS, i.e., coarse-scale, field hasn't)
    Ngsv_anal = np.zeros((ndays, nstrata, 2*grid_size+1, 2*grid_size+1), dtype=np.float64)
    for n in range(ndays):
        if np.sum(Nfine_fields[n,:,:] > 0) >= (nya*nxa*thresh_wetPix) and np.sum(Ncoarse_fields[n,:,:] > 0) >= (nyf*nxf*thresh_wetPix):
            try:
                Ngsv_anal[n,:,:,:] = GSV(ma.filled(Nfine_fields[n,:,:], fill_value=999), grid_size, nstrata, transform_power, variogram_power, masked_value=999)
            except GSV_stratification_error:
                ## There are very rare occasions where the field has only a few different values, which causes the stratification, and thus the computation of the GSV, impossible. 
                ##       These cases raise the custom execption 'GSV_stratification_error', which we catch so that the gsv is not computed (the date is discarded, as if there was not enough wet pixels)
                pass
    if np.sum(Ngsv_anal) == 0:
        warnings.warn('There is not enough wet pixels for estimating the GSV (2 potential reasons: too many masked pixels, or not enough dates with precipitation)')

        
    ## If the model is orographic, we do a 1-parameter optimization to find a good starting value for par_oro (fixing the non-orographic parameters):
    if modelE[-1] == 'o':
        par_start[int(modelE[1])-1] = minimize_scalar(CF_mixTextureClim_parOro_only, args=(par_start, modelE, modelS, Ncoarse_fields, Npred_fields, rs_pred3_spVar, closest_cell, Ngsv_anal, Nfine_fields, mask_field, par_CF, par_GSV, par_GSDM, par_Penh), 
                                                      method='bounded', bounds=(0.,5.)).x
        
    ## Minimization:
    #-------------------------------------------------
    f_start = CF_mixTextureClim(par_start, modelE, modelS, Ncoarse_fields, Npred_fields, rs_pred3_spVar, closest_cell, Ngsv_anal, Nfine_fields, mask_field, par_CF, par_GSV, par_GSDM, par_Penh)
    if verbose is True:
        print('f_start: '+str(f_start))
        print('par_start: '+str(par_start))
    
    result_optim = constrNM(func=CF_mixTextureClim,
                            x0=par_start,
                            LB=[i[0] for i in par_bnds], UB=[i[1] for i in par_bnds],
                            args=(modelE, modelS, Ncoarse_fields, Npred_fields, rs_pred3_spVar, closest_cell, Ngsv_anal, Nfine_fields, mask_field, par_CF, par_GSV, par_GSDM, par_Penh),
                            maxfun=nb_fun_eval, disp=False, full_output=True)
    par_opt = result_optim['xopt']
    f_opt = result_optim['fopt']
    Nf_eval = result_optim['funcalls']
    convergence_message = str(['Converged to a solution',
                               'Maximum number of function evaluations made',
                               'Maximum number of iterations reached'][result_optim['warnflag']])
    if verbose is True:
        print('After '+str(Nf_eval)+' iterations:')
        print('  f_opt: '+str(f_opt))
        print('  par_opt: '+str(par_opt))
    
    result_optim = {'modelE':modelE,
                    'modelS':modelS,
                    'fullNamePred':fullNamePred,
                    'par_opt':par_opt, 
                    'f_opt':f_opt,
                    'par_start':par_start,
                    'f_start':f_start,
                    'Nf_eval':Nf_eval,
                    'convergence_message':convergence_message}
    
    if get_detailedCF is True:
        par_CF_test = par_CF.copy()   # To avoid modifying the par_CF passed as agrument
        
        par_CF_test['w_txtr_clim'] = [1, 0]
        f_start_fullTxtr = CF_mixTextureClim(par_start, modelE, modelS, Ncoarse_fields, Npred_fields, rs_pred3_spVar, closest_cell, Ngsv_anal, Nfine_fields, mask_field, par_CF_test, par_GSV, par_GSDM, par_Penh)
        f_opt_fullTxtr = CF_mixTextureClim(par_opt, modelE, modelS, Ncoarse_fields, Npred_fields, rs_pred3_spVar, closest_cell, Ngsv_anal, Nfine_fields, mask_field, par_CF_test, par_GSV, par_GSDM, par_Penh)

        par_CF_test['w_txtr_clim'] = [0, 1]
        f_start_fullClim = CF_mixTextureClim(par_start, modelE, modelS, Ncoarse_fields, Npred_fields, rs_pred3_spVar, closest_cell, Ngsv_anal, Nfine_fields, mask_field, par_CF_test, par_GSV, par_GSDM, par_Penh)
        f_opt_fullClim = CF_mixTextureClim(par_opt, modelE, modelS, Ncoarse_fields, Npred_fields, rs_pred3_spVar, closest_cell, Ngsv_anal, Nfine_fields, mask_field, par_CF_test, par_GSV, par_GSDM, par_Penh)

        complementary_results = {'f_start_fullTxtr':f_start_fullTxtr,
                                 'f_opt_fullTxtr':f_opt_fullTxtr,
                                 'f_start_fullClim':f_start_fullClim,
                                 'f_opt_fullClim':f_opt_fullClim}
        result_optim.update(complementary_results)
    
    return(result_optim)




## Function that wraps calib_GSDM_mixTextureClim() so that it performs a step-by-step calibration, starting with the least complex parents models and then 
##      increasing the complexity progressively, keeping the optimal parameters from the previous step.
##    The total number of function evalutions are distributed among the steps depending of the number of "new parameters" that are introduced in each step.  
##    The idea of providing 'fullNamePred' and 'previous_results_optims' is that for each step the algorithm can look into past optimization results and 
##      see if the optimization for the same combination modelE/modelS/predictors has already been done. In that case, we don't need to do it again.
#----------------------------------------------------------------
def calib_GSDM_mixTextureClim_stepByStep(modelE, modelS, Nfine_fields, Npred_fields, rs_pred3_spVar, latLon_analFcst, Ncoarse_fields=None, mask_field=None, par_start=None, par_optim=None, par_CF=None, par_GSV=None, par_GSDM=None, par_Penh=None, fullNamePred=['unknown','unknown','unknown'], previous_results_optims=None):
        
    ## MODIFICATIONS IN THIS NEW VERSION OF calib_GSDM_mixTextureClim_stepByStep:        -> NewOro5
    ##     - par_Penh is passed as an additionnal argument (and it's passed to calib_GSDM_mixTextureClim)
    
    results_optims = []    # We will return all results_optim (for all steps) into this list 

    par_optim_inner = par_optim.copy()
    par_CF_inner = par_CF.copy()
    
    verbose = par_optim['verbose']

    modelE_parents, modelS_parents, Nmod = get_parent_models(modelE, modelS, include_current=True)
    Nnew_parE = np.append(int(modelE_parents[0][1]), np.diff([int(modelE_parents[p][1]) for p in range(Nmod)]))
    Nnew_parS = np.append(int(modelS_parents[0][1]), np.diff([int(modelS_parents[p][1]) for p in range(Nmod)]))
    ## Loop over the steps (i.e., over the levels of parent models):
    for s in range(Nmod):
        modE, modS = modelE_parents[s], modelS_parents[s]
        
        ## For each step we adjust 'fullNamePred', since for the child models not all predictors are used
        fullNamePred_s = fullNamePred.copy()
        if int(modE[2]) == 0:
            fullNamePred_s[0] = 'None'
            fullNamePred_s[1] = 'None'
        elif int(modE[2]) == 1:
            fullNamePred_s[0] = 'None'
        if int(modS[2]) == 0:
            fullNamePred_s[2] = 'None'
            
        ## If the optimization has already been done (for the same 'modE', 'modS' and 'fullNamePred_s') and it is stored in 'previous_results_optims', we don't need to do it again:
        do_optimization = True
        if previous_results_optims is not None:
            previous_results_optims_flat = [item for sublist in previous_results_optims for item in sublist]  # Flattens 'previous_results_optim' into a list and not a list of lists 
            for result_optim in previous_results_optims_flat:
                if modE == result_optim['modelE'] and modS == result_optim['modelS'] and fullNamePred_s == result_optim['fullNamePred']:
                    par_opt = result_optim['par_opt']
                    parE_opt = par_opt[:int(modE[1])]
                    parS_opt = par_opt[-int(modS[1]):]
                    results_optims.append(result_optim)
                    do_optimization = False
                    break

        ## Otherwise we do the optimization:
        if do_optimization == True:
            
            ## The number of function evaluation in this inner loop is set proportionnal to the number of new parameters, keeping the total number equal to par_optim['nb_fun_eval']:
            par_optim_inner['nb_fun_eval'] = np.rint((Nnew_parE + Nnew_parS)/np.sum(Nnew_parE + Nnew_parS) * par_optim['nb_fun_eval'])[s]

            ## If modE is not "orographic" we put all the weight in the texture
            if modE[-1] != 'o':
                par_CF_inner['w_txtr_clim'] = [1,0]
            else:
                par_CF_inner['w_txtr_clim'] = par_CF['w_txtr_clim']

            ## To derive par_start, we use the optimal parameters from the previous loop and for the new parameters we take the default values:
            if s == 0:
                parE_start, parS_start = get_default_par_start(modE,modS)
            else:
                if Nnew_parE[s] > 0:
                    parE_start = np.append(parE_opt, get_default_par_start(modE,modS)[0][-Nnew_parE[s]:])
                else:
                    parE_start = parE_opt
                if Nnew_parS[s] > 0:
                    parS_start = np.append(parS_opt, get_default_par_start(modE,modS)[1][-Nnew_parS[s]:])
                else:
                    parS_start = parS_opt
            par_start = np.append(parE_start, parS_start)

            ## Optimization:
            result_optim = calib_GSDM_mixTextureClim(modE, modS, Nfine_fields, Npred_fields, rs_pred3_spVar, latLon_analFcst, Ncoarse_fields, mask_field, par_start, par_optim_inner, par_CF_inner, par_GSV, par_GSDM, par_Penh, fullNamePred_s)
            par_opt = result_optim['par_opt']
            parE_opt = par_opt[:int(modE[1])]
            parS_opt = par_opt[-int(modS[1]):]
            results_optims.append(result_optim)

        ## if verbose is true we print results for each step (no matter if the optimization has been done or the results taken from previous optimizations)
        if verbose == True:
            if do_optimization == True:
                print(modE+' '+modS+' '+str(fullNamePred_s)+', '+str(int(par_optim_inner['nb_fun_eval']))+' iter max')
            else:
                print(modE+' '+modS+' '+str(fullNamePred_s)+', optimization already done')
            print('   CF mix: '+str(round(result_optim['f_start'],3))+' -> '+str(round(result_optim['f_opt'],3)))
            if par_optim_inner['get_detailedCF'] == True:
                print('   Txtr: '+str(round(result_optim['f_start_fullTxtr'],3))+' -> '+str(round(result_optim['f_opt_fullTxtr'],3)))
                print('   Clim: '+str(round(result_optim['f_start_fullClim'],3))+' -> '+str(round(result_optim['f_opt_fullClim'],3)))
            with np.printoptions(precision=5, suppress=True):
                print('   pars: '+str(par_start)+' -> '+str(par_opt))
            print('')
                
    return results_optims


        
        
## Cost function used in the calibration function
##     It penalizes both the error in texture and the error in climatology, with standardization.
#----------------------------------------------------------------
def CF_mixTextureClim(par, modelE, modelS, Ncoarse_fields, Npred_fields, rs_pred3_spVar, closest_cell, Ngsv_anal, Nfine_fields, mask_field=None, par_CF=None, par_GSV=None, par_GSDM=None, par_Penh=None):

    ## MODIFICATIONS IN THIS NEW VERSION OF CF_mixTextureClim:        -> NewOro5
    ##     - It does not take 'clim' and 'clim_std' as argument, but 'Nfine_fields'
    ##     - We compute a different clim for each date and downscaling member, and from this clim we compute 'relDiff_C_av', that is given as input of GSDM_njit.
    ##     - Uses a new function 'compute_relDiff_C_av', which computes 'relDiff_C_av' from 'Nfine_fields', with a fixed random seed.
    ##     - par_Penh is passed as an additionnal argument (contains L_oro and B)
    
    ndays = Ncoarse_fields.shape[0]
    nya, nxa = closest_cell.shape
    
    if par_CF is None:
        par_CF = dict(w_txtr_clim=[0.5,0.5], N_draws=1, seed=123)
    w_txtr_clim, N_draws, seed = par_CF['w_txtr_clim'], par_CF['N_draws'], par_CF['seed']
        
    if par_GSV is None: 
        par_GSV = dict(grid_size=1, nstrata=3, transform_power=0.5, variogram_power=1)
    grid_size, nstrata, transform_power, variogram_power = par_GSV['grid_size'], par_GSV['nstrata'], par_GSV['transform_power'], par_GSV['variogram_power']
    
    if par_GSDM is None:
        par_GSDM = dict(N_iter=10)
    N_iter = par_GSDM['N_iter']
    
    if par_Penh is None:
        par_Penh = dict(L_oro=2, B=100)
    L_oro, B = par_Penh['L_oro'], par_Penh['B']

    if mask_field is None:
        mask_field = np.zeros_like(closest_cell, dtype=bool)
        
    ## Check for correct inputs:
    if int(modelE[1]) + int(modelS[1]) != len(par):
        raise TypeError("Input 'par' is not compatible with 'modelE' and 'modelS'")
    if mask_field.shape != closest_cell.shape:
        raise TypeError("'mask_field' has an incorrect shape")
    if grid_size != int((Ngsv_anal.shape[2] - 1)/2):
        raise ValueError("The 'grid_size' value specified in 'par_GSV' is not compatible with 'Ngsv_anal'")
    if nstrata != Ngsv_anal.shape[1]:
        raise ValueError("The 'nstrata' value specified in 'par_GSV' is not compatible with 'Ngsv_anal'")
    if not (Ncoarse_fields.shape[0] == Npred_fields[0].shape[0] == Ngsv_anal.shape[0]):
        raise TypeError("Number of dates not equal in 'Ncoarse_fields', 'Npred_fields' and 'Ngsv_anal'")
    if len(w_txtr_clim) != 2 or any(w < 0 for w in w_txtr_clim):
        raise ValueError("Weights incorrects in 'w_txtr_clim'")
        
    
    ## Separate 'parE' and 'parS' from 'par':
    parE = par[:int(modelE[1])]
    parS = par[-int(modelS[1]):]
    if len(parE) == 0: parE = [0.]   # will not be used, but we cannot pass en empty list with numba
        
        
    ## Climatology field computed with all dates in Nfine_fields (will be used only for the error):
    clim = np.mean(Nfine_fields, axis=0)

    SAE_txtr = 0   # sum absolute error (SAE) regarding texture
    R = np.zeros((N_draws,ndays,nya,nxa), dtype=np.float64)
    for n in range(ndays):
        
        ASP, ADP, SVP = Npred_fields[0][n,:,:], Npred_fields[1][n,:,:], Npred_fields[2][n,:,:]
        coarse_field = ma.filled(Ncoarse_fields[n,:,:], 0.)
        gsv_anal = Ngsv_anal[n,:,:,:]
        
        if np.sum(coarse_field) == 0.:   # Skip date if the entire field is dry (we don't need to do the downscaling, but the field in R for date n will still contain the zeros, which is important for computing the climatology of the model)
            continue
            
        sae_txtr_dr = np.zeros((N_draws), dtype=np.float64)
        for n_dr in range(N_draws):
            
            ## Climatology field computed with B dates randomly chosen in Nfine_fields:
            if modelE[-1] == 'o':
                relDiff_C_av = compute_relDiff_C_av(Nfine_fields, mask_field, B=B, L_oro=L_oro, seed=seed+(n+1)*(n_dr+1))
            else:
                relDiff_C_av = np.zeros_like(mask_field)

            # Downscaling
            R_dr = GSDM_njit(modelE, modelS, parE, parS, coarse_field, closest_cell, ASP, ADP, SVP, relDiff_C_av=relDiff_C_av, mask_field=mask_field, rs_pred3_spVar=rs_pred3_spVar, N_iter=N_iter, seed=seed+(n+1)*(n_dr+1))
            R[n_dr,n,:,:] = R_dr
            if np.any(gsv_anal != 0):    # If 'gsv_anal' has only zeros, it means that the original field has less than 10% ('thresh_wetPix') pixels wet => We don't compute the GSV
                R_dr[mask_field] = 999    # 'GSV' is in Numba (doesn't accept masked arrays), so we use the value 999
                gsv_R_dr = GSV(R_dr, grid_size, nstrata, transform_power, variogram_power, masked_value=999)
                sae_txtr_dr[n_dr] = np.mean(np.abs((gsv_R_dr[gsv_anal>0] - gsv_anal[gsv_anal>0])))
        
        SAE_txtr = SAE_txtr + np.mean(sae_txtr_dr)
    
    MAE_txtr = SAE_txtr / np.sum(np.any(Ngsv_anal > 0, axis=(1,2,3)))   # Sum Absolute Error (SAE) to Mean Absolute Error (MAE) over the dates
    clim_model = np.mean(R, axis=(0,1))
    RMSE_clim = np.sqrt(np.mean(np.square(clim[mask_field==False] - clim_model[mask_field==False])))
    
    ## Standardization of the errors:
    MAE_txtr = MAE_txtr / np.mean(Ngsv_anal[Ngsv_anal>0])
    RMSE_clim = RMSE_clim / np.mean(clim[mask_field==False])

    error = (w_txtr_clim[0]*MAE_txtr + w_txtr_clim[1]*RMSE_clim)/(w_txtr_clim[0] + w_txtr_clim[1])
    
    return(error)



## Cost function used for estimating a first guess of the orographic parameter.
##     It is basically a wrapper of 'CF_mixTextureClim', but with only par_oro as the parameter that can be optimized.
#----------------------------------------------------------------
def CF_mixTextureClim_parOro_only(par_oro, all_pars, modelE, *args):
    all_pars_modified = all_pars.copy()
    all_pars_modified[np.int(modelE[1])-1] = par_oro
    return (CF_mixTextureClim(all_pars_modified, modelE, *args))





## Function that gives the default paramater values to start the optimization.
#----------------------------------------------------------------
def get_default_par_start(modelE, modelS):
    
    if modelE == 'E10':
        parE_start = np.array([0.2])
    elif modelE == 'E30':
        parE_start = np.array([0.2, 0., 0.])
    elif modelE == 'E21':
        parE_start = np.array([0.2, 0.])
    elif modelE == 'E32':
        parE_start = np.array([0.2, 0., 0.])
    elif modelE == 'E20o':
        parE_start = np.array([0.2, 0.])
    elif modelE == 'E31o':
        parE_start = np.array([0.2, 0., 0.])
    elif modelE == 'E40o':
        parE_start = np.array([0.2, 0., 0., 0.])
    elif modelE == 'E42o':
        parE_start = np.array([0.2, 0., 0., 0.])
        
    if modelS == 'S10':
        parS_start = np.array([0.1])
    elif modelS == 'S20':
        parS_start = np.array([0.1, 0.])
    elif modelS == 'S21':
        parS_start = np.array([0.1, 0.])
    elif modelS == 'S31':       # Same par_start and bounds for S31 and S31r
        parS_start = np.array([0.1, 0., 0.])

    return(parE_start, parS_start)




## Function that returns the list of parents models (from farthest to nearest).
##    When modelE and modelS have a different number of parents, we repeat modelE/modelS.
#----------------------------------------------------------------
def get_parent_models(modelE, modelS, include_current=False):
    
    if modelE == 'E10':
        modelE_parents = []
    elif modelE == 'E30':
        modelE_parents = ['E10']
    elif modelE == 'E21':
        modelE_parents = ['E10']
    elif modelE == 'E32':
        modelE_parents = ['E10','E21']
    elif modelE == 'E20o':
        modelE_parents = ['E10']
    elif modelE == 'E40o':
        modelE_parents = ['E10','E30']
    elif modelE == 'E31o':
        modelE_parents = ['E10','E21']
    elif modelE == 'E42o':
        modelE_parents = ['E10','E21','E32']
    NparentsE = len(modelE_parents)

    if modelS == 'S10':
        modelS_parents = []
    elif modelS == 'S20':
        modelS_parents = ['S10']
    elif modelS == 'S21':
        modelS_parents = ['S10']
    elif modelS == 'S31':
        modelS_parents = ['S10','S20']
    NparentsS = len(modelS_parents)
    
    if NparentsE > NparentsS:
        for i in range(NparentsE - NparentsS):
            modelS_parents = modelS_parents + [modelS]
    if NparentsE < NparentsS:
        for i in range(NparentsS - NparentsE):
            modelE_parents = modelE_parents + [modelE]
    Nmod = len(modelE_parents)      # = len(modelS_parents)

    if include_current == False:
        return modelE_parents, modelS_parents, Nmod
    else:
        return modelE_parents + [modelE], modelS_parents + [modelS], Nmod+1
