## LIST OF FUNCTIONS:
##---------------------
##    parameterSpatialization()
##    find_neighbors()
##    convolution()
##    compute_corner_values()
##    get_pixels_around_edge()
##    weight_edge()
##    labelling()
##    recursive_flood_fill


from scipy.ndimage import label
from scipy.spatial import Delaunay
from skimage.draw import line as skimage_line
import networkx
import warnings



## Function that takes a coarse-scale precipitation field ('coarse_field_lgD'), and returns a field ('field_parDomain') that contains different 
##      integers for different areas over which the same GSDM models, parameters and predictors will be used. These area ares constructed such 
##      that their borders "cut" at least precipitation as possible, to avoid visual discontinuities in the final downscaled fields that would 
##      occur if different GSDM models, parameters and predictors are used over the same rainfall feature.
##  Tunable parameters:
##       T: Threshold (in mm) that identifies a rainfall feature. 
##             T too low (e.g., 0.5 mm) -> Generates very few (but large) features -> Lower risk of "cutting" a precipitation area, but higher risk of using non-optimal GSDM parameters
##             T too high (e.g., 10 mm) -> Generates numerous (but small) features -> Higher risk of "cutting" a precipitation area, but the GSDM parameters will be more optimal
##       radius_in_pix: Radius of the convolution (in number of coarse-scale pixels) that is applied to the coarse-scale before the features are identified.
##             radius_in_pix too low (e.g., 0 pix) -> Very nearby features remain as individual features
##             radius_in_pix too high (e.g., 4 pix) -> Tends to merge quite distinct rainfall features into a single feature (higher risk of using non-optimal GSDM parameters)
##       buffer: Distance (in number of coarse-scale pixels) to which we extent the search area around the source and target points, in the Dijkstra's algorithm
##             buffer too low (e.g., 3 pix) -> Higher risk of "cutting" a precipitation area (especially if it is a large area), but code faster
##             buffer too high (e.g., 15 pix) -> Lower risk of "cutting" a precipitation area, but code slower

def parameterSpatialization(coarse_field_lgD, lons_fcst_lgD, lats_fcst_lgD, Dbounds_area, par_paramSpat=None, mask_coarsefield_lgD=None, seed=123):

    np.random.seed(seed)  # We let the possibility of fixing the random seed, since there is a random process (see later the random positionning of the artificial centroids)
    
    field = coarse_field_lgD
    nyfd, nxfd = field.shape
    id_field_1d = np.arange(nyfd*nxfd).reshape(nyfd,nxfd)
    D = len(Dbounds_area)
    
    if mask_coarsefield_lgD is None:
        mask_coarsefield_lgD = np.full((nyfd, nxfd), False)

    if par_paramSpat is None:
        par_paramSpat = dict(radius_in_pix=2, T=2, buffer=5)
        
            
    ## Original calibration domain (square):
    ##     Remark: If a pixel is over a calibration domain, then it is associated to that domain. If a pixel isn't over any domain, it
    ##        is associated to the domain whose center is the nearest.
    #------------------------------------------------------------------------------------------------------
    field_calibDomain = np.zeros((nyfd, nxfd), dtype=np.int32)
    for y in range(nyfd):
        for x in range(nxfd):
            try:
                field_calibDomain[y,x] = np.where([(lats_fcst_lgD[y] >= Dbounds_area[dom][0] and
                                                    lats_fcst_lgD[y] <= Dbounds_area[dom][1] and
                                                    lons_fcst_lgD[x] >= Dbounds_area[dom][2] and 
                                                    lons_fcst_lgD[x] <= Dbounds_area[dom][3]) for dom in range(D)])[0][0]
            except IndexError:
                field_calibDomain[y,x] = np.argmin([np.sqrt( (lons_fcst_lgD[x] - (Dbounds_area[dom][2] + (Dbounds_area[dom][3] - Dbounds_area[dom][2])/2))**2 
                                                            + (lats_fcst_lgD[y] - (Dbounds_area[dom][0] + (Dbounds_area[dom][1] - Dbounds_area[dom][0])/2))**2 ) for dom in range(D)])

                

    ## If 'keep_calibDom' is True, we just return the calibration domains, and we don't run the full process:
    #------------------------------------------------------------------------------------------------------
    if 'keep_calibDom' in par_paramSpat:
        if par_paramSpat['keep_calibDom'] is True:
            return field_calibDomain
            
            

    radius_in_pix = par_paramSpat['radius_in_pix']     # Radius of the convolution, in pixel unit, in the identification of the rainfall features
    T = par_paramSpat['T']                             # Threshold for identifying the rainfall features
    buffer = par_paramSpat['buffer']                   # Distance (in number of corner points) to which we extent the search area around the source and target points, in the Dijkstra's algorithm
    
    
    lons_fcst_lgD_mat, lats_fcst_lgD_mat = np.meshgrid(lons_fcst_lgD, lats_fcst_lgD)

    res_fcst_lat = np.unique(np.diff(lats_fcst_lgD))[0]
    res_fcst_lon = np.unique(np.diff(lons_fcst_lgD))[0]
#     if np.unique(np.diff(lats_fcst_lgD)).size > 1 or np.unique(np.diff(lons_fcst_lgD)).size > 1:
#         warnings.warn("The coordinates in the coarse grid are not spaced regularly.")


    ## Identify the rainfall features and label them:
    #------------------------------------------------------------------------------------------------------
    ## For the convolution we don't apply the mask (but fill masked values with zeros), such that rainfall features may 
    ##    eventually overflow to masked areas and thus reach "islands" (i.e., non-masked area surrounded by masked pixels)
    field_convolved = convolution(ma.filled(field, 0.), radius_in_pix, mask_field=None)
    field_featuresLabeled, nfeatures = label(np.logical_and(field_convolved >= T, mask_coarsefield_lgD == False))
    unique_featLabels = np.unique(field_featuresLabeled)[np.unique(field_featuresLabeled) != 0]

    ## There may still exist non-masked "islands" identified as different features, although there is no precip. We unmark 
    ##    them as features, and update 'unique_featLabels'.
    for k in range(nfeatures):
        if np.sum(field[np.where(field_featuresLabeled == unique_featLabels[k])]) == 0:
            field_featuresLabeled[np.where(field_featuresLabeled == unique_featLabels[k])] = 0
            nfeatures = nfeatures - 1
    unique_featLabels = np.unique(field_featuresLabeled)[np.unique(field_featuresLabeled) != 0]


    ## Interpolation to the corners:
    #------------------------------------------------------------------------------------------------------
    corner_vals = compute_corner_values(field, mask_coarsefield_lgD)    
    lats_fcst_lgD_corner = np.arange(lats_fcst_lgD[0]-res_fcst_lat/2, lats_fcst_lgD[-1]+res_fcst_lat/2+0.0001, res_fcst_lat)
    lons_fcst_lgD_corner = np.arange(lons_fcst_lgD[0]-res_fcst_lon/2, lons_fcst_lgD[-1]+res_fcst_lon/2+0.0001, res_fcst_lon)


    ## Compute the centroids of the features:
    #------------------------------------------------------------------------------------------------------
    LonLat_centroid = np.zeros((nfeatures,2))
    for k in range(nfeatures):
        id_featk = np.where(field_featuresLabeled.ravel() == unique_featLabels[k])[0]    
        LonLat_centroid[k,0] = np.sum(lons_fcst_lgD_mat.ravel()[id_featk] * field.ravel()[id_featk]) / np.sum(field.ravel()[id_featk])
        LonLat_centroid[k,1] = np.sum(lats_fcst_lgD_mat.ravel()[id_featk] * field.ravel()[id_featk]) / np.sum(field.ravel()[id_featk])

        

    ## Delaunay triangulations of the centroids:
    ##      If there are less than 3 centroids  (i.e., less than 3 rainfall feature), the triangululation cannot be made.
    ##      Thus, we create "artificial" centroids and randomly place them in the field, until there are enough for the 
    ##      triangulation to be made. Note that we constrain these artificial centroids to be at least 4 degrees far from 
    ##      each other. 
    ##      The "except" catches both 'ValueError' (happens if LonLat_centroid is empty, i.e., no centroids) and 'QhullError', which is
    ##      a custom exceptions from  Delauney that is raised if there is less than 3 centroids, but also sometimes when there are enough 
    ##      centroids but they are positionned in such a way that the function Delaunay() cannot perform the triangulation (e.g., flat simplex).
    #------------------------------------------------------------------------------------------------------
    triang = None
    while triang is None:
        try:
            triang = Delaunay(LonLat_centroid)
        except: # (ValueError, QhullError):    
            LonLat_centroid_artif = np.array([np.random.choice(lons_fcst_lgD_corner), np.random.choice(lats_fcst_lgD_corner)])
            dist_with_others = [np.sqrt((LonLat_centroid_artif[0] - LonLat_centroid[k,0])**2 + (LonLat_centroid_artif[1] - LonLat_centroid[k,1])**2) for k in range(nfeatures)]
            while np.any(np.array(dist_with_others) < 4.):
                LonLat_centroid_artif = np.array([np.random.choice(lons_fcst_lgD_corner), np.random.choice(lats_fcst_lgD_corner)])
                dist_with_others = [np.sqrt((LonLat_centroid_artif[0] - LonLat_centroid[k,0])**2 + (LonLat_centroid_artif[1] - LonLat_centroid[k,1])**2) for k in range(nfeatures)]
            LonLat_centroid = np.vstack((LonLat_centroid, LonLat_centroid_artif))
            nfeatures = LonLat_centroid.shape[0]     # nfeatures needs to be updated with the "artificial" centroids
    
    nsimplex = triang.simplices.shape[0]
    

    ## Loop over the simplices of the Delauney triangulation:
    ##     Step A: We identify, for each edge of the simplices, the "low point", i.e., the point under the edge where precipitation is minimal.
    ##     Step B: For each simplex, we link the three low points together, using the path of least "resistance", i.e., the path that divides
    ##             the precipitation area in two sub-areas with the least precipitation at the boundary. 
    ##     Step C: For each simplex, we remove one of the three paths (the one with), as only two are necassary.
    #------------------------------------------------------------------------------------------------------

    xlow_all = ma.array(np.zeros((nfeatures, nfeatures), dtype=np.int32), mask=True)
    ylow_all = ma.array(np.zeros((nfeatures, nfeatures), dtype=np.int32), mask=True)
    nsplx_sharing_edge = np.zeros((nfeatures, nfeatures), dtype=np.int32)

    id_corner_1d = np.arange(corner_vals.size).reshape(corner_vals.shape)
    id_corner_1d_boundary = np.unique(np.append(id_corner_1d[[0,-1],:].ravel(), id_corner_1d[:,[0,-1]].ravel()))

    optimal_paths = []
    list_graphs = []        # saved only for graphic purposes. Contains

    for splx in range(nsimplex):

        pts = triang.simplices[splx,:]
        edges = [[pts[0],pts[1]], [pts[0],pts[2]], [pts[1],pts[2]]]
        [edges[edg].sort() for edg in range(3)]   # we sort the points in every edge so that in the matrices xlow_all and ylow_all only the upper diagonal is filled

        ## Step A: Loop over the 3 edges to find the 3 "low points":
        for e in edges:

            ## The 2 centroid points forming the edge:
            A = e[0]
            B = e[1]

            ## If this edge has already been treated (i.e., the low point has been found) through another simplex, we pass:
            if xlow_all.mask[A,B] == False:
                nsplx_sharing_edge[A,B] = nsplx_sharing_edge[A,B] + 1

            else:
                ## We approximate the two centroids to their nearest "corner" pixel:
                x1 = np.argmin(np.abs(LonLat_centroid[A,0] - lons_fcst_lgD_corner))
                y1 = np.argmin(np.abs(LonLat_centroid[A,1] - lats_fcst_lgD_corner))
                x2 = np.argmin(np.abs(LonLat_centroid[B,0] - lons_fcst_lgD_corner))
                y2 = np.argmin(np.abs(LonLat_centroid[B,1] - lats_fcst_lgD_corner))

                ## We use 'skimage_line()' to return the corner pixels that are traversed by a line from (x1,y1) to (x2,y2) using Bresenham's line algorithm. 
                pts_traversed = skimage_line(y1, x1, y2, x2)
                ## We are then interested by the minimum among these points:
                id_low = np.argmin(corner_vals[pts_traversed])

                ## And finally we save the position of the low point. 
                xlow_all[A,B] = pts_traversed[1][id_low]
                ylow_all[A,B] = pts_traversed[0][id_low]

                nsplx_sharing_edge[A,B] = nsplx_sharing_edge[A,B] + 1

        ## Step B: Loop over the 3 inter-low-point paths:
        xlows = xlow_all[[edges[e][0] for e in range(3)], [edges[e][1] for e in range(3)]]
        ylows = ylow_all[[edges[e][0] for e in range(3)], [edges[e][1] for e in range(3)]]
        pths = [[0,1],[1,2],[0,2]]
        optimal_paths_splx = []
        list_graphs_splx = []     # saved only for graphic purpose
        path_costs_splx= []
        for p in pths:
            ## we need to find the shortest path between the source and the target
            source = (ylows[p[0]], xlows[p[0]])
            target = (ylows[p[1]], xlows[p[1]])

            ## Creating the graph using the module networkx:
            G = networkx.Graph()
            for j in range(min(source[0], target[0])-buffer, max(source[0], target[0])+1+buffer, 1):           # 'buffer' determines how far (in number of corner points) we extent the search area around the source and target points
                for i in range(min(source[1], target[1])-buffer, max(source[1], target[1])+1+buffer, 1):
                    if i >= 0 and i < id_corner_1d.shape[1]-1 and j >= 0 and j < id_corner_1d.shape[0]-1:
                        if i < max(source[1], target[1])+buffer:
                            G.add_edge(id_corner_1d[j,i], id_corner_1d[j,i+1], 
                                       weight=weight_edge(field, id_corner_1d[j,i], id_corner_1d[j,i+1], id_corner_1d, id_field_1d, lats_fcst_lgD, lons_fcst_lgD))
                        if j < max(source[0], target[0])+buffer:
                            G.add_edge(id_corner_1d[j,i], id_corner_1d[j+1,i], 
                                       weight=weight_edge(field, id_corner_1d[j,i], id_corner_1d[j+1,i], id_corner_1d, id_field_1d, lats_fcst_lgD, lons_fcst_lgD))

            ## Shortest path according to Dijkstra's algorithm:
            optimal_path = networkx.shortest_path(G, source=id_corner_1d[source], target=id_corner_1d[target], weight='weight')
            path_cost = networkx.shortest_path_length(G, source=id_corner_1d[source], target=id_corner_1d[target], weight='weight')
            optimal_paths_splx.append(optimal_path)
            list_graphs_splx.append({'G':G, 'source':source, 'target':target})
            path_costs_splx.append(path_cost)

        ## Step C: Among the 3 optimal paths within the simplex, we remove the one with the highest cost, and save the two others:
        del(optimal_paths_splx[np.argmax(path_costs_splx)])
        del(list_graphs_splx[np.argmax(path_costs_splx)])
        optimal_paths.extend(optimal_paths_splx)
        list_graphs.extend(list_graphs_splx)


    ## Now that the low points are linked "inside" the simplices, we still need to loop over the "external" low points, i.e., the low points 
    ##    that are on an outer edge of the Delaunay triangulation, and link them to the boundaries of the domain.
    #------------------------------------------------------------------------------------------------------

    ## The "external" low points are identified as they belong to an edge that is part of only one simplex:
    nexternal_low = np.sum(nsplx_sharing_edge == 1)
    for l in range(nexternal_low):    
        ylow, xlow = (ylow_all.ravel()[nsplx_sharing_edge.ravel() == 1][l], xlow_all.ravel()[nsplx_sharing_edge.ravel() == 1][l])

        ## Find in between which centroids A and B this external low point is, and what's the third point (C) of the simplex:
        A = np.where(nsplx_sharing_edge == 1)[0][l]
        B = np.where(nsplx_sharing_edge == 1)[1][l]
        splx = np.where([(A in triang.simplices[s,:]) and (B in triang.simplices[s,:]) for s in range(nsimplex)])[0][0]
        C = np.setdiff1d(triang.simplices[splx,:], [A,B])[0]

        ## The idea is to delineate a zone for the Dijkstra's algorithm, which opens to the boundaries of the domain and is delineatined by the line BA + the interior af ABC
        BA = (LonLat_centroid[A,0] - LonLat_centroid[B,0], LonLat_centroid[A,1] - LonLat_centroid[B,1])
        BC = (LonLat_centroid[C,0] - LonLat_centroid[B,0], LonLat_centroid[C,1] - LonLat_centroid[B,1])
        CA = (LonLat_centroid[A,0] - LonLat_centroid[C,0], LonLat_centroid[A,1] - LonLat_centroid[C,1])
        CB = (LonLat_centroid[B,0] - LonLat_centroid[C,0], LonLat_centroid[B,1] - LonLat_centroid[C,1])
        BAxBC = BA[0]*BC[1] - BA[1]*BC[0]  # Cross product of BA with BC
        CAxCB = CA[0]*CB[1] - CA[1]*CB[0]
        CBxCA = -CAxCB
        G = networkx.Graph()
        for j in range(corner_vals.shape[0]):
            for i in range(corner_vals.shape[1]):
                Bji = (lons_fcst_lgD_corner[i] - LonLat_centroid[B,0], lats_fcst_lgD_corner[j] - LonLat_centroid[B,1])
                Cji = (lons_fcst_lgD_corner[i] - LonLat_centroid[C,0], lats_fcst_lgD_corner[j] - LonLat_centroid[C,1])
                BAxBji = BA[0]*Bji[1] - BA[1]*Bji[0]  # Cross product of CB with C(j,i)
                CBxCji = CB[0]*Cji[1] - CB[1]*Cji[0]
                CAxCji = CA[0]*Cji[1] - CA[1]*Cji[0]
#                 if np.sign(BAxBji) != np.sign(BAxBC) and np.sign(CBxCji) == np.sign(CBxCA) and np.sign(CAxCji) == np.sign(CAxCB):
#                 if np.sign(CBxCji) == np.sign(CBxCA) and np.sign(CAxCji) == np.sign(CAxCB):
                if np.sign(BAxBC) != np.sign(BAxBji) or (np.sign(BAxBC) == np.sign(BAxBji) and np.sign(CBxCji) == np.sign(CBxCA) and np.sign(CAxCji) == np.sign(CAxCB)):
                    if j < corner_vals.shape[0] - 1:
                        G.add_edge(id_corner_1d[j,i], id_corner_1d[j+1,i], 
                                   weight=weight_edge(field, id_corner_1d[j,i], id_corner_1d[j+1,i], id_corner_1d, id_field_1d, lats_fcst_lgD, lons_fcst_lgD))
                    if i < corner_vals.shape[1] - 1:
                        G.add_edge(id_corner_1d[j,i], id_corner_1d[j,i+1], 
                                   weight=weight_edge(field, id_corner_1d[j,i], id_corner_1d[j,i+1], id_corner_1d, id_field_1d, lats_fcst_lgD, lons_fcst_lgD))
                    ## To make sure the low point is connected to the graph:
                    if j == ylow and np.abs(i - xlow) == 1 or i == xlow and np.abs(j - ylow) == 1:
                        G.add_edge(id_corner_1d[j,i], id_corner_1d[ylow,xlow], 
                                   weight=weight_edge(field, id_corner_1d[j,i], id_corner_1d[ylow,xlow], id_corner_1d, id_field_1d, lats_fcst_lgD, lons_fcst_lgD))

        ## Shortest path (according to Dijkstra's algorithm) from the source (i.e., the external low point) to all nodes in the graph (i.e., the search area delineated by CA and CB):
        optimal_path_allNodes = networkx.shortest_path(G, source=id_corner_1d[ylow,xlow], target=None, weight='weight')
        path_cost_allNodes = networkx.shortest_path_length(G, source=id_corner_1d[ylow,xlow], target=None, weight='weight')

        ## We then select the node associated with the minimal cost, but only among the nodes that are at the boundary of the domain:
        targets_at_boundary = np.intersect1d(np.array(list(optimal_path_allNodes.keys())), id_corner_1d_boundary)
        which_is_shorter = np.argmin([path_cost_allNodes[targets_at_boundary[t]] for t in range(targets_at_boundary.size)])
        optimal_path = optimal_path_allNodes[targets_at_boundary[which_is_shorter]]
        optimal_paths.append(optimal_path)
        list_graphs.append({'G':G, 'source':(ylow,xlow), 'target':None})



    ## We then label (with different integers) the different areas that are delinetated by the paths we have drawn, using a "flood fill" algorithm:
    #------------------------------------------------------------------------------------------------------

    ## To do so, we first need to construct, from all the optimal paths we have found, the array 'fences' that gives for each pixel (j,i) 
    ##     whether or not there is a fence down (1st dim = 0), left (1), up (2), right (3):

    fences = np.full((4,nyfd, nxfd), False)     ## 1st dimension:  0, 1, 2, 3 -> down, left, up, right
    for pth in optimal_paths:
        length = len(pth)
        for e in range(length-1):
            try:
                pix_1, pix_2 = get_pixels_around_edge(pth[e], pth[e+1], id_corner_1d, id_field_1d, lats_fcst_lgD, lons_fcst_lgD)
                y1, x1 = np.where(id_field_1d == pix_1)[0][0], np.where(id_field_1d == pix_1)[1][0]
                y2, x2 = np.where(id_field_1d == pix_2)[0][0], np.where(id_field_1d == pix_2)[1][0]

                if x1 < x2:   # pix_2 at the right of pix_1
                    fences[3,y1,x1] = True    # not possible to go right from pix_1
                    fences[1,y2,x2] = True    # not possible to go left from pix_2
                if x2 < x1:   # pix_1 at the right of pix_2
                    fences[1,y1,x1] = True    # not possible to go left from pix_1
                    fences[3,y2,x2] = True    # not possible to go right from pix_2
                if y1 < y2:   # pix_2 at the top of pix_1
                    fences[2,y1,x1] = True    # not possible to go up from pix_1
                    fences[0,y2,x2] = True    # not possible to go down from pix_2
                if y2 < y1:   # pix_1 at the top of pix_2
                    fences[0,y1,x1] = True    # not possible to go down from pix_1
                    fences[2,y2,x2] = True    # not possible to go up from pix_2

            ## If a path is on the boundary, we don't put a fence:
            except EdgeAtBoundary:
                pass    

    ## Labelling using the recursive flood fill algorithm:
    field_label = np.zeros((nyfd, nxfd), dtype=np.int32)
    labelling(field_label, fences)

    differentLabels = np.unique(field_label)
    nb_differentLabels = differentLabels.size


    ## Finally, we assign to each of these area the integer of square calibration domain it belongs to, using as criteria the area with the most cumulative rainfall
    #------------------------------------------------------------------------------------------------------
    differentCalibDomains = np.unique(field_calibDomain)

    field_parDomain = np.zeros_like(field_label, dtype=np.int32)
    for l in range(nb_differentLabels):
        sum_precip = np.zeros(D)
        for d in range(D):
            id_crossArea = np.where(np.logical_and(field_label == differentLabels[l], field_calibDomain == differentCalibDomains[d]))
            if id_crossArea[0].size > 0:
                sum_precip[d] = np.sum(field[id_crossArea]) + 0.00001     #  '+ 0.00001' for the cases where a labelled area doesn't havve any precip
        field_parDomain[np.where(field_label == differentLabels[l])] = differentCalibDomains[np.argmax(sum_precip)]

    return field_parDomain









## Function that, given any point a, will return all other points which are vertices of any
##     simplex (i.e. triangle) that a is also a vertex of (the neighbors of a in the triangulation):
def find_neighbors(pindex, triang):
    return triang.vertex_neighbor_vertices[1][triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex+1]]




## Function that smoothes a field by performing a convolution with a radius R (expressed in pixels)
##    Masked pixels must be provided in 'mask_field' (optionnal). These pixels will be ignored in the convolution.
#---------------------------------------------------------------

## Wrapper function (to add an optionnal mask, and pass the fields as 1d arrays):
def convolution(field, radius_in_pix, mask_field=None):
    ny, nx = field.shape
    if mask_field is None:
        mask_field = np.full((ny,nx), False)
    
    field_convolved = ma.array(convolution_njit(field.reshape(ny*nx), ny, nx, radius_in_pix, mask_field.reshape(ny*nx)), mask=False)
    field_convolved.mask[mask_field] = True
    
    return field_convolved

## Wrapped function (in Numba):    
@njit
def convolution_njit(field_1d, ny, nx, radius_in_pix, mask_field_1d):
    field_convolved = np.full((ny,nx), 999.)
    y = np.arange(ny)
    x = np.arange(nx)   
    y_1d = np.repeat(y, repeats=nx)
    x_1d = np.repeat(x, repeats=ny).reshape(nx,ny).transpose().copy().reshape(ny*nx)
    y_mat = y_1d.reshape(ny,nx)
    x_mat = x_1d.reshape(ny,nx)
    for j in range(ny):
        for i in range(nx):
            id_toAverage = np.where(np.sqrt(np.square(y_mat[j,i] - y_1d) + np.square(x_mat[j,i] - x_1d)) <= radius_in_pix)[0]
            if np.all(mask_field_1d[id_toAverage]):
                continue
            else:
                id_toAverage_withMask = id_toAverage[np.invert(mask_field_1d[id_toAverage])]
            field_convolved[j,i] = np.mean(field_1d[id_toAverage_withMask])
    return field_convolved




## Function that bilinearly interpolates a field to its corner coordinates
##    Masked data (provided via the optional 'mask_field') are ignored in the interpolation.
#---------------------------------------------------------------

## Wrapper for compute_corner_values_njit (to add a mask)
def compute_corner_values(field, mask_field=None):
    ny, nx = field.shape
    if mask_field is None:
        mask_field = np.full((ny,nx), False)
    
    corner_vals = ma.array(compute_corner_values_njit(field, mask_field), mask=False)
    corner_vals.mask[np.abs(corner_vals - 999.) < 1e-8] = True
    
    return corner_vals

@njit
def compute_corner_values_njit(field, mask_field):
    ny, nx = field.shape
    field_1d = field.reshape(ny*nx)
    mask_field_1d = mask_field.reshape(ny*nx)
    id_field1d = np.arange(ny*nx).reshape(ny,nx)
    corner_vals = np.full((ny+1,nx+1), 999.)
    for j in range(ny+1):
        for i in range(nx+1):
            a = (j-1) if j > 0 else 0
            b = (j+1)
            c = (i-1) if i > 0 else 0
            d = (i+1)
            id_toAverage = id_field1d[a:b,c:d].ravel()
            if np.all(mask_field_1d[id_toAverage]):
                continue
            else:
                id_toAverage_withMask = id_toAverage[np.invert(mask_field_1d[id_toAverage])]
            corner_vals[j,i] = np.mean(field_1d[id_toAverage_withMask])            
    return corner_vals



## Custom exception          
class EdgeAtBoundary(ValueError): pass    

## Function that returns the two field pixels that are around an edge made by two corner pixels
def get_pixels_around_edge(corner_a, corner_b, id_corner_1d, id_field_1d, lats_fcst_lgD, lons_fcst_lgD):
        
    resol_lat = np.unique(np.diff(lats_fcst_lgD))[0]
    resol_lon = np.unique(np.diff(lons_fcst_lgD))[0]
    lats_fcst_lgD_corner = np.arange(lats_fcst_lgD[0]-resol_lat/2, lats_fcst_lgD[-1]+resol_lat/2+0.0001, resol_lat)
    lons_fcst_lgD_corner = np.arange(lons_fcst_lgD[0]-resol_lon/2, lons_fcst_lgD[-1]+resol_lon/2+0.0001, resol_lon)    
    
    ya, xa = np.where(id_corner_1d == corner_a)
    yb, xb = np.where(id_corner_1d == corner_b)
    lat_a, lon_a, lat_b, lon_b = lats_fcst_lgD_corner[ya][0], lons_fcst_lgD_corner[xa][0], lats_fcst_lgD_corner[yb][0], lons_fcst_lgD_corner[xb][0]
    
    if not ((lat_a == lat_b and np.abs(np.abs(lon_a-lon_b) - resol_lon) < 1e-8) or (lon_a == lon_b and np.abs(np.abs(lat_a-lat_b) - resol_lat) < 1e-8)):
        raise ValueError("'corner_a' and 'corner_b' are not adjacent")
    if lat_a == lat_b:
        if np.abs(lat_a - lats_fcst_lgD_corner[0]) < 1e-8 or np.abs(lat_a -lats_fcst_lgD_corner[-1]) < 1e-8:
            raise EdgeAtBoundary("'corner_a' and 'corner_b' are at the boundaries")
        lat_1 = lat_a - resol_lat/2
        lat_2 = lat_a + resol_lat/2
        lon_1 = lon_a + (lon_b - lon_a)/2
        lon_2 = lon_1
    elif lon_a == lon_b:
        if np.abs(lon_a - lons_fcst_lgD_corner[0]) < 1e-8 or np.abs(lon_a - lons_fcst_lgD_corner[-1]) < 1e-8:
            raise EdgeAtBoundary("'corner_a' and 'corner_b' are at the boundaries")
        lat_1 = lat_a + (lat_b - lat_a)/2
        lat_2 = lat_1
        lon_1 = lon_a - resol_lon/2
        lon_2 = lon_a + resol_lon/2
    
    pix_1 = id_field_1d[np.where(np.abs(lats_fcst_lgD - lat_1) < 1e-8)[0][0], np.where(np.abs(lons_fcst_lgD - lon_1) < 1e-8)[0][0]]
    pix_2 = id_field_1d[np.where(np.abs(lats_fcst_lgD - lat_2) < 1e-8)[0][0], np.where(np.abs(lons_fcst_lgD - lon_2) < 1e-8)[0][0]]
    
    return(pix_1, pix_2)
        

    
## Function that returns the weight of the edge between the two points corner_a and corner_b.
def weight_edge(field, corner_a, corner_b, id_corner_1d, id_field_1d, lats_fcst_lgD, lons_fcst_lgD):
   
    resol_lat = np.unique(np.diff(lats_fcst_lgD))[0]
    resol_lon = np.unique(np.diff(lons_fcst_lgD))[0]
    lats_fcst_lgD_corner = np.arange(lats_fcst_lgD[0]-resol_lat/2, lats_fcst_lgD[-1]+resol_lat/2+0.0001, resol_lat)
    lons_fcst_lgD_corner = np.arange(lons_fcst_lgD[0]-resol_lon/2, lons_fcst_lgD[-1]+resol_lon/2+0.0001, resol_lon)    

    ny, nx = field.shape
    id_field_1d = np.arange(ny*nx).reshape(ny,nx)
    
    try:
        pix_1, pix_2 = get_pixels_around_edge(corner_a, corner_b, id_corner_1d, id_field_1d, lats_fcst_lgD, lons_fcst_lgD)
        val_1, val_2 = field[np.where(id_field_1d == pix_1)][0], field[np.where(id_field_1d == pix_2)][0]

        weight = np.square(min(val_1, val_2) + min(val_1, val_2)/10) + 0.01     # the '+ 0.01' aims at making the optimal paths shorter in case of zeros everywhere
        try:
            if weight.mask == True:
                weight = 0.0001
        except AttributeError:
            pass
        
    except EdgeAtBoundary:
        weight = 0.0001
    
    return weight




## Function that assigns different labels (integers) to bounded areas delineated by "fences":
def labelling(field_label, fences, start_label_val=1):
    ny, nx = field_label.shape
    j, i = 0, 0
    label_val = start_label_val
    while np.any(field_label == 0):
        if field_label[j,i] == 0:
            recursive_flood_fill(j, i, field_label, fences, label_val)
            label_val = label_val + 1
        if i < nx-1:
            i = i + 1
        else: 
            if j < ny-1:
                i = 0
                j = j + 1
        
## Recursive "flood fill" algorithm:
def recursive_flood_fill(j, i, field_label, fences, label_val):
    ny, nx = field_label.shape
    
    ## If (j,i) has already been labeled, we end the function, otherwise we label (j,i)
    if field_label[j,i] != 0:
        return
    field_label[j,i] = label_val
    
    ## If from (j,i) there is no fence down, left, up or right, respectively, we move in that direction and call 'recursive_flood_fill()' again:
    if fences[0,j,i] == False and j > 0:
        recursive_flood_fill(j-1, i, field_label, fences, label_val)
    if fences[1,j,i] == False and i > 0:
        recursive_flood_fill(j, i-1, field_label, fences, label_val)
    if fences[2,j,i] == False and j < (ny-1):
        recursive_flood_fill(j+1, i, field_label, fences, label_val)
    if fences[3,j,i] == False and i < (nx-1):
        recursive_flood_fill(j, i+1, field_label, fences, label_val)



