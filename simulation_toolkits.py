import warnings
warnings.filterwarnings("ignore")

from scipy import sparse
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# import mnnpy2.mnnpy as mnnpy
import SpaGCN as spg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
import random, torch
import os
from IPython.display import clear_output
from sklearn.decomposition import PCA
import umap

# sc.logging.print_versions()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3

def read_st_data_for_scanpy(pathway, library_ID, resolution, reverse_resolution):
    from pathlib import Path
    path = pathway
    path = Path(path)
    library_id = library_ID
    adata = sc.read_10x_mtx(path=pathway, cache=True)
    adata.uns["spatial"] = dict()
 


    # from h5py import File
    # with File('V1_Mouse_Brain_Sagittal_Anterior_filtered_feature_bc_matrix.h5', mode="r") as f:
    #     attrs = dict(f.attrs)
    adata.uns["spatial"][library_id] = dict()


    files = dict(
        tissue_positions_file=path / 'spatial/tissue_positions_list.csv',
        scalefactors_json_file=path / 'spatial/scalefactors_json.json',
        hires_image=path / 'spatial/tissue_hires_image.png',
        lowres_image=path / 'spatial/tissue_lowres_image.png',
    )


    # check if files exists, continue if images are missing
    from matplotlib.image import imread
    adata.uns["spatial"][library_id]['images'] = dict()
    res = resolution #'lowres'
    adata.uns["spatial"][library_id]['images'][res] = imread(str(files[f'{res}_image']))




    # read json scalefactors
    import json
    adata.uns["spatial"][library_id]['scalefactors'] = json.loads(
        files['scalefactors_json_file'].read_bytes()
    )

    if (reverse_resolution) :
        #### change resolution back, since I reversed resolutions for seurat 
        tmp = adata.uns["spatial"][library_id]['scalefactors']['tissue_hires_scalef']
        adata.uns["spatial"][library_id]['scalefactors']['tissue_hires_scalef'] = adata.uns["spatial"][library_id]['scalefactors']['tissue_lowres_scalef']
        adata.uns["spatial"][library_id]['scalefactors']['tissue_lowres_scalef'] = tmp

    # adata.uns["spatial"][library_id]["metadata"] = {
    #     k: (str(attrs[k], "utf-8") if isinstance(attrs[k], bytes) else attrs[k])
    #     for k in ("chemistry_description", "software_version")
    #     if k in attrs
    # }

    # read coordinates
    positions = pd.read_csv(files['tissue_positions_file'], header=None)
    positions.columns = [
        'barcode',
        'in_tissue',
        'array_row',
        'array_col',
        'pxl_col_in_fullres',
        'pxl_row_in_fullres',
    ]
    positions.index = positions['barcode']

    adata.obs = adata.obs.join(positions, how="left")

    adata.obsm['spatial'] = adata.obs[
        ['pxl_row_in_fullres', 'pxl_col_in_fullres']
    ].to_numpy()
    adata.obs.drop(
        columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'],
        inplace=True,
    )

    # put image path in uns
    # get an absolute path
    # adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = pathway + 'spatial/tissue_hires_image.png'

    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True) 
    
    return adata

def get_spatial_neighbors(data, coordinate, k):
    """
    data: adata.X
    coordinate: adata.obsm['spatial']
    k: number of neighbors
    """
    
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors = k)
    knn.fit(coordinate)
    distance_mat, neighbors_mat = knn.kneighbors(coordinate)
    return neighbors_mat

def spatial_mnn2(data, coordinate, k, num_pc):
    """
    data: adata.X
    coordinate: adata.obsm['spatial']
    k: number of neighbors
    num_pc: Number of PC to be used
    """
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=num_pc)
    # PC = np.zeros((n, k))
    neighbors_mat = get_spatial_neighbors(data, coordinate, k)
    n = data.shape[0]
    p = data.shape[1]
    mnn_matrix = np.zeros((n, p + num_pc*k))
    for obs, neighbors in enumerate(neighbors_mat):
        A = np.zeros((k, k))
        for i in range(len(neighbors)):
            for j in range(len(neighbors)):
                A[i, j] = np.linalg.norm((data[neighbors[i], :] - data[neighbors[j], :]))
        pca.fit(A)
        # PC[obs, :] = pca.components_
        mnn_matrix[obs, :] = np.append(np.array(data[obs, :]).ravel(), pca.components_.ravel()) # gene expression + linear sum of first pc
    return mnn_matrix

def get_corrected_pc(corrected, k, num_pc):
    '''
    corrected: A batch of Integrated AnnData after MNN correction
    k: Number of Neighbors
    num_pc: Number of PC to be used
    '''
    corrected_pc = spatial_mnn2(corrected.X, corrected.obsm['spatial'], k, num_pc)
    adata_pc = sc.AnnData(np.array(corrected_pc[:, -k*num_pc:]), 
                         dict(obs_names=corrected.obs_names))
    a = corrected.copy().T.concatenate(adata_pc.T)
    b = a.T
    b.obs.columns = corrected.obs.columns
    b.obsm['spatial'] = corrected.obsm['spatial']
    return b

def silhouette_plot(adata, cluster_str, xlim1, xlim2):
    '''
    adata: AnnData with cluster labels
    cluster_str: the string name identifying clusters
    xlim1, xlim2: range of xlim on the plot
    '''
    labels = adata.obs[cluster_str].tolist()
    sh_score = metrics.silhouette_score(adata.X, labels)
    sh_score
    # Create a subplot with 1 row and 2 columns
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(10, 8)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([xlim1, xlim2])
    n_clusters = len(set(labels))
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, adata.X.shape[0] + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    # clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    # cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters

    silhouette_avg = silhouette_score(adata.X, labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(adata.X, labels)

    y_lower = 10
    labels = adata.obs[cluster_str].tolist()
    labels = np.array([int(l) for l in labels])
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

def silhouette_plot_umap(adata, cluster_str, xlim1, xlim2):
    '''
    adata: AnnData with cluster labels
    cluster_str: the string name identifying clusters
    xlim1, xlim2: range of xlim on the plot
    '''
    labels = adata.obs[cluster_str].tolist()
    sh_score = metrics.silhouette_score(adata.obsm['X_umap'], labels)
    sh_score
    # Create a subplot with 1 row and 2 columns
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(10, 8)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([xlim1, xlim2])
    n_clusters = len(set(labels))
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, adata.obsm['X_umap'].shape[0] + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    # clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    # cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters

    silhouette_avg = silhouette_score(adata.obsm['X_umap'], labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(adata.obsm['X_umap'], labels)

    y_lower = 10
    labels = adata.obs[cluster_str].tolist()
    labels = np.array([int(l) for l in labels])
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )    
    
def SpaGCN_implement(adata0, normalize = False, log1p = False, s=1, b=49, p=0.5, x_pixel=None, y_pixel=None, histology=False, n_clusters = 7):
    '''
    adata: Anndata to be clustered
    s: weight of histology
    b: square area of pixels to calculate weighted RGB value
    p: the target value to find l, such that the average of row sum of A-I across all spots is equal to p
    x_pixel, y_pixel: pixel location matrix
    histology: whether to use the histology data
    n_clusters: number of clusters expected
    '''
    import copy
    adata = copy.deepcopy(adata0)
    if x_pixel == None and y_pixel == None:
        x_pixel=adata.obsm['spatial'][:,0]
        y_pixel=adata.obsm['spatial'][:,1]
    #If histlogy image is not available, SpaGCN can calculate the adjacent matrix using the fnction below
    adj=spg.calculate_adj_matrix(x=x_pixel,y=y_pixel, histology=histology)
    spg.prefilter_genes(adata,min_cells=3) # avoiding all genes are zeros
    spg.prefilter_specialgenes(adata)
    #Normalize and take log for UMI
    adata = sc.pp.normalize_per_cell(adata, copy=True)
    adata = sc.pp.log1p(adata, copy=True)
    #Find the l value given p
    l=spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    #For this toy data, we set the number of clusters=7 since this tissue has 7 layers
    #Set seed
    r_seed=t_seed=n_seed=100
    #Search for suitable resolution
    res=spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)
    clf=spg.SpaGCN()
    clf.set_l(l)
    #Set seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    #Run
    clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, prob=clf.predict()
    adata.obs["pred"]= y_pred
    adata.obs["pred"]=adata.obs["pred"].astype('category')
    #Do cluster refinement(optional)
    #shape="hexagon" for Visium data, "square" for ST data.
    adj_2d=spg.calculate_adj_matrix(x=x_pixel,y=y_pixel, histology=False)
    refined_pred=spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
    adata.obs["refined_pred"]=refined_pred
    adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')
    #Save results
    #adata.write_h5ad("./sample_results/results.h5ad")
    os.system('clear')
    return adata

def plot_domains(adata, domains, size = None):
    '''
    adata: AnnData object with 'pred', 'refined_pred' in adata.obs
    domains: domain to be plotted
    size: size of point
    '''
    plot_color=["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#3A84E6","#997273","#787878","#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]
    #Plot spatial domains
    # domains="pred"
    num_celltype=len(adata.obs[domains].unique())
    adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
    adata.obs["x_pixel"]=adata.obsm['spatial'][:,0]
    adata.obs["y_pixel"]=adata.obsm['spatial'][:,1]
    plt.rcParams["figure.figsize"] = (8, 8)
    if size == None:
        size = 100000/adata.shape[0]
    ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=size)
    ax.set_aspect('equal', 'box')
    ax.axes.invert_yaxis()
    # ax.axes.invert_xaxis()

def sc_leiden_visualization(adata, normalize = False, log1p = False, resolution = 1):
    import copy
    adata1 = copy.deepcopy(adata)
    if normalize == True:
        adata1 = sc.pp.normalize_total(adata1, copy = True)
    if log1p == True:
        adata1 = sc.pp.log1p(adata1, copy = True)
    # sc.pp.highly_variable_genes(adata2_visium, flavor="seurat", n_top_genes=2000)
    sc.pp.neighbors(adata1)
    sc.tl.umap(adata1)
    sc.tl.leiden(adata1, key_added="clusters", resolution = resolution)
    clear_output()
    plt.rcParams["figure.figsize"] = (4, 4)
    sc.pl.umap(adata1, color=["total_counts", "clusters"], wspace=0.4)
    sc.pl.spatial(adata1, img_key="hires", color=["clusters"])

def get_lambda_gene(adata):
    '''
    Calculate average gene expression in each cell
    '''
    qc_metrics = sc.pp.calculate_qc_metrics(adata)
    gene_name = qc_metrics[1].index
    gene_counts = qc_metrics[1]['total_counts']
    total_gene_counts = np.sum(gene_counts)
    gene_pct = pd.DataFrame({'percentage' : gene_counts / total_gene_counts})
    # gene_pct = gene_pct.sort_values(by = "percentage", ascending = False)
    lambda_gene = gene_pct * total_gene_counts / adata.n_obs
    return lambda_gene

def gen_poisson_sample(n, lambda_list):
    '''
    n: number of cells to be generated
    lambda_list: list of lambda for poisson distribution
    '''
    df = np.empty((n, len(lambda_list)))
    for i in range(n):
        df[i, ] = np.ravel(np.random.poisson(lam = lambda_list))
    return df

def gen_negbin_sample(n, lambda_list, var_ratio):
    '''
    n: number of cells to be generated
    lambda_list: list of lambda for poisson distribution
    var_ratio: ratio of variance to mean (must > 1)
    '''
    p = 1-1/var_ratio
    r = lambda_list*(1-p)/p
    idx_zero = np.where(r==0)[0]
    idx_nonzero = np.where(r!=0)[0]
    df = np.empty((n, len(lambda_list)))
    for i in range(n):
        df[i, idx_nonzero] = np.ravel(np.random.negative_binomial(n = r.iloc[idx_nonzero,], p = p))
        df[i, idx_zero] = 0
    return df

def combine_adata(adata1, adata2):
    X = sparse.csr_matrix(np.concatenate((adata1.X.todense(), adata2.X.todense())))
    obs = pd.DataFrame({})
    for key in adata1.obs.keys():
        obs1 = pd.DataFrame({key: pd.concat([adata1.obs[key], adata2.obs[key]])})
        obs = pd.concat([obs, obs1], axis=1)
    var = adata1.var
    adata = sc.AnnData(X = X, obs = obs, var = var)
    adata.uns = adata1.uns
    adata.obsm['spatial'] = np.concatenate((adata1.obsm['spatial'], adata2.obsm['spatial']))
    return adata

from collections import deque
def pref_to_rank(pref):
    return {
        a: {b: idx for idx, b in enumerate(a_pref)}
        for a, a_pref in pref.items()
    }

def gale_shapley(*, A, B, A_pref, B_pref):
    """Create a stable matching using the
    Gale-Shapley algorithm.
    
    A -- set[str].
    B -- set[str].
    A_pref -- dict[str, list[str]].
    B_pref -- dict[str, list[str]].

    Output: list of (a, b) pairs.
    """
    B_rank = pref_to_rank(B_pref)
    ask_list = {a: deque(bs) for a, bs in A_pref.items()}
    pair = {}
    #
    remaining_A = set(A)
    while len(remaining_A) > 0:
        a = remaining_A.pop()
        b = ask_list[a].popleft()
        if b not in pair:
            pair[b] = a
        else:
            a0 = pair[b]
            b_prefer_a0 = B_rank[b][a0] < B_rank[b][a]
            if b_prefer_a0:
                remaining_A.add(a)
            else:
                remaining_A.add(a0)
                pair[b] = a
    #
    return [(a, b) for b, a in pair.items()]

def get_celltype_correspondence(spa_T_sim, label='pred'):
    tab = pd.crosstab(spa_T_sim.obs[label], spa_T_sim.obs['cell_type'])
    ind = list(tab.index)
    spa_T_sim.obs['pred_cell_type'] = 0
    for i in ind:
        spa_T_sim.obs['pred_cell_type'][spa_T_sim.obs[label]==i] = tab.columns[np.argmax(tab.loc[i,:])]
    return spa_T_sim