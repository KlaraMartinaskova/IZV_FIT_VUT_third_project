#!/usr/bin/python3.10
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
import sklearn.cluster
from sklearn.cluster import KMeans
import numpy as np
import os

# AUTOR: Klára Martinásková

def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """ Konvertovani DataFrame do geopandas.GeoDataFrame se spravnym kodovanim"""
    geo_df = df.copy()
    geo_df = geo_df.dropna(subset=['d', 'e'])

    geo_df = geopandas.GeoDataFrame(geo_df,
                                    geometry=geopandas.points_from_xy(  
                                                                        geo_df["d"], 
                                                                        geo_df["e"]
                                                                    ),
                                    crs="EPSG:5514")
    return geo_df   
   

def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """ Vykresleni grafu s nehodami s alkoholem pro roky 2018-2021 """

    # select region 'ZLK' and p11 >= 3 (drugs)
    #set geometry crs to gdf
    gdf = gdf.set_geometry(gdf.centroid).to_crs("EPSG:5514")
    gdf_OH_drugs =  gdf[(gdf.region == 'ZLK') & (gdf.p11 >= 3)]

    # select p2a where is 2018 in gdf
    gdf_18 = gdf_OH_drugs[gdf_OH_drugs['p2a'].dt.year == 2018]
    # select p2a where is 2019 in gdf
    gdf_19 = gdf_OH_drugs[gdf_OH_drugs['p2a'].dt.year == 2019]
    # select p2a where is 2020 in gdf
    gdf_20 = gdf_OH_drugs[gdf_OH_drugs['p2a'].dt.year == 2020]
    # select p2a where is 2021 in gdf
    gdf_21 = gdf_OH_drugs[gdf_OH_drugs['p2a'].dt.year == 2021]
    
    # Subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot data
    gdf_18.plot(ax=axs[0,0], markersize=1, label="Nehody (2018)", color='green')
    gdf_18.boundary.plot(ax=axs[0,0], color="0.7")
    gdf_19.plot(ax=axs[0,1], markersize=1, label="Nehody (2019)", color='red')
    gdf_19.boundary.plot(ax=axs[0,1], color="0.7")
    gdf_20.plot(ax=axs[1,0], markersize=1, label="Nehody (2020)", color='blue')
    gdf_20.boundary.plot(ax=axs[1,0], color="0.7")
    gdf_21.plot(ax=axs[1,1], markersize=1, label="Nehody (2021)", color='purple')
    gdf_21.boundary.plot(ax=axs[1,1], color="0.7")
    
    # Add basemaps
    ctx.add_basemap(axs[0,0], crs=gdf_18.crs.to_string(), source=ctx.providers.Stamen.TonerLite, zoom=10, alpha=0.9)
    axs[0,0].set_title("Nehody ve Zlínském kraji (2018)")
    axs[0,0].axis("off")

    ctx.add_basemap(axs[0,1], crs=gdf_19.crs.to_string(), source=ctx.providers.Stamen.TonerLite, zoom=10, alpha=0.9)
    axs[0,1].set_title("Nehody ve Zlínském kraji (2019)")
    axs[0,1].axis("off")

    ctx.add_basemap(axs[1,0], crs=gdf_20.crs.to_string(), source=ctx.providers.Stamen.TonerLite, zoom=10, alpha=0.9)
    axs[1,0].set_title("Nehody ve Zlínském kraji (2020)")
    axs[1,0].axis("off")

    ctx.add_basemap(axs[1,1], crs=gdf_21.crs.to_string(), source=ctx.providers.Stamen.TonerLite, zoom=10, alpha=0.9)
    axs[1,1].set_title("Nehody ve Zlínském kraji (2021)")
    axs[1,1].axis("off")

    if(fig_location):
        dir_name = os.path.dirname(fig_location)
        if(dir_name != ''):
            if not os.path.exists(dir_name):
                os.makedirs(
                    dir_name
                )
        plt.savefig(fig_location, dpi=120)
    if(show_figure):
        plt.show()


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """ Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru """
    
    # Subplots
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    

    gdf_JHM = gdf[(gdf["region"] == "JHM") ]
    gdf_JHM = gdf_JHM[(gdf_JHM["p36"] == 1)|(gdf_JHM["p36"] == 2)|(gdf_JHM["p36"]==3)]
    
    # print(gdf_JHM)
    gdf_JHM = gdf_JHM.set_geometry(gdf_JHM.centroid).to_crs(epsg=3857)

    
    coords = np.dstack([gdf_JHM.geometry.x, gdf_JHM.geometry.y]).reshape(-1, 2) # coordinates
    
    # Chosen agglomerative clustering
    #  - this graph was most similar to the graph in the assignment
    #  - the results were also the best
    #  - outliers generally do not get added to a cluster until the end of the process  when all 
    #       of the other observations have already been handled, so the presence of a few outliers 
    #       is not likely to affect the algorithm
    #
    # I tried also KMeans or MiniBatch KMeans clustering, but the results were not as good as with 
    # Agglomerative clustering. 
    # gdf_JHM["frequency_group"] = sklearn.cluster.KMeans(n_clusters=20).fit(coords).labels_
    # gdf_JHM["frequency_group"] = sklearn.cluster.MiniBatchKMeans(n_clusters=20).fit(coords).labels_

    # Cluster into groups with frequancy of accidents
    gdf_JHM["accident_group"] = sklearn.cluster.AgglomerativeClustering(n_clusters=20).fit(coords).labels_
     
    gdf_JHM = gdf_JHM.dissolve(by="accident_group", aggfunc={"p1": "count"}) # dissolve into clusters
    
    gdf_JHM.plot(ax=ax, markersize=1, column="p1", legend=True) # plot accidents

    ax.set_axis_off()
    ctx.add_basemap(ax, crs=gdf_JHM.crs.to_string(), alpha=0.9, attribution_size=6,
                    reset_extent=False, source=ctx.providers.Stamen.TonerLite)
    ax.set_title("Nehody v JMK kraji (silnice 1., 2. a 3. třídy)", fontsize="small")
    plt.tight_layout()

    if(fig_location):
        dir_name = os.path.dirname(fig_location)
        if(dir_name != ''):
            if not os.path.exists(dir_name):
                os.makedirs(
                    dir_name
                )
        plt.savefig(fig_location, dpi=120)
    if(show_figure):
        plt.show()


if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    gdf = pd.read_pickle("accidents.pkl.gz")
    gdf['p2a'] = pd.to_datetime(gdf['p2a'], format='%Y-%m-%d %H:%M:%S') # convert column 'p2a' to datetime
    gdf = make_geo(gdf) 

    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)
