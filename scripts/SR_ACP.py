# Análise de Componentes Principais com dados de Seguro Rural

## Importando as bibliotecas

!pip install geopandas==0.8.1
!pip install --upgrade pyshp
!pip install shapely==1.7.0
!pip install jenkspy
!pip install --upgrade descartes
!pip install mapclassify==2.3.0 libpysal==4.3.0 splot==1.1.3

import pandas as pd
import numpy as np
import scipy.stats as stats

# para gráficos
import jenkspy

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches, colors
from matplotlib.lines import Line2D
from matplotlib.collections import EventCollection
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection  
from matplotlib import rcParams
import matplotlib.image as mpimg
import matplotlib.ticker as mtick
import seaborn as sns

# para a análise de componentes principais
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler 

# para a análise de dados espaciais
import geopandas
import splot
import mapclassify as mc
from libpysal.weights import Queen
from esda import Moran, Moran_Local, G_Local
from splot.esda import plot_moran, moran_scatterplot, lisa_cluster, plot_local_autocorrelation

import shapefile
from google.colab import drive, files

drive.mount("/content/drive")

## Funções

def moran_hot_cold_spots(moran_loc, p=0.05):
    sig = 1 * (moran_loc.p_sim < p)
    HH = 1 * (sig * moran_loc.q == 1)
    LL = 3 * (sig * moran_loc.q == 3)
    LH = 2 * (sig * moran_loc.q == 2)
    HL = 4 * (sig * moran_loc.q == 4)
    cluster = HH + LL + LH + HL
    return cluster

def mask_local_auto(moran_loc, p=0.5):
    
    # create a mask for local spatial autocorrelation
    cluster = moran_hot_cold_spots(moran_loc, p)

    cluster_labels = ['não significativo', 'AA', 'BA', 'BB', 'AB']
    labels = [cluster_labels[i] for i in cluster]

    colors5 = {0: 'lightgrey',
               1: '#d7191c',
               2: '#abd9e9',
               3: '#2c7bb6',
               4: '#fdae61'}
    colors = [colors5[i] for i in cluster]  # for Bokeh
    # for MPL, keeps colors even if clusters are missing:
    x = np.array(labels)
    y = np.unique(x)

    colors5_mpl = {'AA': '#d7191c',
                   'BA': '#abd9e9',
                   'BB': '#2c7bb6',
                   'AB': '#fdae61',
                   'não significativo': 'lightgrey'}
    colors5 = [colors5_mpl[i] for i in y]  # for mpl

    # HACK need this, because MPL sorts these labels while Bokeh does not
    cluster_labels.sort()
    return cluster_labels, colors5, colors, labels

def lisa_cluster(moran_loc, gdf, p=0.05, ax=None,
                 legend=True, legend_kwds=None, **kwargs):

    # retrieve colors5 and labels from mask_local_auto
    _, colors5, _, labels = mask_local_auto(moran_loc, p=p)

    # define ListedColormap
    hmap = colors.ListedColormap(colors5)

    if ax is None:
        figsize = kwargs.pop('figsize', None)
        fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig = ax.get_figure()

    gdf.assign(cl=labels).plot(column='cl', categorical=True,
                               k=2, cmap=hmap, linewidth=0, ax=ax,
                               edgecolor='white', legend=False, # tirei a legenda aqui 
                               legend_kwds=legend_kwds, **kwargs)
    ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.3)) # Adiciona linha dos estados 
    ax.set_axis_off()
    ax.set_aspect('equal')
    return fig, ax

def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley,s=30, edgecolors='black', linewidths=.7)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.8)
        if labels is None:
            plt.text(coeff[i,0]* 1.1, coeff[i,1] * 1.1, "Var"+str(i+1), color = 'green', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.1, coeff[i,1] * 1.1, labels[i], color = 'g', ha = 'center', va = 'center')
 
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    ax.set(xlim=(None, 0.75), ylim=(None, 0.7))
    plt.grid()

img=mpimg.imread('/content/drive/My Drive/Mestrado/Imagens/rosa_dos_ventos_3.png')

#Regioes geograficas
sf = shapefile.Reader('/content/drive/My Drive/Mestrado/Dados/estados/estados_2010.shp')
shapes = sf.shapes()
Nshp = len(shapes)

ptchs   = []
for nshp in range(Nshp):
    pts     = np.array(shapes[nshp].points)
    prt     = shapes[nshp].parts
    par     = list(prt) + [pts.shape[0]]

    for pij in range(len(prt)):
       ptchs.append(Polygon(pts[par[pij]:par[pij+1]]))

#UF
sf_uf = shapefile.Reader('/content/drive/My Drive/Mestrado/Dados/estados/estados_2010.shp')
shapes = sf.shapes()
shapes_uf = sf_uf.shapes()
Nshp_uf = len(shapes_uf)

ptchs_uf   = []
for nshp_uf in range(Nshp_uf):
    pts_uf     = np.array(shapes_uf[nshp_uf].points)
    prt_uf     = shapes_uf[nshp_uf].parts
    par_uf     = list(prt_uf) + [pts_uf.shape[0]]

    for pij_uf in range(len(prt_uf)):
       ptchs_uf.append(Polygon(pts_uf[par_uf[pij_uf]:par_uf[pij_uf+1]]))


fig     = plt.figure(figsize = (9,9))
ax      = fig.add_subplot(111)

ax.add_collection(PatchCollection(ptchs,facecolor='0.75', edgecolor='w', linewidths=0))
ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.5))
ax.axis('auto'); ax.axis('off')
plt.show()
# Fonte: http://www.phme.it/wilt/2017/05/06/dynamic-mapping-with-shapefiles-and-python/

# Dados

### Dados de seguro rural

link = 'https://raw.githubusercontent.com/walefmachado/spreg_rural_insurance/main/dados/'
dados_br = pd.read_csv(link+'/dados_06_19.csv')

### Dados para espacial 
cod = pd.read_csv(link+'/codigos-mun.csv')
br = geopandas.read_file(link+'/br.json')
br = br.rename(columns={'CD_GEOCMU': 'mun'})
br.mun = br.mun.astype(int)
br2 = br.drop('NM_MUNICIP', axis=1)

cod_dados = cod.merge(br2, how='left')
cod_dados = geopandas.GeoDataFrame(cod_dados) # Ate aqui junta geometry com todos os códigos 

dados_br = cod_dados.merge(dados_br, on='mun', how='left')
dados_br = dados_br.fillna(0)
dados_br = dados_br.drop(['rm', 'nome_mun_y', 'nome_meso_y', 'sinistralidade_media'], axis = 1)
dados_br.rename(columns = {'nome_mun_x':'nome_mun', 'nome_meso_x':'nome_meso' }, inplace = True)

dados_br = dados_br.rename({'apolices_contratadas':'ap_contrat',
                            'total_segurado_mil':'t_segurado',
                            'soma_premio_total_mil':'soma_premio',
                            'total_subvencao_mil':'t_subvencao',
                            'valor_indenizacoes_pagas_mil':'inde_pagas',
                            'taxa_media':'tx_media',
                            'apolices_indenizadas':'ap_indeniz'}, axis=1)
#variaveis = ['apolices_contratadas', 'total_segurado_mil', 'soma_premio_total_mil', 'total_subvencao_mil', 'valor_indenizacoes_pagas_mil', 'taxa_media', 'apolices_indenizadas']
variaveis = ['ap_contrat', 't_segurado', 'soma_premio', 't_subvencao', 'inde_pagas', 'tx_media', 'ap_indeniz']
geometry = ['mun', 'geometry']

dados_br.drop(index=dados_br[dados_br['mun'] == 2605459].index, inplace=True) # retira F. Noronha e Ilhabela
dados_br.drop(index=dados_br[dados_br['mun'] == 3520400].index, inplace=True)

dados_19 = dados_br[dados_br['ano']==2019][variaveis]
dados_19_geo = dados_br[dados_br['ano']==2019][geometry]
dados_geo = dados_br[dados_br['ano']==2019][geometry]

anos = dados_br.ano.unique()

from matplotlib import rcParams
rcParams['axes.titlepad'] = -30  # posição do titulo "Ano"
f, axs = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(20, 20)) # 
axs = axs.flatten()
cbar_ax = f.add_axes([.65, .14, .15, .02])
for i, ano in enumerate(anos): 
    cols = dados_br[dados_br['ano']==ano][variaveis].corr().index
    cm = dados_br[dados_br['ano']==ano][cols].corr()
    mask = np.zeros_like(cm)
    mask[np.triu_indices_from(mask)] = True
    ax = axs[i]
    ax.set_title(ano, fontsize=15)
    sns.heatmap(cm,  cbar=i == 0, annot=True, cmap = 'Greys', ax=ax, cbar_ax=None if i else cbar_ax, cbar_kws={"orientation": "horizontal"}, mask=mask)   # 
axs[10].xaxis.set_tick_params(labelbottom=True, rotation=90)
axs[11].xaxis.set_tick_params(labelbottom=True, rotation=90)
axs[14].set_axis_off()
axs[15].set_axis_off()
plt.subplots_adjust(top=0.8, wspace=0.01, hspace=0.01)
plt.show();

## Padronizando os dados

dados_anos_p = []
for i, ano in enumerate(anos):
    cols = dados_br[dados_br['ano']==ano][variaveis]
    dadosp = StandardScaler().fit_transform(cols)
    dadosp = pd.DataFrame(dadosp, columns=cols.columns)
    dadosp['ano'] = ano
    dados_anos_p.append(dadosp)
dadosp = dados_anos_p[0]
for i in range(len(dados_anos_p)-1):
    dadosp = pd.concat([dadosp, dados_anos_p[i+1]], axis=0) # Concatenando os dados

## Encontrando os Componentes Principais

pcamodel = PCA()

cps  =['cp1', 'cp2', 'cp3', 'cp4', 'cp5', 'cp6', 'cp7']

pca_f = []
exp_variance = []
var_ratio = []
comp_corr = []
comp_cor = []
for i, ano in enumerate(anos):
    pca = pcamodel.fit_transform(dadosp[dadosp['ano'] == ano].drop('ano', axis=1))

    exp_variance.append(np.cumsum(np.round(pcamodel.explained_variance_ratio_, decimals=4)*100))
    var_ratio.append(pcamodel.explained_variance_ratio_)
    comp_corr.append(pcamodel.components_.T)
    comp_cor.append(pcamodel.components_)


    pca = pd.DataFrame(pca, columns=[cps])
    pca['ano'] = ano
    pca_f.append(pca)
    
pca = pca_f[0]
for i in range(len(pca_f)-1):
    pca = pd.concat((pca, pca_f[i+1]), axis=0, ignore_index=True) # Concatenando os dados
pca.columns = pca.columns.map(''.join)

#resultado = pd.DataFrame({'cp1':pca[:, 0], 'cp2':pca[:, 1]}, index=x.index)

pcamodel.explained_variance_

pcamodel.explained_variance_ratio_

# variância explicada acumulada
np.cumsum(np.round(pcamodel.explained_variance_ratio_, decimals=4)*100)

np.round(pcamodel.explained_variance_ratio_,2)

## Análise gráfica dos componentes

### Gráfico da variância acumulada

# variâncias acumuladas
tot = sum(pcamodel.explained_variance_)
var_exp = [(i / tot)*100 for i in sorted(pcamodel.explained_variance_, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
# número de CPs gerados (k = p)
cp =['CP'+str(i+1) for i in range(len(pcamodel.components_))]
# dataframe com as variâncias acumuladas para criar o scree plot
df = pd.DataFrame({'cp': cp, 'var_exp': var_exp, 'cum_var_exp': cum_var_exp})
# scree plot com %
fig, ax = plt.subplots(figsize=(9,5))
df.plot.bar('cp', 'var_exp', color='gray', ax=ax, legend=False)
df.plot.line('cp', 'cum_var_exp', color='black', ax=ax)
plt.legend(labels=['variância explicada acumulada'],
           loc='center right',)
plt.ylabel('% variância explicada')
plt.xlabel('');

### Distribuição do primeiro componente

from matplotlib import rcParams
rcParams['axes.titlepad'] = 3  # posição do titulo "Ano"
f, axs = plt.subplots(nrows=4, ncols=4, figsize=(20, 15)) # 
axs = axs.flatten()
for i, ano in enumerate(anos): 
    pca_ano = pca[pca['ano'] == ano]
    axs[i].hist(pca_ano.iloc[:,0], density=True, bins=50)
    ax = axs[i]
    ax.set_title(ano, fontsize=15)
    ax.set_xlabel('CP1')
axs[14].set_axis_off()
axs[15].set_axis_off()
plt.subplots_adjust(top=0.8, wspace=0.25, hspace=0.4)
plt.show();

### Gráfico da razão da variância explicada

var_ratio_df = pd.DataFrame(var_ratio, index=anos).T

round(var_ratio_df,4).T#.to_latex()

sns.set_palette('muted')

f, ax = plt.subplots(figsize=(13,5)) # 
plt.plot(var_ratio_df, linewidth=2.5)
plt.xlabel('Número de componentes', fontsize = 15)
plt.ylabel('Variância explicada acumulada', fontsize = 15)

xvals = np.linspace(1,7, 7)
ax.set_xticks(np.arange(len(range(0,7))))
ax.set_xticklabels(xvals, fontsize = 12)
ax.set_xticklabels(["{:,.0f}".format(x) for x in xvals], fontsize=12)

yvals = np.linspace(0,0.8, 7)
ax.set_yticks(yvals)
ax.set_yticklabels(["{:.2f}".format(y) for y in yvals], fontsize=12)
ax.yaxis.grid()

label_06 = "2006"

plt.annotate(label_06, # this is the text
             (0,0.5530), # these are the coordinates to position the label
             textcoords="offset points", # how to position the text
             xytext=(0,-15), # distance from text to points (x,y)
             ha='center', # horizontal alignment can be left, right or center
             bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="gray", lw=1))
label_07 = "2007"

plt.annotate(label_07, # this is the text
             (0,0.6514), # these are the coordinates to position the label
             textcoords="offset points", # how to position the text
             xytext=(0,-20), # distance from text to points (x,y)
             ha='center', # horizontal alignment can be left, right or center
             bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="gray", lw=1)) 


plt.show();

### *Scree plot*

plt.figure(figsize=(9,5))
plt.plot(pcamodel.explained_variance_)
plt.xlabel('Número de componentes')
plt.ylabel('Variância explicada acumulada')
plt.grid()
plt.show()

### Diagrama de dispersão entre PC1 e PC2

from matplotlib import rcParams
rcParams['axes.titlepad'] = 3  # posição do titulo "Ano"
f, axs = plt.subplots(nrows=4, ncols=4, figsize=(20, 15)) # 
axs = axs.flatten()
for i, ano in enumerate(anos): 
    pca_ano = pca[pca['ano'] == ano]
    axs[i].scatter(pca_ano.cp1, pca_ano.cp2, edgecolors='black', alpha=0.8, linewidths=.7, s=40, c='gray');
    #axs[i].grid()
    ax = axs[i]
    ax.set_title(ano, fontsize=15)
    ax.set_xlabel('CP1')
    ax.set_ylabel('CP2')

axs[14].set_axis_off()
axs[15].set_axis_off()
plt.subplots_adjust(top=0.8, wspace=0.25, hspace=0.4)
plt.show();

plt.figure(figsize=(9,5))
plt.scatter(pca[:, 0], pca[:, 1], edgecolors='black', linewidths=.7);
plt.xlabel('CP1')
plt.ylabel('CP2')
plt.grid();

### Efeito das variaveis em cada componente

rcParams['axes.titlepad'] = -30  # posição do titulo "Ano"
f, axs = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(20, 20)) # 
axs = axs.flatten()
cbar_ax = f.add_axes([.65, .14, .15, .02])
for i, ano in enumerate(anos): 
    ax = axs[i]
    ax.set_title(ano, fontsize=15)
    sns.heatmap(comp_corr[i],  
                cbar=i == 0, 
                annot=True, 
                cmap = 'Greys', 
                ax=ax, 
                mask=mask, 
                cbar_ax=None if i else cbar_ax, cbar_kws={"orientation": "horizontal"}, 
                xticklabels=[ "CP"+str(X) for X in range(1,pcamodel.n_components_+1)])
    ax.set_yticklabels(variaveis, rotation=0)
axs[10].xaxis.set_tick_params(labelbottom=True)
axs[11].xaxis.set_tick_params(labelbottom=True)
axs[14].set_axis_off()
axs[15].set_axis_off()
plt.subplots_adjust(top=0.8, wspace=0.01, hspace=0.01)
plt.show();

plt.figure(figsize=(14,8))
ax = sns.heatmap(comp_corr[5], #pcamodel.components_.T,
                 annot=True,
                 cmap='viridis',
                 yticklabels=list(x.columns),
                 xticklabels=[ "PCA"+str(X) for X in range(1,pcamodel.n_components_+1)])#,
                 #cbar_kws={"orientation": "horizontal"})
#ax.set_aspect("equal")

### ACP *Biplot*

fig, ax = plt.subplots(figsize=(10,6))
myplot(pca[:,0:2],np.transpose(pcamodel.components_[0:2, :]),list(x.columns))
plt.show()

from matplotlib import rcParams
rcParams['axes.titlepad'] = 3  # posição do titulo "Ano"
f, axs = plt.subplots(nrows=4, ncols=4, figsize=(20, 15)) # 
axs = axs.flatten()

for i, ano in enumerate(anos):
    ax = axs[i]
    ax.set_title(ano, fontsize=15)
    score = pca[pca['ano']==ano].iloc[:,0:2].values
    coeff = comp_cor[i][0:2].T
    labels = list(dados_19.columns)
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    axs[i].scatter(xs * scalex, ys * scaley, edgecolors='black', alpha=0.8, linewidths=.7, s=30, c='gray')
    for j in range(n):
        axs[i].arrow(0, 0, coeff[j,0], coeff[j,1],color = 'r',alpha = 0.8)
        if labels is None:
            axs[i].text(coeff[j,0]* 1.1, coeff[j,1] * 1.1, "Var"+str(i+1), color = 'green', ha = 'center', va = 'center')
        else:
            axs[i].text(coeff[j,0]* 1.1, coeff[j,1] * 1.1, labels[j], color = 'g', ha = 'center', va = 'center')
    ax.set_xlabel('CP1')
    ax.set_ylabel('CP2')
    #ax.set(xlim=(None, 0.75), ylim=(None, 0.7))

axs[14].set_axis_off()
axs[15].set_axis_off()
plt.subplots_adjust(top=0.95, wspace=0.25, hspace=0.4)
plt.show();

## Análise Espacial com CP

dados_geo = dados_br[dados_br['ano']==2019][geometry]
dados_geo.reset_index(inplace=True)
dados_geo.drop('index', axis=1, inplace=True)
dados_geo.set_index(dados_geo.index, inplace=True)
dados_geo.reset_index(level=0, inplace=True)
#dados_geo

pca_final = []
for ano in anos:
    pca_anos = pca[pca['ano']==ano]
    pca_anos.reset_index(inplace=True)
    pca_anos = pca_anos.drop('index', axis=1)
    pca_anos.set_index(pca_anos.index, inplace=True)
    pca_anos.reset_index(level=0, inplace=True)
    pca_anos = pca_anos.merge(dados_geo)
    pca_final.append(pca_anos)

pca_y = pca_final[0]
for i in range(len(pca_final)-1):
    pca_y = pd.concat((pca_y, pca_final[i+1]), axis=0, ignore_index=True) # Concatenando os dados

pca_y.drop('index', axis=1, inplace=True)

dados_cp = geopandas.GeoDataFrame(pca_y)
dados_cp

## Mapas Temáticos - CP1

referencia = jenkspy.jenks_breaks( dados_cp[dados_cp['ano']==2019]['cp1'][dados_cp[dados_cp['ano']==2019]['cp1'] != 0], nb_class=4)   #Fisher Jenks a partir dos dados de 2019
referencia[4] =  dados_cp['cp1'].max()

len(referencia), referencia

labels = [str(round(referencia[0])) + ' - 0',
          '0 - '+ str(round(referencia[1])),
          str(round(referencia[1])) + ' - '+ str(round(referencia[2])),
          str(round(referencia[2])) + ' - '+ str(round(referencia[3])), 
          str(round(referencia[3])) + ' - '+ str(round(referencia[4]))]     #rótulos da legenda
len(labels), labels

classif = mc.UserDefined(dados_cp['cp1'].values,referencia)
len(classif.counts), classif

cl = [labels[i] for i in classif.yb]
dados_cp = dados_cp.assign(cl = cl)
dados_cp.cl = pd.Categorical(dados_cp.cl,ordered=True,categories=labels)

color_list = ["lightgrey", "darkgrey", "gray", "dimgrey", "black"]
colors_map = colors.LinearSegmentedColormap.from_list("", color_list)

f, axs = plt.subplots(nrows=4, ncols=4, figsize=(30, 30)) # 
axs = axs.flatten()
for i, ano in enumerate(anos):
    ax = axs[i]
    dados_cp[dados_cp['ano']==ano].plot(column='cl', ax=ax, legend=False, categorical=True, cmap=colors_map); # plot
    ax.set_axis_off()
    ax.set_title(ano, fontsize=30)
    ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.5))   # adiciona a divisão estadual
lines = [Line2D([0], [0], color=c, marker='o', markersize=30, linestyle='') for c in color_list] # linewidth=20, linestyle='dotted'
axs[13].legend(lines, labels, loc='botton left', bbox_to_anchor=(1.85, 0.75), frameon=False, prop={'size': 30})
axs[14].set_axis_off()
axs[15].set_axis_off()
axs[15].imshow(img)
plt.subplots_adjust(left=7, bottom=7, right=8, top=8, wspace=0.001, hspace=0.005)
plt.show();

f, axs = plt.subplots(nrows=4, ncols=4, figsize=(30, 30)) # 
axs = axs.flatten()
for i, ano in enumerate(anos):
    ax = axs[i]
    dados_cp[dados_cp['ano']==ano].plot(column='cp1',  ax=ax, legend=False, k=5, linewidth=0, scheme='quantiles', cmap=colors_map); 
    ax.set_axis_off()
    ax.set_title(ano, fontsize=30)
    ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.5))   # adiciona a divisão estadual
lines = [Line2D([0], [0], color=c, marker='o', markersize=30, linestyle='') for c in color_list] # linewidth=20, linestyle='dotted'
#axs[13].legend(lines, labels, loc='botton left', bbox_to_anchor=(1.85, 0.75), frameon=False, prop={'size': 30})
axs[14].set_axis_off()
axs[15].set_axis_off()
axs[15].imshow(img)
plt.subplots_adjust(left=7, bottom=7, right=8, top=8, wspace=0.001, hspace=0.005)
plt.show();

f, axs = plt.subplots(nrows=4, ncols=4, figsize=(30, 30)) # 
axs = axs.flatten()
for i, ano in enumerate(anos):
    ax = axs[i]
    dados_cp[dados_cp['ano']==ano].plot(column='cp1',  ax=ax, legend=False, k=5, linewidth=0, scheme='equal_interval', cmap=colors_map); 
    ax.set_axis_off()
    ax.set_title(ano, fontsize=30)
    ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.5))   # adiciona a divisão estadual
lines = [Line2D([0], [0], color=c, marker='o', markersize=30, linestyle='') for c in color_list] # linewidth=20, linestyle='dotted'
#axs[13].legend(lines, labels, loc='botton left', bbox_to_anchor=(1.85, 0.75), frameon=False, prop={'size': 30})
axs[14].set_axis_off()
axs[15].set_axis_off()
axs[15].imshow(img)
plt.subplots_adjust(left=7, bottom=7, right=8, top=8, wspace=0.001, hspace=0.005)
plt.show();

f, axs = plt.subplots(nrows=4, ncols=4, figsize=(30, 30)) # 
axs = axs.flatten()
for i, ano in enumerate(anos):
    ax = axs[i]
    dados_cp[dados_cp['ano']==ano].plot(column='cp1',  ax=ax, legend=False, k=5, linewidth=0, scheme='natural_breaks', cmap=colors_map); 
    ax.set_axis_off()
    ax.set_title(ano, fontsize=30)
    ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.5))   # adiciona a divisão estadual
lines = [Line2D([0], [0], color=c, marker='o', markersize=30, linestyle='') for c in color_list] # linewidth=20, linestyle='dotted'
#axs[13].legend(lines, labels, loc='botton left', bbox_to_anchor=(1.85, 0.75), frameon=False, prop={'size': 30})
axs[14].set_axis_off()
axs[15].set_axis_off()
axs[15].imshow(img)
plt.subplots_adjust(left=7, bottom=7, right=8, top=8, wspace=0.001, hspace=0.005)
plt.show();

def max_p(values, k):
   
    from mapclassify import MaxP
    binning = MaxP(values, k=k)
    return binning.yb

dados_cp[dados_cp['ano']==2019]['cp1'].values

#dados_cp_19 = dados_cp[dados_cp['ano']==2019]
max_p(dados_cp[dados_cp['ano']==2019]['cp1'].values, k=2)
#dados_cp_19.head()

f, axs = plt.subplots(nrows=4, ncols=4, figsize=(30, 30)) # 
axs = axs.flatten()
for i, ano in enumerate(anos):
    ax = axs[i]
    dados_cp[dados_cp['ano']==ano].plot(column='cp1',  ax=ax, legend=False, k=5, linewidth=0, scheme='natural_breaks', cmap=colors_map); 
    ax.set_axis_off()
    ax.set_title(ano, fontsize=30)
    ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.5))   # adiciona a divisão estadual
lines = [Line2D([0], [0], color=c, marker='o', markersize=30, linestyle='') for c in color_list] # linewidth=20, linestyle='dotted'
#axs[13].legend(lines, labels, loc='botton left', bbox_to_anchor=(1.85, 0.75), frameon=False, prop={'size': 30})
axs[14].set_axis_off()
axs[15].set_axis_off()
axs[15].imshow(img)
plt.subplots_adjust(left=7, bottom=7, right=8, top=8, wspace=0.001, hspace=0.005)
plt.show();

# demora um pouco
y = dados_cp[dados_cp['ano']==2019]['cp1'].values
w = Queen.from_dataframe(dados_cp[dados_cp['ano']==2019])
w.transform = 'r'

## Tabela I de Moran

mi_results = []
table_I = pd.DataFrame(dados_br.ano.unique(), columns=['anos'])

for cp in cps:
    mi_results = [Moran(dados_cp[dados_cp['ano']==ano][cp].values, w) for ano in anos]
    table_p = pd.DataFrame([(round(res.I, 3)) for ano, res in zip(anos, mi_results)], columns=[cp])
    table_I = pd.concat([table_I.reset_index(drop=True), table_p], axis=1)
#table_I.to_latex()
table_I

f, ax = plt.subplots(figsize=(13,5)) # 
plt.plot(np.arange(len(table_I['anos'])), table_I.cp1, markersize=7, color='#4c72b0', linewidth=4) # label='CP1',

ax.set_xticks(np.arange(len(table_I['anos'])))
ax.set_xticklabels(anos, fontsize = 12)
ax.set_ylabel('I de Moran', fontsize = 15)
ax.set_xlabel('Anos', fontsize = 15)

yvals = np.linspace(0,0.8, 7)
ax.set_yticks(yvals)
ax.set_yticklabels(["{:.2f}".format(y) for y in yvals], fontsize=12)
ax.yaxis.grid()

## Lisa cluster:  CP1

lisa_cluster(moran_loc_br, dados_cp, p=0.05, figsize = (9,9));

# cria a legenda LISA
labels = ['AA', 'BA', 'BB', 'AB', 'não significativo']
color_list = ["#d7191c", "#abd9e9", "#2c7bb6", "#fdae61", "lightgrey"]
hmap = colors.ListedColormap("", color_list)
lines = [Line2D([0], [0], color=c, marker='o', markersize=30, linestyle='') for c in color_list] # linewidth=20, linestyle='dotted'

# cria a legenda LISA para o EXEMPLO
labels = ['AA', 'BA', 'BB', 'AB', 'não significativo']
color_list = ["#d7191c", "#abd9e9", "#2c7bb6", "#fdae61", "lightgrey"]
hmap = colors.ListedColormap("", color_list)
lines = [Line2D([0], [0], color=c, marker='o', markersize=15, linestyle='') for c in color_list] # linewidth=20, linestyle='dotted'

y = dados_cp[dados_cp['ano'] == 2019]['cp1'].values
moran_loc_br = Moran_Local(y, w)
f, ax = plt.subplots(figsize=(16,16)) # 
lisa_cluster(moran_loc_br, dados_cp[dados_cp['ano'] == 2019], p=0.05, ax=ax, figsize = (10,10))
ax.legend(lines, labels, loc='botton left', bbox_to_anchor=(1.08, 1), frameon=False,  prop={'size': 20})
plt.figimage(img2, 750, 25, zorder=1)
plt.show();

f, axs = plt.subplots(nrows=4, ncols=4, figsize=(30, 30)) # 
anos = dados_br.ano.unique()
axs = axs.flatten()
for i, ano in enumerate(anos):
    ax = axs[i]
    y = dados_cp[dados_cp['ano'] == 2019]['cp1'].values
    moran_loc_br = Moran_Local(y, w)
    lisa_cluster(moran_loc_br, dados_cp[dados_cp['ano']==ano], p=0.05, ax=ax, figsize = (10,10))
    ax.set_axis_off()
    ax.set_title(ano, fontsize=30)
    ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.5))   # adiciona a divisão estadual
axs[13].legend(lines, labels, loc='botton left', bbox_to_anchor=(2, 0.75), frameon=False,  prop={'size': 25})
axs[14].set_axis_off()
axs[15].set_axis_off()
axs[15].imshow(img)
plt.subplots_adjust(left=7, bottom=7, right=8, top=8, wspace=0.001, hspace=0.005)
plt.show();