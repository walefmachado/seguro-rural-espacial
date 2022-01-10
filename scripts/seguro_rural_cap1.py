#\begin{verbatim}

# Análise Exploratória de Dados Espaciais e Gráficos Descritivos

## Instalando bibliotecas 
#!pip install geopandas==0.8.1
#!pip install --upgrade pyshp
#!pip install shapely==1.7.0
#!pip install --upgrade descartes
#!pip install mapclassify==2.3.0 libpysal==4.3.0 splot==1.1.3
#!pip install jenkspy
#!pip install pyshp

## Importando as bibliotecas
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression # Para simular os dados do Diagrama de dispersão de Moran
import scipy.stats as stats
import jenkspy

# para a análise de dados espaciais
import geopandas
import splot
import shapefile
import mapclassify as mc
from libpysal.weights import Queen
from libpysal import weights
from esda import Moran, Moran_Local, G_Local
from splot.esda import plot_moran, moran_scatterplot, lisa_cluster, plot_local_autocorrelation

# para gráficos
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches, colors
from matplotlib.lines import Line2D
from matplotlib.collections import EventCollection
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection  
import matplotlib.image as mpimg
import matplotlib.ticker as mtick
from matplotlib import ticker
import seaborn as sns
sns.set_palette('muted')
pd.options.display.float_format = '{:.5f}'.format

import warnings
warnings.filterwarnings("ignore")

# Descomente para clonar o repositório com os dados utilisados
#!git clone -l -s git://github.com/walefmachado/seguro_rural_espacial.git dados
#%cd dados
#!ls

# Funções

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

def anual_g(v_i, v_f, a_i, a_f):
    return round((v_f/v_i) ** (1/(a_f - a_i) - 1 ) * 100,2)

# Dados

### Dados de seguro rural
link = 'https://raw.githubusercontent.com/walefmachado/seguro_rural_espacial/main/dados/'
dados_br = pd.read_csv(link+'/dados_06_19.csv')

### Dados do Atlas do Seguro rural
atlas_seg = pd.read_csv('https://raw.githubusercontent.com/walefmachado/spatial_cluster/master/data/atlas_seguro_rural_anuais.csv',sep=';', decimal=',')

### Dados Culturas
dados_cult = pd.read_csv('https://raw.githubusercontent.com/walefmachado/spatial_cluster/master/data/dados_cult_f.csv')

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
dados_br = dados_br.drop(['rm', 'nome_mun_y', 'nome_meso_y'], axis = 1)
dados_br.rename(columns = {'nome_mun_x':'nome_mun', 'nome_meso_x':'nome_meso' }, inplace = True)

# Rosa dos ventos para os mapas
img=mpimg.imread('https://github.com/walefmachado/seguro_rural_espacial/blob/main/figuras/rosa_dos_ventos_3.png?raw=true')

# Divisão em Estados
sf = shapefile.Reader('/content/dados/dados/estados_2010.shp')
shapes = sf.shapes()
Nshp = len(shapes)

ptchs   = []
for nshp in range(Nshp):
    pts     = np.array(shapes[nshp].points)
    prt     = shapes[nshp].parts
    par     = list(prt) + [pts.shape[0]]

    for pij in range(len(prt)):
       ptchs.append(Polygon(pts[par[pij]:par[pij+1]]))

sf_uf = shapefile.Reader('/content/dados/dados/estados_2010.shp')
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


variaveis = ['apolices_contratadas', 'total_segurado_mil', 'soma_premio_total_mil',
             'total_subvencao_mil', 'valor_indenizacoes_pagas_mil',
             'sinistralidade_media', 'taxa_media', 'apolices_indenizadas'] # 
anos = dados_br.ano.unique()

dados_19 = dados_br[dados_br['ano']==2019]

color_list = ["lightgrey", "darkgrey", "gray", "dimgrey", "black"]
colors_map = colors.LinearSegmentedColormap.from_list("", color_list)

# Cria os intervalos Fsher-Jenks para os mapas (para verificação apenas)
for variavel in variaveis:
    referencia = jenkspy.jenks_breaks(dados_19[variavel][dados_19[variavel] != 0], nb_class=4)   #Fisher Jenks a partir dos dados de 2019
    referencia[0] = 0 
    referencia[4] = dados_br[variavel].max()
    if variavel == 'taxa_media':
        labels = ['0','0 - '+str(round(referencia[1],1)),
                  str(round(referencia[1],1)) + ' - '+ str(round(referencia[2],1)), 
                str(round(referencia[2],1)) + ' - '+ str(round(referencia[3],1)), 
                str(round(referencia[3],1)) + ' - '+ str(round(referencia[4],1))]     #rótulos da legenda
        classif = mc.UserDefined(dados_br[variavel].values,referencia)
        cl = [labels[i] for i in classif.yb]
        dados_br = dados_br.assign(cl = cl)
        dados_br.cl = pd.Categorical(dados_br.cl,ordered=True,categories=labels)
    else:
        labels = ['0','1 - '+str(round(referencia[1])),
                str(round(referencia[1])) + ' - '+ str(round(referencia[2])), 
                str(round(referencia[2])) + ' - '+ str(round(referencia[3])), 
                str(round(referencia[3])) + ' - '+ str(round(referencia[4]))]     #rótulos da legenda
        classif = mc.UserDefined(dados_br[variavel].values,referencia)
        cl = [labels[i] for i in classif.yb]
        dados_br = dados_br.assign(cl = cl)
        dados_br.cl = pd.Categorical(dados_br.cl,ordered=True,categories=labels)

    print(referencia, labels)

# Mapa de todas as variáveis e todos os anos (demora bastante!)
for variavel in variaveis:
    referencia = jenkspy.jenks_breaks(dados_19[variavel][dados_19[variavel] != 0], nb_class=4)   #Fisher Jenks a partir dos dados de 2019
    referencia[0] = 0 
    referencia[4] = dados_br[variavel].max()
    if variavel == 'taxa_media':
        labels = ['0','0 - '+str(round(referencia[1],1)),
                  str(round(referencia[1],1)) + ' - '+ str(round(referencia[2],1)), 
                str(round(referencia[2],1)) + ' - '+ str(round(referencia[3],1)), 
                str(round(referencia[3],1)) + ' - '+ str(round(referencia[4],1))]     #rótulos da legenda
        classif = mc.UserDefined(dados_br[variavel].values,referencia)
        cl = [labels[i] for i in classif.yb]
        dados_br = dados_br.assign(cl = cl)
        dados_br.cl = pd.Categorical(dados_br.cl,ordered=True,categories=labels)
    else:
        labels = ['0','1 - '+str(round(referencia[1])),
                str(round(referencia[1])) + ' - '+ str(round(referencia[2])), 
                str(round(referencia[2])) + ' - '+ str(round(referencia[3])), 
                str(round(referencia[3])) + ' - '+ str(round(referencia[4]))]     #rótulos da legenda
        classif = mc.UserDefined(dados_br[variavel].values,referencia)
        cl = [labels[i] for i in classif.yb]
        dados_br = dados_br.assign(cl = cl)
        dados_br.cl = pd.Categorical(dados_br.cl,ordered=True,categories=labels)
    
    f, axs = plt.subplots(nrows=4, ncols=4, figsize=(30, 30)) 
    anos = dados_br.ano.unique()
    axs = axs.flatten()
    for i, ano in enumerate(anos):
        ax = axs[i]
        dados_br[dados_br['ano']==ano].plot(column='cl', ax=ax, legend=False, categorical=True, cmap=colors_map); # plot
        ax.set_axis_off()
        ax.set_title(ano, fontsize=30)
        ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.5))   # adiciona a divisão estadual
    lines = [Line2D([0], [0], color=c, marker='o', markersize=30, linestyle='') for c in color_list] # linewidth=20, linestyle='dotted'
    axs[13].legend(lines, labels, loc='botton left', bbox_to_anchor=(1.85, 0.75), frameon=False, prop={'size': 30})
    axs[14].set_axis_off()
    axs[15].set_axis_off()
    axs[15].imshow(img)
    plt.subplots_adjust(left=7, bottom=7, right=8, top=8, wspace=0.001, hspace=0.005)
    print(variavel)
    plt.show();

# Verificando quantos municípios em cada têm zero apólices contratadas
apol_0 = []
for i in anos:
    apol_0.append(len(dados_br[dados_br['ano']==i].query('apolices_contratadas ==0')))
print(apol_0)

# Retirando as ilhas
dados_lisa = dados_br
dados_lisa.drop(index=dados_lisa[dados_lisa['mun'] == 2605459].index, inplace=True) # retira F. Noronha e Ilhabela
dados_lisa.drop(index=dados_lisa[dados_lisa['mun'] == 3520400].index, inplace=True) 
dados_19.drop(index=dados_19[dados_19['mun'] == 2605459].index, inplace=True) # retira F. Noronha e Ilhabela
dados_19.drop(index=dados_19[dados_19['mun'] == 3520400].index, inplace=True)

dados_br[dados_br['mun']==2605459] # conferindo se foram detiradas de todos os anos 
dados_br[dados_br['mun']==3520400]

# Criando a matriz de pesos espaciais (demora um pouco)
y = dados_19['apolices_contratadas'].values
w = Queen.from_dataframe(dados_19)
w.transform = 'r'

## Tabela I de Moran

mi_results = []
table_I = pd.DataFrame(dados_br.ano.unique(), columns=['anos'])

for variavel in variaveis:
    mi_results = [Moran(dados_lisa[dados_lisa['ano']==ano][variavel].values, w) for ano in anos]
    table_p = pd.DataFrame([(round(res.I, 3)) for ano, res in zip(anos, mi_results)], columns=[variavel])
    table_I = pd.concat([table_I.reset_index(drop=True), table_p], axis=1)
table_I

#table_I.to_latex() # Para exportar para o LaTeX

# I de Moran para as variáveis de seguro rural no Brasil por municípios entre 2006 e 2019
f, ax = plt.subplots(figsize=(13,5)) # 
plt.plot(np.arange(len(table_I['anos'])), table_I.apolices_contratadas, label='Apolices contratadas',markersize=7, color='#4c72b0', linewidth=4)
plt.plot(np.arange(len(table_I['anos'])), table_I.total_segurado_mil, label='Total segurado',markersize=7, color='#dd8452', linewidth=4)
plt.plot(np.arange(len(table_I['anos'])), table_I.soma_premio_total_mil, label='Soma prêmio total',markersize=7, color='#55a868', linewidth=4)
plt.plot(np.arange(len(table_I['anos'])), table_I.total_subvencao_mil, label='Total subvenção',markersize=7, color='#c44e52', linewidth=4)
plt.plot(np.arange(len(table_I['anos'])), table_I.valor_indenizacoes_pagas_mil, '.-', label='Valor indenizações pagas',markersize=7, color='#8172b3', linewidth=4)

plt.plot(np.arange(len(table_I['anos'])), table_I.taxa_media, label='Taxa média',markersize=7, color='#da8bc3', linewidth=4)
plt.plot(np.arange(len(table_I['anos'])), table_I.apolices_indenizadas, label='Apolices indenizadas ',markersize=7, color='#8c8c8c', linewidth=4)

ax.set_xticks(np.arange(len(table_I['anos'])))
ax.set_xticklabels(anos, fontsize = 12)
ax.set_ylabel('I de Moran', fontsize = 15)
ax.set_xlabel('Anos', fontsize = 15)

yvals = np.linspace(0,0.8, 7)
ax.set_yticks(yvals)
ax.set_yticklabels(["{:.2f}".format(y) for y in yvals], fontsize=12)
ax.yaxis.grid()

plt.legend(bbox_to_anchor=(0.89, -0.15), ncol=4)

## Mapas LISA

# cria a legenda LISA
labels = ['AA', 'BA', 'BB', 'AB', 'não significativo']
color_list = ["#d7191c", "#abd9e9", "#2c7bb6", "#fdae61", "lightgrey"]
hmap = colors.ListedColormap("", color_list)
lines = [Line2D([0], [0], color=c, marker='o', markersize=30, linestyle='') for c in color_list] # linewidth=20, linestyle='dotted'

# Mapas LISA para todas as variáveis e anos (demora bastante!)
for variavel in variaveis:
    f, axs = plt.subplots(nrows=4, ncols=4, figsize=(30, 30)) # 
    anos = dados_br.ano.unique()
    axs = axs.flatten()
    for i, ano in enumerate(anos):
        ax = axs[i]
        y = dados_lisa[dados_lisa['ano'] == ano][variavel].values
        moran_loc_br = Moran_Local(y, w)
        lisa_cluster(moran_loc_br, dados_lisa[dados_lisa['ano']==ano], p=0.05, ax=ax, figsize = (10,10))
        ax.set_axis_off()
        ax.set_title(ano, fontsize=30)
        ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.5))   # adiciona a divisão estadual
    axs[13].legend(lines, labels, loc='botton left', bbox_to_anchor=(2, 0.75), frameon=False,  prop={'size': 25})
    axs[14].set_axis_off()
    axs[15].set_axis_off()
    axs[15].imshow(img)
    plt.subplots_adjust(left=7, bottom=7, right=8, top=8, wspace=0.001, hspace=0.005)
    print(variavel)
    plt.show();

## Identificando grupos Lisa

# Ano 2006
lisa_06 = dados_lisa[dados_lisa['ano'] == 2006]
y = dados_lisa[dados_lisa['ano'] == 2006]['apolices_indenizadas'].values
moran_loc_br = Moran_Local(y, w)
sig = moran_loc_br.p_sim < 0.05  # identificar significativos
posicoes = np.where(sig)    # municípios significativos
lisa_06['quad'] = moran_loc_br.q   # criar uma coluna nos dados identificando o quadrante

# Significativos
AA_06 = lisa_06.iloc[posicoes].query('quad == 1') # Alto-Alto
BA_06 = lisa_06.iloc[posicoes].query('quad == 2')  # Baixo-alto
BB_06 = lisa_06.iloc[posicoes].query('quad == 3') # Baixo-baixo
AB_06 = lisa_06.iloc[posicoes].query('quad == 4') # Alto-baixo

# Significativos ou não 
#AA_06 = lisa_06.query('quad == 1') # Alto-Alto
#BA_06 = lisa_06.query('quad == 2')  # Baixo-alto
#BB_06 = lisa_06.query('quad == 3') # Baixo-baixo
#AB_06 = lisa_06.query('quad == 4') # Alto-baixo

# Ano 2019
lisa_19 = dados_lisa[dados_lisa['ano'] == 2019]
y = dados_lisa[dados_lisa['ano'] == 2019]['apolices_indenizadas'].values
moran_loc_br = Moran_Local(y, w)
sig = moran_loc_br.p_sim < 0.05  # identificar significativos   
posicoes = np.where(sig)    # municípios significativos
lisa_19['quad'] = moran_loc_br.q   # criar uma coluna nos dados identificando o quadrante

# Significativos
AA_19 = lisa_19.iloc[posicoes].query('quad == 1') # Alto-Alto
BA_19 = lisa_19.iloc[posicoes].query('quad == 2')  # Baixo-alto
BB_19 = lisa_19.iloc[posicoes].query('quad == 3') # Baixo-baixo
AB_19 = lisa_19.iloc[posicoes].query('quad == 4') # Alto-baixo

# se queremos os municípios identificados como HL queremos o quadrante 4
# 1 - HH  
# 2 - LH  
# 3 - LL  
# 4 - HL

# Regiões AA 2006
(AA_06.groupby(['nome_regiao', 'nome_uf'])['quad'].count() / AA_06.groupby(['nome_uf', 'nome_regiao'])['quad'].count().sum()) * 100
# Regiões AA 2019
(AA_19.groupby(['nome_regiao', 'nome_uf'])['quad'].count() / AA_19.groupby(['nome_uf', 'nome_regiao'])['quad'].count().sum()) * 100

## Gráficos descritivos

### Apólices contratadas

f, ax = plt.subplots(figsize=(13,5)) # 
plt.plot(np.arange(len(atlas_seg.Ano)), atlas_seg.apolices, color='#4c72b0', linewidth=4, label= "Apólices contratadas")
plt.plot(np.arange(len(atlas_seg.Ano)), atlas_seg.produtores, color='#dd8452', linewidth=4, label="Produtores")

yvals = np.linspace((atlas_seg.produtores).min(), (atlas_seg.apolices).max(), 7)
ax.set_yticks(yvals)
ax.set_yticklabels(["{:.0f}".format(y) for y in yvals], fontsize=12)

ax.set_xticks(np.arange(len(atlas_seg.Ano)))
ax.set_xticklabels(atlas_seg.Ano, fontsize = 12)
ax.set_ylabel('Número de produtores e apólices', fontsize = 15)
ax.set_xlabel('Anos', fontsize = 15)
ax.yaxis.grid()
plt.legend(bbox_to_anchor=(0.25, 0.95))

round(np.mean(atlas_seg.apolices/atlas_seg.produtores),2)

round(atlas_seg.apolices.max()/atlas_seg.apolices.min(),2)

anual_g(atlas_seg.apolices.min(), atlas_seg.apolices.max(), 2006, 2020)

round(atlas_seg.produtores.max()/atlas_seg.produtores.min(),2)

### Valores do prêmio e de subvenção ao prêmio de seguro rural

f, ax = plt.subplots(figsize=(13,5)) # 
plt.plot(np.arange(len(atlas_seg.Ano)), atlas_seg['tot_subvencao']/1000000, color='#4c72b0', linewidth=4, label="Subvenção total")
plt.plot(np.arange(len(atlas_seg.Ano)), atlas_seg['premio_prod']/1000000, color='#dd8452', linewidth=4, label="Prêmio produtor") 
plt.plot(np.arange(len(atlas_seg.Ano)), atlas_seg['premio_seg']/1000000, color='#55a868', linewidth=4, label="Prêmio seguradora")

ax.set_xticks(np.arange(len(atlas_seg.Ano)))
ax.set_xticklabels(atlas_seg.Ano, fontsize = 12)

yvals = np.linspace((atlas_seg['premio_seg']/1000000).min(), (atlas_seg['premio_seg']/1000000).max(), 7)
ax.set_yticks(yvals)
ax.set_yticklabels(["{:.0f}".format(y) for y in yvals], fontsize=12)

ax.set_ylabel('Valor em milhões de R\$', fontsize = 15)
ax.set_xlabel('Anos', fontsize = 15)
ax.yaxis.grid()
plt.legend( bbox_to_anchor=(0.25, 0.95),);


round(np.mean(atlas_seg.apolices/atlas_seg.produtores),2)
round(atlas_seg.apolices.max()/atlas_seg.apolices.min(),2)
anual_g(atlas_seg.apolices.min(), atlas_seg.apolices.max(), 2006, 2020)
round(atlas_seg.produtores.max()/atlas_seg.produtores.min(),2)

### Total segurado

f, ax = plt.subplots(figsize=(13,5)) # 
plt.plot(np.arange(len(atlas_seg.Ano)), atlas_seg['import_segurada']/1000000, color='#4c72b0', linewidth=4) 

ax.set_xticks(np.arange(len(atlas_seg.Ano)))
ax.set_xticklabels(atlas_seg.Ano, fontsize = 12)

yvals = np.linspace((atlas_seg['import_segurada']/1000000).min(), (atlas_seg['import_segurada']/1000000).max(), 7)
ax.set_yticks(yvals)
ax.set_yticklabels(["{:.0f}".format(y) for y in yvals], fontsize=12)
ax.set_ylabel('Total segurado (em milhões de R$)', fontsize = 15)
ax.set_xlabel('Anos', fontsize = 15)
ax.yaxis.grid()

import_seg = atlas_seg.import_segurada/1000000
round(import_seg.max() / import_seg[0],2)
round(import_seg[14]/import_seg[13],2)

### Taxa média de contratação de seguro rural

f, ax = plt.subplots(figsize=(13,5)) 
plt.plot(np.arange(len(atlas_seg.Ano)), atlas_seg.taxa_media, color='#4c72b0', linewidth=4) 

ax.set_xticks(np.arange(len(atlas_seg.Ano)))
ax.set_xticklabels(atlas_seg.Ano, fontsize = 12)

yvals = np.linspace((atlas_seg.taxa_media).min(), (atlas_seg.taxa_media).max(), 7)
ax.set_yticks(yvals)
ax.set_yticklabels(["{:,.1%}".format(y) for y in yvals], fontsize=12)

ax.set_ylabel('Taxa média', fontsize = 15)
ax.set_xlabel('Anos', fontsize = 15)
ax.yaxis.grid()


f, ax = plt.subplots(figsize=(13,5)) # 
plt.plot(np.arange(len(atlas_seg.Ano)), atlas_seg.area_segurada, color='#4c72b0', linewidth=4, label= "Área segurada em hectares") 

yvals = np.linspace((atlas_seg.area_segurada).min(), (atlas_seg.area_segurada).max(), 7)
ax.set_yticks(yvals)
ax.set_yticklabels(["{:.0f}".format(y) for y in yvals], fontsize=12)

ax.set_xticks(np.arange(len(atlas_seg.Ano)))
ax.set_xticklabels(atlas_seg.Ano, fontsize = 12)
ax.set_ylabel('Área segurada em hectares', fontsize = 15)
ax.set_xlabel('Anos', fontsize = 15)
ax.yaxis.grid()

df_soma_anos = dados_br.groupby('ano').sum()
df_soma_anos.reset_index(level = 0, inplace = True)


f, ax = plt.subplots(figsize=(13,5)) # 
plt.plot(np.arange(len(df_soma_anos['ano'])), df_soma_anos['apolices_indenizadas'], label='Apólices indenizadas', color='#4c72b0', linewidth=4)

yvals = np.linspace((df_soma_anos['apolices_indenizadas']).min(), (df_soma_anos['apolices_indenizadas']).max(), 7)
ax.set_yticks(yvals)
ax.set_yticklabels(["{:.0f}".format(y) for y in yvals], fontsize=12)

ax.set_xticks(np.arange(len(df_soma_anos['ano'])))
ax.set_xticklabels(anos, fontsize = 12)
ax.set_ylabel('Número de apólices indenizadas', fontsize = 15)
ax.set_xlabel('Anos', fontsize = 15)
ax.yaxis.grid();


f, ax = plt.subplots(figsize=(13,5)) # 
plt.plot(np.arange(len(df_soma_anos['ano'])), df_soma_anos['valor_indenizacoes_pagas_mil']/1000000, label='Valor indenizações pagas', color='#4c72b0', linewidth=4)

yvals = np.linspace((df_soma_anos['valor_indenizacoes_pagas_mil']/1000000).min(), (df_soma_anos['valor_indenizacoes_pagas_mil']/1000000).max(), 7)
ax.set_yticks(yvals)
ax.set_yticklabels(["{:.0f}".format(y) for y in yvals], fontsize=12)

ax.set_xticks(np.arange(len(df_soma_anos['ano'])))
ax.set_xticklabels(anos, fontsize = 12)
ax.set_ylabel('Valor das indenizações em milhões de R\$', fontsize = 12)
ax.set_xlabel('Anos', fontsize = 15)
ax.yaxis.grid();

# Culturas 

apolices_anos = df_soma_anos['apolices_contratadas']
soja = (dados_cult[dados_cult['cultura']=="Soja"]['apolices_contratadas'].values / apolices_anos.values) * 100
milho_1 = (dados_cult[dados_cult['cultura']=='Milho 1ª safra']['apolices_contratadas'].values / apolices_anos.values) * 100
milho_2 = (np.concatenate([[0], dados_cult[dados_cult['cultura']=='Milho 2ª safra']['apolices_contratadas'].values]) / apolices_anos.values) * 100
maca = (dados_cult[dados_cult['cultura']=='Maçã']['apolices_contratadas'].values / apolices_anos.values) * 100
uva = (dados_cult[dados_cult['cultura']=="Uva"]['apolices_contratadas'].values / apolices_anos.values) * 100
#algodao = (dados_cult[dados_cult['cultura']=='Algodão']['apolices_contratadas'].values / apolices_anos.values) * 100
cana = (dados_cult[dados_cult['cultura']=='Cana-de-açúcar']['apolices_contratadas'].values / apolices_anos.values) * 100
#floresta = (dados_cult[dados_cult['cultura']== 'Floresta']['apolices_contratadas'].values / apolices_anos.values) * 100
#pecuaria = (dados_cult[dados_cult['cultura']=='Pecuário']['apolices_contratadas'].values / apolices_anos.values) * 100
outros = np.array(soja + milho_1 + milho_2 + maca + uva + cana)/100
outros = (np.ones(14) - outros )* 100

milho_1.max(), milho_1.min(), milho_1.std(), milho_1.mean()

culturas_prc = pd.DataFrame(anos, columns=['anos'])
culturas_prc['soja'] = soja.tolist()
culturas_prc['milho_1'] = milho_1
culturas_prc['milho_2'] = milho_2
culturas_prc['maca'] = maca
culturas_prc['uva'] = uva
#culturas_prc['algodao'] = algodao
culturas_prc['cana'] =  cana
#culturas_prc['floresta'] = floresta
#culturas_prc['pecuaria'] = pecuaria
culturas_prc['outros'] = outros


f, ax = plt.subplots(figsize=(13,6)) 
pos = np.arange(len(anos))

plt.bar(pos, soja, label='Soja')
plt.bar(pos, milho_1, bottom = soja, label='Milho 1ª safra')
plt.bar(pos, milho_2, bottom = soja + milho_1, label='Milho 2ª safra')
plt.bar(pos, maca, bottom = soja + milho_1 + milho_2, label='Maçã')
plt.bar(pos, uva, bottom = soja + milho_1 + milho_2 + maca, label='Uva')
#plt.bar(pos, algodao, bottom = soja + milho_1 + milho_2 + maca + uva, label='Algodão')
plt.bar(pos, cana, bottom = soja + milho_1 + milho_2 + maca + uva, label='Cana-de-açúcar')
#plt.bar(pos, floresta, bottom = soja + milho_1 + milho_2 + maca + uva + algodao + cana, label='Floresta')
#plt.bar(pos, pecuaria, bottom = soja + milho_1 + milho_2 + maca + uva + algodao + cana + floresta, label='Pecuária')
plt.bar(pos, outros, bottom = soja + milho_1 + milho_2 + maca + uva + cana, label='Outros')
ax.set_xticks(pos)
ax.set_xticklabels(anos, fontsize = 12)
ax.set_ylabel('Percentual de apólices contratadas', fontsize = 15)
ax.set_xlabel('Anos', fontsize = 15)
fmt = '%.0f%%' 
yticks = mtick.FormatStrFormatter(fmt)
ax.yaxis.set_major_formatter(yticks)
#plt.legend(bbox_to_anchor=(0.23, 0.4))
plt.legend(bbox_to_anchor=(0.67, -0.13), ncol=4)
plt.show()

## Diagrama de dispersão de Moran - Simulação

x = np.random.randn(300)
y = 0.98 * x + np.random.randn(300)

nx = np.reshape(x, (-1, 1))
ny = np.reshape(y, (-1, 1))
model = LinearRegression()
model.fit(nx, ny)
fig, ax = plt.subplots(figsize=(12,6))

plt.ylim(-5,5)
plt.xlim(-4,4)
plt.axhline(0, color='black', linestyle = 'dashed')
plt.axvline(0, color='black', linestyle = 'dashed')

ax.annotate('AA', xy=(2.5, 2.5), size=25,
    xytext=(-6, 3), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="0.8"))
ax.annotate('AB', xy=(2.5, -2.5), size=25,
    xytext=(-6, 3), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="0.8"))      
ax.annotate('BA', xy=(-2.5, 2.5), size=25,
    xytext=(-6, 3), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="0.8"))
ax.annotate('BB', xy=(-2.5, -2.5), size=25,
    xytext=(-6, 3), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="0.8"))

ax.set_xlabel('Variável ($z$)', fontsize = 15)
ax.set_ylabel('Defasagem espacial da variável ($Wz$)', fontsize = 15)
ax.plot(nx, model.predict(nx), color='red',linewidth=2)
ax.scatter(nx, ny, color ='black');

#\end{verbatim}