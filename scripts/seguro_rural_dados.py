#\begin{verbatim}

# Dados de Seguro Rural

## Bibliotecas

import pandas as pd
import numpy as np

## Funções

# função para deixar todos os nomes de municípios iguais 
# (sem acento, sem espaço)
def simplificar_nomes(df, variavel):
  df[variavel] = (df[variavel].str.strip()
                 .str.lower()
                 .str.replace(' ', '_')
                 .str.replace('\'', '')
                 .str.replace('-','_')
                 .str.replace('á', 'a')
                 .str.replace('é', 'e')
                 .str.replace('í', 'i')
                 .str.replace('ó', 'o')
                 .str.replace('ú', 'u')
                 .str.replace('â', 'a')
                 .str.replace('ê', 'e')
                 .str.replace('ô', 'o')
                 .str.replace('í', 'i')
                 .str.replace('ã','a')
                 .str.replace('õ','o')
                 .str.replace('ç','c')
                 .str.replace('à', 'a')
                 .str.replace('ü', 'u'))

## Dados

#link do repositório no GitHub
link = 'https://raw.githubusercontent.com/walefmachado/spatial_cluster/master/data' 
cod = pd.read_csv(link+'/codigos-mun.csv')
cod_uf = pd.read_csv(link+'/cod_uf.csv', encoding='utf-8')
cod = cod.merge(cod_uf, on='cod_uf', how='left')
cod = cod[['mun', 'nome_mun', 'nome_meso', 'uf']]
dados = pd.read_csv(link+'/dados_06_19_b.csv', encoding='utf-8')

# simplificar os nomes dos municípios nos dois dataframes
simplificar_nomes(cod, 'nome_mun') 
simplificar_nomes(dados, 'nome_mun')

# Verifica quais são as localidades em divergência
list(set(dados['nome_mun']) - set(cod['nome_mun'])) 

# Número de localidades divergentes
len(list(set(dados['nome_mun']) - set(cod['nome_mun']))) 

replacers = {'pereirinhas': 'desterro_de_entre_rios',
             'torrinhas': 'pinheiro_machado',
             'cerro_do_roque': 'butia',
             'pinheiro_marcado': 'carazinho',
             'ourilandia': 'barbosa_ferraz',
             'nova_brasilia': 'araruna',
             'trentin': 'jaboticaba',
             'vale_veneto': 'sao_joao_do_polesine',
             'sao_pedro_tobias': 'dionisio_cerqueira',
             'vale_formoso': 'novo_horizonte',
             'cavalheiro': 'ipameri',
             'rio_do_salto': 'cascavel',
             'cascata': 'pelotas',
             'gramadinho': 'itapetininga',
             'cavajureta': 'sao_vicente_do_sul',
             'conceicao_de_monte_alegre': 'paraguacu_paulista',
             'taquarichim': 'jaguari',
             'tres_placas': 'tapejara',
             'passinhos': 'osorio',
             'sede_alvorada': 'cascavel',
             'juliania': 'herculandia',
             'basilio': 'herval',
             'esperanca_do_norte': 'alvorada_do_sul',
             'itaboa': 'ribeirao_branco',
             'alto_alvorada': 'orizona',
             'jafa': 'garca',
             'itahum': 'dourados',
             'arapuan': 'arapua',
             'rio_do_mato': 'francisco_beltrao',
             'nossa_senhora_da_candelaria': 'bandeirantes',
             'sarandira': 'juiz_de_fora',
             'plano_alto': 'uruguaiana',
             'rocas_novas': 'caete',
             'frei_timoteo': 'jataizinho',
             'vidigal': 'cianorte',
             'colonia_esperanca': 'arapongas',
             'cristo_rei': 'capanema',
             'graccho_cardoso': 'gracho_cardoso',
             'marajo': 'nova_aurora',
             'valerio': 'planalto',
             'sao_camilo': 'palotina',
             'triolandia': 'ribeirao_do_pinhal',
             'ibare': 'lavras_do_sul',
             'bocaja': 'douradina',
             'itao': 'itaqui',
             'vila_gandhi': 'primeiro_de_maio',
             'honoropolis': 'campina_verde',
             'colonia_sao_joao': 'cruz_alta',
             'otavio_rocha': 'flores_da_cunha',
             'engenheiro_maia': 'itabera',
             'hidraulica': 'pelotas',
             'palmitopolis': 'nova_aurora',
             'biritiba_ussu': 'mogi_das_cruzes',
             'trevo_do_jose_rosario': 'leopoldo_de_bulhoes',
             'poema': 'nova_tebas',
             'espigao': 'regente_feijo',
             'irere': 'londrina',
             'bairro_limoeiro': 'londrina',
             'capao_grande': 'muitos_capoes',
             'santo_antonio_do_paranapanema': 'candido_mota',
             'rincao_do_cristovao_pereira': 'mostardas',
             'sao_luiz_do_oeste': 'toledo',
             'colonia_socorro': 'guarapuava',
             'colonia_vitoria': 'guarapuava',
             'vale_dos_vinhedos': 'bento_goncalves',
             'barro_vermelho': 'gravatai',
             'santo_antonio_do_rio_verde': 'catalao',
             'nova_lourdes': 'sao_joao',
             'santa_cruz_do_timbo': 'porto_uniao',
             'santauta': 'camaqua',
             'guaragi': 'ponta_grossa',
             'caetano_mendes': 'tibagi',
             'torquato_severo': 'dom_pedrito',
             'pontoes': 'afonso_claudio',
             'ivailandia': 'engenheiro_beltrao',
             'capao_seco': 'sidrolandia',
             'aparecida_do_oeste': 'tuneiras_do_oeste',
             'engenheiro_luiz_englert': 'sertao',
             'bela_vista_do_sul': 'mafra',
             'gamadinho': 'cascavel',
             'juruce': 'jardinopolis',
             'novo_diamantino': 'diamantino',
             'paranagi': 'sertaneja',
             'roda_velha': 'sao_desiderio',
             'ferreira': 'cachoeira_do_sul',
             'mirante_do_piquiri': 'alto_piquiri',
             'bentopolis': 'nossa_senhora_das_gracas',
             'passo_liso': 'laranjeiras_do_sul',
             'saica': 'cacequi',
             'indapolis': 'dourados',
             'taim': 'rio_grande',
             'correia_de_freitas': 'apucarana',
             'iolopolis': 'sao_jorge_doeste',
             'lajeado_cerne': 'santo_angelo',
             'itapocu': 'araquari',
             'cordilheira': 'cachoeira_do_sul',
             'colonia_castrolanda': 'castro',
             'capivarita': 'rio_pardo',
             'nossa_senhora_de_caravaggio': 'nova_veneza',
             'candia': 'pontal',
             'herveira': 'campina_da_lagoa',
             'santa_flora': 'santa_maria',
             'santa_lucia_do_piai': 'caxias_do_sul',
             'figueira_do_oeste': 'engenheiro_beltrao',
             'porto_mendes': 'marechal_candido_rondon',
             'holambra_ii': 'paranapanema',
             'perola_independente': 'maripa',
             'alto_santa_fe': 'nova_santa_rosa',
             'calcilandia': 'goias',
             'comandai': 'santo_angelo',
             'cerrito_alegre': 'pelotas',
             'fazenda_jangada': 'cascavel',
             'guajuvira': 'araucaria',
             'guacu': 'dourados',
             'agua_azul': 'lapa',
             'barra_grande': 'itapejara_doeste',
             'colonia_samambaia': 'guarapuava',
             'encantado_doeste': 'assis_chateaubriand',
             'pulinopolis': 'mandaguacu',
             'piquirivai': 'campo_mourao',
             'bateias': 'campo_largo',
             'novo_sarandi': 'toledo',
             'nova_sardenha': 'farroupilha',
             'mariental': 'lapa',
             'arroio_do_so': 'santa_maria',
             'warta': 'londrina',
             'vila_marques': 'aral_moreira',
             'nova_concordia': 'francisco_beltrao',
             'piriquitos': 'ponta_grossa',
             'vila_ipiranga': 'toledo',
             'doutor_oliveira_castro': 'guaira',
             'colonia_z_3': 'pelotas',
             'novo_sobradinho': 'toledo',
             'vila_diniz': 'cruzmaltina',
             'catanduvas_do_sul': 'contenda',
             'jansen': 'farroupilha',
             'polo_petroquimico_de_triunfo': 'triunfo',
             'carumbe': 'itapora',
             'sao_joao_doeste': 'cascavel',
             'albardao': 'rio_pardo',
             'banhado_do_colegio': 'camaqua',
             'bacupari': 'palmares_do_sul',
             'souza_ramos': 'getulio_vargas',
             'montese': 'itapora',
             'rincao_dos_mendes': 'santo_angelo',
             'batatuba': 'piracaia',
             'vila_oliva': 'caxias_do_sul',
             'rincao_del_rei': 'rio_pardo',
             'turiba_do_sul': 'itabera',
             'aracaiba': 'apiai',
             'macaia': 'bom_sucesso',
             'criuva': 'caxias_do_sul',
             'bom_sucesso_de_patos': 'patos_de_minas',
             'amparo_da_serra': 'amparo_do_serra',
             'arace': 'domingos_martins',
             'agisse': 'rancharia',
             'perico': 'sao_joaquim',
             'capao_da_porteira': 'viamao',
             'gardenia': 'rancharia',
             'couto_de_magalhaes': 'couto_magalhaes',
             'fazenda_souza': 'caxias_do_sul',
             'cardeal': 'elias_fausto',
             'sitio_grande': 'sao_desiderio',
             'sao_valerio_da_natividade': 'sao_valerio',
             'capao_da_lagoa': 'guarapuava',
             'santa_rita_do_ibitipoca': 'santa_rita_de_ibitipoca',
             'ipiabas': 'barra_do_pirai',
             'charco': 'castro',
             'dez_de_maio': 'toledo',
             'jubai': 'conquista',
             'aquidaban': 'marialva',
             'bragantina': 'assis_chateaubriand',
             'guaravera': 'londrina',
             'ibitiuva': 'pitangueiras',
             'indios': 'lages',
             'tres_bicos': 'candido_de_abreu',
             'santa_rita_do_oeste': 'santa_rita_doeste',
             'vila_nova': 'toledo',
             'lerroville': 'londrina',
             'panema': 'santa_mariana',
             'santo_antonio_dos_campos': 'divinopolis',
             'vila_seca': 'caxias_do_sul',
             'frutal_do_campo': 'candido_mota',
             'colonia_centenario': 'cascavel',
             'garibaldina': 'garibaldi',
             'macucos': 'getulina',
             'siqueira_belo': 'barracao',
             'sao_joao_dos_mellos': 'julio_de_castilhos',
             'crispim_jaques': 'teofilo_otoni',
             'cruzaltina': 'douradina',
             'bacuriti': 'cafelandia',
             'coronel_prestes': 'encruzilhada_do_sul',
             'sao_bento_baixo': 'nova_veneza',
             'atiacu': 'sarandi',
             'pacheca': 'camaqua',
             'bom_jardim_do_sul': 'ivai',
             'pampeiro': 'santana_do_livramento',
             'quinzopolis': 'santa_mariana',
             'bonfim_paulista': 'ribeirao_preto',
             'uvaia': 'ponta_grossa',
             'itororo_do_paranapanema': 'pirapozinho',
             'yolanda': 'ubirata',
             'nova_altamira': 'faxinal',
             'rincao_do_meio': 'sao_tome',
             'rincao_doce': 'santo_antonio_do_planalto',
             'paiquere': 'londrina',
             'ouroana': 'rio_verde',
             'abapa': 'castro',
             'calogeras': 'arapoti',
             'alexandrita': 'iturama',
             'campo_do_bugre': 'rio_bonito_do_iguacu',
             'barao_de_lucena': 'nova_esperanca',
             'porteira_preta': 'fenix',
             'sao_jose_da_reserva': 'santa_cruz_do_sul',
             'sao_luiz_do_puruna': 'balsa_nova',
             'dorizon': 'mallet',
             'bernardelli': 'rondon',
             'lagoa_do_bauzinho': 'rio_verde',
             'nova_cardoso': 'itajobi',
             'bela_vista_do_piquiri': 'campina_da_lagoa',
             'nossa_senhora_da_aparecida': 'rolandia',
             'monte_alverne': 'santa_cruz_do_sul',
             'azevedo_sodre': 'sao_gabriel',
             'sao_joaquim_do_pontal': 'itambaraca',
             'bourbonia': 'barbosa_ferraz',
             'guaraciaba_doeste': 'tupi_paulista',
             'colonia_melissa': 'cascavel',
             'selva': 'londrina',
             'cabeceira_do_apa': 'ponta_pora',
             'cachoeira_de_emas': 'pirassununga',
             'barragem_do_itu': 'macambara',
             'taquaruna': 'londrina',
             'sede_progresso': 'francisco_beltrao',
             'porto_vilma': 'deodapolis',
             'irui': 'rio_pardo',
             'novo_tres_passos': 'marechal_candido_rondon',
             'tereza_breda': 'barbosa_ferraz',
             'guaipora': 'cafezal_do_sul',
             'vida_nova': 'sapopema',
             'fazenda_colorado': 'fortaleza_dos_valos',
             'conselheiro_zacarias': 'santo_antonio_da_platina',
             'palmira': 'sao_joao_do_triunfo',
             'capivara': 'erval_seco',
             'nova_patria': 'presidente_bernardes',
             'espinilho_grande': 'tupancireta',
             'aguas_claras': 'viamao',
             'santa_rita_da_floresta': 'cantagalo',
             'papagaios_novos': 'palmeira',
             'passo_real': 'salto_do_jacui',
             'triangulo': 'engenheiro_beltrao',
             'capela_sao_paulo': 'sao_luiz_gonzaga',
             'nova_casa_verde': 'nova_andradina',
             'curral_alto': 'santa_vitoria_do_palmar',
             'ipomeia': 'rio_das_antas',
             'tapinas': 'itapolis',
             'vassoural': 'ibaiti',
             'cachoeira_do_espirito_santo': 'ribeirao_claro',
             'picadinha': 'dourados',
             'palmeirinha': 'guarapuava',
             'passo_do_verde': 'sao_sepe',
             'alto_do_amparo': 'tibagi',
             'jacarandira': 'resende_costa',
             'guarapua': 'dois_corregos',
             'pedra_branca_de_itarare': 'itarare',
             'nova_milano': 'farroupilha',
             'rio_toldo': 'getulio_vargas',
             'juvinopolis': 'cascavel',
             'granja_getulio_vargas': 'palmares_do_sul',
             'porto_santana': 'porto_barreiro',
             'coxilha_rica': 'itapejara_doeste',
             'vila_freire': 'cerrito',
             'bonfim_da_feira': 'feira_de_santana',
             'mimoso': 'barao_de_melgaco',
             'felpudo': 'campo_largo',
             'pulador': 'passo_fundo',
             'bau': 'candiota',
             'tupinamba': 'astorga',
             'jazidas': 'formigueiro',
             'mariza': 'sao_pedro_do_ivai',
             'patrocinio_de_caratinga': 'caratinga',
             'campo_seco': 'rosario_do_sul',
             'sao_miguel_do_cambui': 'marialva',
             'pau_dalho_do_sul': 'assai',
             'conciolandia': 'perola_doeste',
             'margarida': 'marechal_candido_rondon',
             'concordia_do_oeste': 'toledo',
             'daltro_filho': 'imigrante',
             'dario_lassance': 'candiota',
             'amandina': 'ivinhema',
             'guarapuavinha': 'inacio_martins',
             'vila_nova_de_florenca': 'sao_jeronimo_da_serra',
             'geremia_lunardelli': 'nova_cantu',
             'riverlandia': 'rio_verde',
             'sao_jose_das_laranjeiras': 'maracai',
             'fluviopolis': 'sao_mateus_do_sul',
             'cerrito_do_ouro': 'sao_sepe',
             'juciara': 'kalore',
             'colonia_jordaozinho': 'guarapuava',
             'barra_dourada': 'neves_paulista',
             'sao_clemente': 'santa_helena',
             'santa_cruz_da_estrela': 'santa_rita_do_passa_quatro',
             'pedro_lustosa': 'reserva_do_iguacu',
             'guaipava': 'paraguacu',
             'bexiga': 'rio_pardo',
             'boca_do_monte': 'santa_maria',
             'tupi_silveira': 'candiota',
             'iguipora': 'marechal_candido_rondon',
             'passo_das_pedras': 'capao_do_leao',
             'capane': 'cachoeira_do_sul',
             'jangada_do_sul': 'general_carneiro',
             'malu': 'terra_boa',
             'esquina_piratini': 'bossoroca',
             'caraja': 'jesuitas',
             'santo_antonio_do_palmital': 'rio_bom',
             'joao_arregui': 'uruguaiana',
             'clemente_argolo': 'lagoa_vermelha',
             'alto_da_uniao': 'ijui',
             'fernao_dias': 'munhoz_de_melo',
             'taquara_verde': 'cacador',
             'apiaba': 'imbituva',
             'ponte_vermelha': 'sao_gabriel_do_oeste',
             'floropolis': 'paranacity',
             'apiai_mirim': 'capao_bonito',
             'jacipora': 'dracena',
             'silveira': 'sao_jose_dos_ausentes',
             'piquiri': 'nova_esperanca_do_sul',
             'prudencio_e_moraes': 'general_salgado',
             'ibiporanga': 'tanabi',
             'saltinho_do_oeste': 'alto_piquiri',
             'guardinha': 'sao_sebastiao_do_paraiso',
             'bom_retiro_da_esperanca': 'angatuba',
             'ouro_verde_do_piquiri': 'corbelia',
             'campina_de_fora': 'ribeirao_branco',
             'santa_esmeralda': 'santa_cruz_de_monte_castelo',
             'cambaratiba': 'ibitinga',
             'romeopolis': 'arapua',
             'clarinia': 'santa_cruz_do_rio_pardo',
             'tres_vendas': 'cachoeira_do_sul',
             'candeia': 'maripa',
             'joca_tavares': 'bage',
             'veredas': 'joao_pinheiro',
             'lageado_de_aracaiba': 'apiai',
             'guarizinho': 'itapeva',
             'santa_fe_do_pirapo': 'marialva',
             'santa_izabel': 'sao_joaquim',
             'vila_formosa': 'dourados',
             'rincao_comprido': 'augusto_pestana',
             'espigao_do_oeste': 'espigao_doeste',
             'tres_capoes': 'guarapuava',
             'bandeirantes_doeste': 'formosa_do_oeste',
             'jurupema': 'taquaritinga',
             'covo': 'mangueirinha',
             'parana_doeste': 'moreira_sales',
             'sao_francisco_de_imbau': 'congonhinhas',
             'jaracatia': 'goioere',
             'barreiro': 'ijui',
             'colonia_cachoeira': 'guarapuava',
             'arvore_grande': 'paranaiba',
             'vila_vargas': 'dourados',
             'ubauna': 'sao_joao_do_ivai',
             'ibiaci': 'primeiro_de_maio',
             'aparecida_de_minas': 'frutal',
             'retiro_grande': 'campo_largo',
             'pedras': 'lapa',
             'campo_lindo': 'campo_limpo_de_goias',
             'lagoa_branca': 'casa_branca',
             'herval_grande': 'laranjeiras_do_sul', 
             'boa_vista_de_santa_maria': 'unai'}

# faz a correção nos municípios
dados['nome_mun'] = dados['nome_mun'].replace(replacers) 

# Verifica se ainda há divergências
list(set(dados['nome_mun']) - set(cod['nome_mun'])) 

# Número de localidades divergentes
len(list(set(dados['nome_mun']) - set(cod['nome_mun']))) 

# Faz a verificação de divergências por ano
anos = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 
        2013, 2014, 2015, 2016, 2017, 2018, 2019] 
for i in anos:
    print(i,':', len(list(set(dados[dados['ano'] == i]\
        ['nome_mun'])-set(cod['nome_mun']))))


## Por município
dados.ano = dados.ano.apply(str)
dados_anos = []
for i in anos:
    df_anos = dados.merge(cod, on=['nome_mun','uf'], how='right')
    df_anos = df_anos[df_anos['ano']==str(i)]
    df_anos = df_anos.groupby('mun').sum()
    df_anos.reset_index(level = 0, inplace = True)
    df_anos = df_anos.merge(cod, on='mun', how='right')
    df_anos['ano'] = str(i)
    df_anos.fillna(0,inplace=True)
    dados_anos.append(df_anos)
dadosf = dados_anos[0]
for i in range(len(dados_anos)-1):
    dadosf = pd.concat([dadosf, dados_anos[i+1]], axis=0)

# Salvando os dados
dadosf.to_csv('dados_06_19.csv', index=False) 


## Por seguradora
dados_seg = []
for i in anos:
    df_seg = dados.merge(cod, on=['nome_mun','uf'], how='right')
    df_seg = df_seg[df_seg['ano']==str(i)]
    df_seg = df_seg.groupby('seguradora').sum()
    df_seg.reset_index(level = 0, inplace = True)
    df_seg['ano'] = str(i)
    dados_seg.append(df_seg)
dados_seg_f = dados_seg[0]
for i in range(len(dados_seg)-1):
    dados_seg_f = pd.concat([dados_seg_f, dados_seg[i+1]], axis=0) 

# Salvando os dados
dados_seg_f.to_csv('dados_seg_f.csv', index=False) 


## Por cultura
dados_cult = []
for i in anos:
    df_cult = dados.merge(cod, on=['nome_mun','uf'], how='right')
    df_cult = df_cult[df_cult['ano']==str(i)]
    df_cult = df_cult.groupby('cultura').sum()
    df_cult.reset_index(level = 0, inplace = True)
    df_cult['ano'] = str(i)
    dados_cult.append(df_cult)
dados_cult_f = dados_cult[0]
for i in range(len(dados_cult)-1):
    dados_cult_f = pd.concat([dados_cult_f, dados_cult[i+1]], axis=0) 

# Salvando os dados
dados_cult_f.to_csv('dados_cult_f.csv', index=False) 

#\end{verbatim}