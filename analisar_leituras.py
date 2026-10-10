import numpy as np
import pandas as pd

from gerar_dataset import (
    ler_prata, preparar_leituras, calcular_vizinhos,
    COLUNAS_TRECHO, COLUNAS_SERIE, JANELA_VIZINHOS_MESES,
)

ARQUIVO_RESUMO = './relatorio_leituras_multiplas.txt'
ARQUIVO_DETALHE = './relatorio_leituras_multiplas.csv'
ARQUIVO_LEVANTAMENTOS = './relatorio_levantamentos.csv'
ARQUIVO_COPIAS = './relatorio_copias.csv'

# Unidade de análise: um trecho de 1 km, em um sentido, em um mês
COLUNAS_GRUPO = COLUNAS_SERIE + ['mes']

CAUSA_PEDACOS = 'pedaços menores que 1 km'
CAUSA_DATAS = 'datas diferentes no mês'
CAUSA_MESMA_DATA = 'mesma data'

FAIXAS_DESVIO = [0, 5, 10, 20, np.inf]
ROTULOS_FAIXAS_DESVIO = ['até 5', '5 a 10', '10 a 20', 'acima de 20']

# Um levantamento é o conjunto de leituras de uma rodovia, em um sentido, em uma data
COLUNAS_LEVANTAMENTO = ['uf', 'rodovia', 'sentido', 'data']
# Só são avaliados levantamentos com pelo menos este número de km comparáveis
# (km que têm leitura no mês anterior ou no seguinte)
MINIMO_KM_LEVANTAMENTO = 10
# Levantamento suspeito: diferença mediana para os vizinhos de pelo menos este valor
LIMITE_DIFERENCA_LEVANTAMENTO = 25
# Km discrepante dentro de um levantamento: diferença para os vizinhos acima deste valor
LIMITE_KM_DISCREPANTE = 30

# Cópias: levantamento cujas leituras repetem as de outro levantamento, nos
# mesmos km e no mesmo dia, só com o mês adiantado em um destes deslocamentos
DESLOCAMENTOS_COPIA_MESES = [1, 2, 3]
COLUNAS_LOCAL = ['uf', 'rodovia', 'sentido', 'km inicial', 'km final']
MINIMO_KM_COPIA = 10
LIMITE_IDENTICOS_COPIA = 0.95


def formatar_leituras(valores):
    # Separadas por " | " porque a vírgula é o separador decimal
    return ' | '.join(f"{valor:g}".replace('.', ',') for valor in sorted(valores))


def descrever_multiplas(df):
    """Retorna uma linha por trecho-sentido-mês com mais de uma leitura."""
    df['km_leitura'] = (
        df[['km inicial', 'km final']].min(axis=1).astype(str) + '-'
        + df[['km inicial', 'km final']].max(axis=1).astype(str)
    )

    leituras = df.groupby(COLUNAS_GRUPO, dropna=False)['icm'].transform('size')
    multiplas = df[leituras > 1].sort_values('data')

    # ICM médio de cada data, para comparar a primeira e a última avaliação do mês
    multiplas['icm_na_data'] = (
        multiplas.groupby(COLUNAS_GRUPO + ['data'], dropna=False)['icm'].transform('mean')
    )

    detalhe = multiplas.groupby(COLUNAS_GRUPO, dropna=False).agg(
        leituras=('icm', 'size'),
        valores_icm=('icm', 'nunique'),
        icm_min=('icm', 'min'),
        icm_max=('icm', 'max'),
        icm_media=('icm', 'mean'),
        icm_mediana=('icm', 'median'),
        # Variância e desvio padrão amostrais (divisor n − 1)
        icm_variancia=('icm', 'var'),
        icm_desvio_padrao=('icm', 'std'),
        icm_lidos=('icm', formatar_leituras),
        datas=('data', 'nunique'),
        icm_data_mais_antiga=('icm_na_data', 'first'),
        icm_data_mais_recente=('icm_na_data', 'last'),
        trechos_lidos=('km_leitura', 'nunique'),
        arquivos=('arquivo', 'nunique'),
        lista_arquivos=('arquivo', lambda arquivos: ', '.join(sorted(set(arquivos)))),
        sentido_deduzido=('sentido_deduzido', 'any'),
    ).reset_index()

    detalhe[['icm_variancia', 'icm_desvio_padrao']] = detalhe[['icm_variancia', 'icm_desvio_padrao']].round(4)
    # Só há avaliação mais antiga e mais recente quando as leituras têm datas diferentes
    datas_iguais = detalhe['datas'] == 1
    detalhe.loc[datas_iguais, ['icm_data_mais_antiga', 'icm_data_mais_recente']] = np.nan
    detalhe['diferenca_icm'] = detalhe['icm_max'] - detalhe['icm_min']
    detalhe['valores'] = np.where(detalhe['valores_icm'] > 1, 'diferentes', 'iguais')
    detalhe['causa'] = np.select(
        [detalhe['trechos_lidos'] > 1, detalhe['datas'] > 1],
        [CAUSA_PEDACOS, CAUSA_DATAS],
        default=CAUSA_MESMA_DATA
    )
    detalhe['faixa_desvio'] = pd.cut(
        detalhe['icm_desvio_padrao'], FAIXAS_DESVIO, labels=ROTULOS_FAIXAS_DESVIO
    )

    return adicionar_vizinhos(df, detalhe)


def adicionar_vizinhos(df, detalhe):
    """Acrescenta a leitura anterior e a próxima da série do trecho e onde
    elas ficam em relação à menor (posição 0) e à maior (posição 1) leitura do mês."""
    mensal = calcular_vizinhos(
        df.groupby(COLUNAS_GRUPO, dropna=False)['icm'].mean().reset_index()
    ).drop(columns=['icm', 'icm_vizinhos'])
    mensal['mes'] = mensal['mes'].astype(str)
    vizinhos = ['anterior', 'proxima']

    detalhe['mes'] = detalhe['mes'].astype(str)
    detalhe = detalhe.merge(mensal, on=COLUNAS_GRUPO, how='left')

    amplitude = detalhe['diferenca_icm'].where(detalhe['diferenca_icm'] > 0)
    for nome in vizinhos:
        posicao = (detalhe[f'icm_{nome}'] - detalhe['icm_min']) / amplitude
        detalhe[f'posicao_{nome}'] = posicao.round(4)
        detalhe[f'{nome}_mais_perto_da'] = np.select(
            [posicao < 0.5, posicao > 0.5, posicao == 0.5],
            ['menor', 'maior', 'meio'],
            default=''
        )

    return detalhe


def avaliar_levantamentos(df):
    """Compara cada levantamento (rodovia, sentido e data) com as leituras dos
    meses vizinhos dos mesmos km. Um levantamento muito distante dos vizinhos
    em quase toda a sua extensão é suspeito de erro de registro."""
    mensal = calcular_vizinhos(
        df.groupby(COLUNAS_GRUPO, dropna=False)['icm'].mean().reset_index()
    )
    leituras = df.merge(mensal[COLUNAS_GRUPO + ['icm_vizinhos']], on=COLUNAS_GRUPO, how='left')
    leituras['diferenca'] = leituras['icm'] - leituras['icm_vizinhos']
    leituras['km_discrepante'] = (leituras['diferenca'].abs() > LIMITE_KM_DISCREPANTE).where(
        leituras['diferenca'].notna()
    )

    levantamentos = leituras.groupby(COLUNAS_LEVANTAMENTO).agg(
        km_avaliados=('icm', 'size'),
        km_comparaveis=('diferenca', 'count'),
        icm_medio=('icm', 'mean'),
        icm_vizinhos_medio=('icm_vizinhos', 'mean'),
        diferenca_mediana=('diferenca', 'median'),
        diferenca_abs_mediana=('diferenca', lambda diferencas: diferencas.abs().median()),
        km_discrepantes=('km_discrepante', 'sum'),
        icp_zero=('icp', lambda icp: (icp == 0).mean()),
        icm_distintos=('icm', 'nunique'),
        arquivos=('arquivo', lambda arquivos: ', '.join(sorted(set(arquivos)))),
    ).reset_index()

    levantamentos = levantamentos[levantamentos['km_comparaveis'] >= MINIMO_KM_LEVANTAMENTO].copy()
    levantamentos['km_discrepantes'] = levantamentos['km_discrepantes'].astype(int)
    levantamentos['proporcao_km_discrepantes'] = (
        levantamentos['km_discrepantes'] / levantamentos['km_comparaveis']
    )
    levantamentos['suspeito'] = (
        levantamentos['diferenca_mediana'].abs() >= LIMITE_DIFERENCA_LEVANTAMENTO
    )
    levantamentos['data'] = levantamentos['data'].dt.strftime('%Y-%m-%d')

    colunas_decimais = [
        'icm_medio', 'icm_vizinhos_medio', 'diferenca_mediana', 'diferenca_abs_mediana',
        'icp_zero', 'proporcao_km_discrepantes',
    ]
    levantamentos[colunas_decimais] = levantamentos[colunas_decimais].round(4)

    return levantamentos


def resumir_levantamentos(levantamentos):
    suspeitos = levantamentos[levantamentos['suspeito']]
    abaixo = suspeitos[suspeitos['diferenca_mediana'] < 0]
    acima = suspeitos[suspeitos['diferenca_mediana'] > 0]

    linhas = [
        "",
        "9) Levantamentos suspeitos",
        "   Levantamento: leituras de uma rodovia, em um sentido, em uma data. Cada km é comparado",
        "   com a média do mês anterior e do seguinte (até "
        f"{JANELA_VIZINHOS_MESES} meses) no mesmo km e sentido.",
        f"   Suspeito: diferença mediana para os vizinhos de {LIMITE_DIFERENCA_LEVANTAMENTO} pontos ou mais.",
        "",
        f"   Levantamentos avaliados (≥ {MINIMO_KM_LEVANTAMENTO} km comparáveis): {len(levantamentos)}",
        f"   Suspeitos: {len(suspeitos)} ({percentual(len(suspeitos), len(levantamentos))}), "
        f"somando {suspeitos['km_avaliados'].sum()} km avaliados",
        f"      abaixo dos vizinhos (ICM baixo demais): {len(abaixo)}",
        f"      acima dos vizinhos (ICM alto demais):   {len(acima)}",
        "",
        "   Distribuição da diferença mediana de todos os levantamentos avaliados:",
    ]
    percentis = levantamentos['diferenca_mediana'].quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    linhas.append("   " + "   ".join(
        f"p{int(quantil * 100)}: {valor:.1f}" for quantil, valor in percentis.items()
    ))

    colunas = [
        'uf', 'rodovia', 'sentido', 'data', 'km_avaliados', 'icm_medio', 'icm_vizinhos_medio',
        'diferenca_mediana', 'proporcao_km_discrepantes', 'icp_zero', 'icm_distintos',
    ]
    for titulo, tabela in [
        ("Mais abaixo dos vizinhos", abaixo.nsmallest(10, 'diferenca_mediana')),
        ("Mais acima dos vizinhos", acima.nlargest(10, 'diferenca_mediana')),
    ]:
        linhas += ["", f"   {titulo}:"]
        linhas += ["   " + linha for linha in tabela[colunas].round(2).to_string(index=False).splitlines()]

    linhas += [
        "",
        "   icp_zero: proporção de km com ICP = 0; icm_distintos: quantidade de valores de ICM diferentes.",
        f"   Lista completa, com todos os levantamentos avaliados, em {ARQUIVO_LEVANTAMENTOS}.",
    ]
    return linhas


def detectar_copias(df):
    """Procura levantamentos que repetem um levantamento anterior com a data
    adiantada em alguns meses (mesmos km, mesmo dia, ICC, ICP e ICM idênticos)."""
    leituras = df[COLUNAS_LOCAL + ['data', 'icc', 'icp', 'icm', 'arquivo']]
    resultados = []

    for meses in DESLOCAMENTOS_COPIA_MESES:
        adiantadas = leituras.assign(data=leituras['data'] + pd.DateOffset(months=meses))
        pares = leituras.merge(adiantadas, on=COLUNAS_LOCAL + ['data'], suffixes=('', '_original'))
        pares['identica'] = (
            (pares['icc'] == pares['icc_original'])
            & (pares['icp'] == pares['icp_original'])
            & (pares['icm'] == pares['icm_original'])
        )
        por_levantamento = pares.groupby(COLUNAS_LEVANTAMENTO).agg(
            km_pareados=('identica', 'size'),
            proporcao_identicos=('identica', 'mean'),
            icm_medio=('icm', 'mean'),
            icm_distintos=('icm', 'nunique'),
            arquivo=('arquivo', 'first'),
            arquivo_original=('arquivo_original', 'first'),
        ).reset_index()
        por_levantamento['meses_adiantados'] = meses
        por_levantamento['data_original'] = por_levantamento['data'] - pd.DateOffset(months=meses)
        resultados.append(por_levantamento)

    copias = pd.concat(resultados, ignore_index=True)
    copias = copias[
        (copias['km_pareados'] >= MINIMO_KM_COPIA)
        & (copias['proporcao_identicos'] >= LIMITE_IDENTICOS_COPIA)
    ].copy()

    for coluna in ['data', 'data_original']:
        copias[coluna] = copias[coluna].dt.strftime('%Y-%m-%d')
    copias[['proporcao_identicos', 'icm_medio']] = copias[['proporcao_identicos', 'icm_medio']].round(4)

    colunas = COLUNAS_LEVANTAMENTO + [
        'data_original', 'meses_adiantados', 'km_pareados', 'proporcao_identicos',
        'icm_medio', 'icm_distintos', 'arquivo', 'arquivo_original',
    ]
    return copias[colunas].sort_values(['arquivo', 'uf', 'rodovia', 'sentido', 'data'])


def resumir_copias(copias):
    linhas = [
        "",
        "10) Levantamentos copiados de meses anteriores",
        "   Cópia: levantamento cujos km repetem, no mesmo dia, as leituras de um levantamento",
        f"   {', '.join(map(str, DESLOCAMENTOS_COPIA_MESES))} mês(es) antes, com ICC, ICP e ICM idênticos em pelo menos "
        f"{LIMITE_IDENTICOS_COPIA:.0%} dos km",
        f"   (mínimo de {MINIMO_KM_COPIA} km pareados).",
        "",
        f"   Cópias encontradas: {len(copias)} levantamentos, {copias['km_pareados'].sum()} km",
    ]
    if copias.empty:
        return linhas

    por_arquivo = copias.groupby(['arquivo_original', 'arquivo', 'meses_adiantados']).agg(
        levantamentos=('km_pareados', 'size'),
        km=('km_pareados', 'sum'),
    ).sort_values('km', ascending=False)
    linhas += ["", "   Por arquivo:"]
    linhas += ["   " + linha for linha in por_arquivo.to_string().splitlines()]

    linhas += [
        "",
        "   Cópias com poucos valores distintos de ICM (até 3) podem ser rodovias que de fato",
        "   não mudaram; as demais dificilmente coincidem por acaso.",
        f"   Lista completa em {ARQUIVO_COPIAS}.",
    ]
    return linhas


def percentual(parte, total):
    return f"{parte / total:.1%}" if total else "-"


def resumir_vizinho(casos, nome, agrupador):
    """Resume onde a leitura vizinha (anterior ou próxima) cai em relação à menor
    e à maior leitura do mês, considerando só vizinhas dentro da janela."""
    casos = casos[casos[f'meses_ate_{nome}'] <= JANELA_VIZINHOS_MESES]
    posicao = casos[f'posicao_{nome}']
    grupos = casos[agrupador]

    proporcao = lambda condicao: condicao.groupby(grupos, observed=True).mean().map('{:.0%}'.format)
    return pd.DataFrame({
        'casos': posicao.groupby(grupos, observed=True).size(),
        'perto_da_menor': proporcao(posicao < 0.5),
        'perto_da_maior': proporcao(posicao > 0.5),
        'no_meio': proporcao(posicao == 0.5),
        'abaixo_da_menor': proporcao(posicao < 0),
        'acima_da_maior': proporcao(posicao > 1),
        'posicao_mediana': posicao.groupby(grupos, observed=True).median().round(2),
    })


def montar_resumo(df, detalhe, levantamentos, copias):
    total_grupos = df.groupby(COLUNAS_GRUPO, dropna=False).ngroups
    total_trechos = df.groupby(COLUNAS_TRECHO).ngroups
    trechos_afetados = detalhe.groupby(COLUNAS_TRECHO).ngroups
    multiplos = len(detalhe)
    diferentes = (detalhe['valores'] == 'diferentes').sum()

    linhas = [
        "RELATÓRIO DE LEITURAS MÚLTIPLAS DE ICM",
        "Unidade de análise: trecho de 1 km (uf, rodovia, km) em um sentido e um mês.",
        "",
        f"Leituras analisadas (sem repetições entre arquivos): {len(df)}",
        f"Leituras com sentido deduzido pela ordem dos kms:    {df['sentido_deduzido'].sum()}",
        f"Leituras ainda sem sentido:                          {df['sentido'].isna().sum()}",
        f"Trechos de 1 km:                                     {total_trechos}",
        f"Combinações trecho × sentido × mês:                  {total_grupos}",
        "",
        "1) Trecho-sentido-mês com mais de uma leitura no mesmo mês",
        f"   {multiplos} combinações ({percentual(multiplos, total_grupos)}), "
        f"em {trechos_afetados} trechos ({percentual(trechos_afetados, total_trechos)} dos trechos)",
        "",
        "2) Destas, quantas têm valores de ICM diferentes",
        f"   diferentes: {diferentes} ({percentual(diferentes, multiplos)})",
        f"   iguais:     {multiplos - diferentes} ({percentual(multiplos - diferentes, multiplos)})",
        "",
        "3) Causa × valores",
    ]

    tabela = pd.crosstab(detalhe['causa'], detalhe['valores'], margins=True, margins_name='total')
    linhas += ["   " + linha for linha in tabela.to_string().splitlines()]

    linhas += ["", "4) Diferença de ICM (máximo − mínimo) quando os valores diferem"]
    com_diferenca = detalhe[detalhe['valores'] == 'diferentes']
    estatisticas = com_diferenca.groupby('causa')['diferenca_icm'].describe(percentiles=[0.5, 0.9])
    estatisticas = estatisticas[['count', '50%', '90%', 'max']].rename(
        columns={'count': 'casos', '50%': 'mediana', '90%': 'p90', 'max': 'máximo'}
    )
    estatisticas['casos'] = estatisticas['casos'].astype(int)
    linhas += ["   " + linha for linha in estatisticas.round(2).to_string().splitlines()]

    linhas += [
        "",
        "5) Dispersão das leituras de um mesmo trecho-sentido-mês, quando os valores diferem",
        "   (variância e desvio padrão amostrais de cada caso, resumidos por causa)",
    ]
    dispersao = com_diferenca.assign(
        media_menos_mediana=(com_diferenca['icm_media'] - com_diferenca['icm_mediana']).abs()
    ).groupby('causa').agg(
        casos=('leituras', 'size'),
        leituras_media=('leituras', 'mean'),
        desvio_medio=('icm_desvio_padrao', 'mean'),
        desvio_mediano=('icm_desvio_padrao', 'median'),
        desvio_p90=('icm_desvio_padrao', lambda desvios: desvios.quantile(0.9)),
        variancia_media=('icm_variancia', 'mean'),
        variancia_mediana=('icm_variancia', 'median'),
        media_menos_mediana=('media_menos_mediana', 'mean'),
    )
    linhas += ["   " + linha for linha in dispersao.round(2).to_string().splitlines()]
    linhas += [
        "   media_menos_mediana: diferença absoluta média entre média e mediana do caso;",
        "   valores altos indicam uma leitura muito diferente das demais.",
    ]

    linhas += [
        "",
        "6) Leituras vizinhas: a série tende à menor ou à maior leitura do mês?",
        f"   Compara a leitura anterior e a próxima do mesmo trecho e sentido (até "
        f"{JANELA_VIZINHOS_MESES} meses de distância)",
        "   com a menor e a maior leitura do mês. Posição = (vizinha − menor) / (maior − menor):",
        "   0 = igual à menor, 1 = igual à maior. Só casos com valores diferentes.",
    ]
    for nome, titulo in [('proxima', 'Próxima leitura'), ('anterior', 'Leitura anterior')]:
        linhas += ["", f"   {titulo}, por faixa de desvio padrão do caso"]
        tabela = resumir_vizinho(com_diferenca, nome, 'faixa_desvio')
        linhas += ["   " + linha for linha in tabela.to_string().splitlines()]
    linhas += ["", "   Próxima leitura, por causa"]
    tabela = resumir_vizinho(com_diferenca, 'proxima', 'causa')
    linhas += ["   " + linha for linha in tabela.to_string().splitlines()]

    # Nas leituras em datas diferentes, a avaliação mais recente do mês é a
    # que melhor antecipa a próxima? (indicaria manutenção entre as datas)
    datas = com_diferenca[
        (com_diferenca['causa'] == CAUSA_DATAS)
        & (com_diferenca['meses_ate_proxima'] <= JANELA_VIZINHOS_MESES)
        & (com_diferenca['icm_data_mais_antiga'] != com_diferenca['icm_data_mais_recente'])
    ]
    distancia_recente = (datas['icm_proxima'] - datas['icm_data_mais_recente']).abs()
    distancia_antiga = (datas['icm_proxima'] - datas['icm_data_mais_antiga']).abs()
    recente_menor = datas['icm_data_mais_recente'] < datas['icm_data_mais_antiga']
    linhas += [
        "",
        f"   Datas diferentes no mês ({len(datas)} casos): a próxima leitura fica mais perto da",
        f"   avaliação mais recente em {percentual((distancia_recente < distancia_antiga).sum(), len(datas))}"
        f" e da mais antiga em {percentual((distancia_recente > distancia_antiga).sum(), len(datas))};",
        f"   a avaliação mais recente é a menor em {percentual(recente_menor.sum(), len(datas))} dos casos.",
    ]

    no_mesmo_arquivo = (detalhe['arquivos'] == 1).sum()
    linhas += [
        "",
        "7) Origem das leituras múltiplas",
        f"   no mesmo arquivo:       {no_mesmo_arquivo} ({percentual(no_mesmo_arquivo, multiplos)})",
        f"   em arquivos diferentes: {multiplos - no_mesmo_arquivo} "
        f"({percentual(multiplos - no_mesmo_arquivo, multiplos)})",
        "",
        "8) Arquivos com mais ocorrências",
    ]
    por_arquivo = detalhe['lista_arquivos'].str.split(', ').explode().value_counts().head(10)
    linhas += [f"   {arquivo:25} {quantidade}" for arquivo, quantidade in por_arquivo.items()]

    linhas += resumir_levantamentos(levantamentos)
    linhas += resumir_copias(copias)

    return '\n'.join(linhas)


def main():
    df = preparar_leituras(ler_prata())
    detalhe = descrever_multiplas(df)
    levantamentos = avaliar_levantamentos(df)
    copias = detectar_copias(df)
    resumo = montar_resumo(df, detalhe, levantamentos, copias)

    print()
    print(resumo)

    with open(ARQUIVO_RESUMO, 'w', encoding='utf-8') as f:
        f.write(resumo + '\n')
    detalhe.sort_values(COLUNAS_GRUPO).to_csv(ARQUIVO_DETALHE, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    levantamentos.sort_values('diferenca_mediana').to_csv(
        ARQUIVO_LEVANTAMENTOS, index=False, sep=';', decimal=',', encoding='utf-8-sig'
    )
    copias.to_csv(ARQUIVO_COPIAS, index=False, sep=';', decimal=',', encoding='utf-8-sig')

    print()
    print(f"Resumo salvo em {ARQUIVO_RESUMO}")
    print(f"Detalhe de cada caso salvo em {ARQUIVO_DETALHE}")
    print(f"Avaliação dos levantamentos salva em {ARQUIVO_LEVANTAMENTOS}")
    print(f"Levantamentos copiados salvos em {ARQUIVO_COPIAS}")


if __name__ == "__main__":
    main()
