import os
import glob
import numpy as np
import pandas as pd

PASTA_PRATA = './prata'
PASTA_OURO = './ouro'
ARQUIVO_DATASET = os.path.join(PASTA_OURO, 'dataset_icm.csv')
# Registro das células em que a média foi trocada pela leitura mais próxima dos vizinhos
ARQUIVO_AJUSTES = os.path.join(PASTA_OURO, 'ajustes_vizinhos.csv')

COLUNAS_TRECHO = ['uf', 'rodovia', 'km_inicial', 'km_final']

# Série de um trecho: o mesmo km, no mesmo sentido, ao longo dos meses
COLUNAS_SERIE = COLUNAS_TRECHO + ['sentido']

# Uma mesma leitura aparece em vários arquivos (os arquivos mais recentes são
# retratos acumulados da malha), por isso é removida antes de calcular médias.
# Levantamentos copiados de meses anteriores com a data adiantada (a maior parte
# de fev/2024 repete jan/2024) são mantidos de propósito; eles são listados pelo
# analisar_leituras.py em relatorio_copias.csv
COLUNAS_LEITURA = ['uf', 'rodovia', 'sentido', 'km inicial', 'km final', 'data', 'icm']

# Leituras do mesmo trecho, sentido e mês com desvio padrão acima deste limite
# são consideradas discrepantes: em vez da média, usa-se a leitura mais próxima
# do mês anterior e do seguinte
LIMITE_DESVIO = 20

# Distância máxima, em meses, para que uma leitura seja considerada vizinha
JANELA_VIZINHOS_MESES = 3


def ler_prata():
    arquivos = sorted(glob.glob(os.path.join(PASTA_PRATA, '*.csv')))
    print(f"Encontrados {len(arquivos)} arquivos na camada prata.")

    return pd.concat(
        (
            pd.read_csv(
                arquivo,
                sep=';',
                decimal=',',
                encoding='utf-8-sig',
                dtype={'uf': str, 'rodovia': str, 'sentido': str},
                parse_dates=['data']
            ).assign(arquivo=os.path.basename(arquivo))
            for arquivo in arquivos
        ),
        ignore_index=True
    )


def deduzir_sentido(df):
    # Arquivos antigos não têm a coluna sentido, mas o km de início é maior
    # que o de fim no sentido decrescente
    pela_ordem_km = pd.Series(
        np.select(
            [df['km inicial'] < df['km final'], df['km inicial'] > df['km final']],
            ['C', 'D'],
            default=None
        ),
        index=df.index
    )
    df['sentido_deduzido'] = df['sentido'].isna() & pela_ordem_km.notna()
    df['sentido'] = df['sentido'].fillna(pela_ordem_km)
    return df


def preparar_leituras(df):
    df = df.dropna(subset=['uf', 'rodovia', 'km inicial', 'km final', 'data', 'icm'])
    df['uf'] = df['uf'].str.strip().str.upper()
    df['rodovia'] = df['rodovia'].str.strip().str.upper()

    total = len(df)
    df = df.drop_duplicates(subset=COLUNAS_LEITURA)
    print(f"{total} leituras, {total - len(df)} repetidas entre arquivos removidas.")

    df = deduzir_sentido(df)

    # No sentido decrescente o km inicial é maior que o final; o trecho é
    # identificado pelo km de início, e pedaços menores que 1 km entram no km
    # a que pertencem
    km_inicio = df[['km inicial', 'km final']].min(axis=1)
    df['km_inicial'] = np.floor(km_inicio).astype(int)
    df['km_final'] = df['km_inicial'] + 1

    df['mes'] = df['data'].dt.to_period('M')

    return df


def calcular_vizinhos(mensal):
    """Recebe o ICM médio por trecho, sentido e mês e acrescenta a leitura
    anterior e a próxima da mesma série (dentro da janela) e a média das duas."""
    mensal = mensal.sort_values(COLUNAS_SERIE + ['mes']).reset_index(drop=True)
    numero_mes = mensal['mes'].dt.year * 12 + mensal['mes'].dt.month
    chaves_serie = [mensal[coluna] for coluna in COLUNAS_SERIE]

    for nome, deslocamento in {'anterior': 1, 'proxima': -1}.items():
        mensal[f'icm_{nome}'] = mensal['icm'].groupby(chaves_serie).shift(deslocamento)
        mensal[f'meses_ate_{nome}'] = (
            numero_mes - numero_mes.groupby(chaves_serie).shift(deslocamento)
        ).abs().astype('Int64')

    dentro_da_janela = {
        nome: mensal[f'icm_{nome}'].where(mensal[f'meses_ate_{nome}'] <= JANELA_VIZINHOS_MESES)
        for nome in ['anterior', 'proxima']
    }
    mensal['icm_vizinhos'] = pd.concat(dentro_da_janela, axis=1).mean(axis=1)

    return mensal


def calcular_icm_por_sentido(df):
    """ICM de cada trecho, sentido e mês: a média das leituras, ou, quando elas
    são discrepantes, a leitura mais próxima dos meses vizinhos."""
    grupos = COLUNAS_SERIE + ['mes']

    mensal = df.groupby(grupos, dropna=False)['icm'].agg(
        icm='mean', desvio='std', leituras='size'
    ).reset_index()
    mensal = calcular_vizinhos(mensal)

    discrepantes = mensal[(mensal['desvio'] > LIMITE_DESVIO) & mensal['icm_vizinhos'].notna()]

    candidatas = df.merge(discrepantes[grupos + ['icm_vizinhos']], on=grupos)
    candidatas['distancia'] = (candidatas['icm'] - candidatas['icm_vizinhos']).abs()
    escolhidas = candidatas.loc[candidatas.groupby(grupos)['distancia'].idxmin(), grupos + ['icm']]

    ajustes = discrepantes.merge(
        escolhidas.rename(columns={'icm': 'icm_escolhido'}), on=grupos
    ).rename(columns={'icm': 'icm_media'})

    mensal = mensal.merge(ajustes[grupos + ['icm_escolhido']], on=grupos, how='left')
    mensal['icm'] = mensal['icm_escolhido'].fillna(mensal['icm'])

    return mensal, ajustes


def calcular_icm_mensal(icm_por_sentido):
    # Média entre os sentidos, depois de cada sentido já ter um único valor no
    # mês, para que um sentido com mais leituras não pese mais que o outro
    return icm_por_sentido.groupby(COLUNAS_TRECHO + ['mes'])['icm'].mean()


def montar_series(icm_mensal):
    series = icm_mensal.unstack('mes')

    # Colunas mensais contínuas: meses sem levantamento ficam vazios, para que
    # cada coluna corresponda a um passo de tempo regular da série
    meses = pd.period_range(series.columns.min(), series.columns.max(), freq='M')
    series = series.reindex(columns=meses)
    series.columns = [str(mes) for mes in series.columns]

    return series.round(2).sort_index().reset_index()


def salvar_ajustes(ajustes, df):
    leituras = (
        df.merge(ajustes[COLUNAS_SERIE + ['mes']], on=COLUNAS_SERIE + ['mes'])
        .groupby(COLUNAS_SERIE + ['mes'])['icm']
        .agg(lambda valores: ' | '.join(f"{valor:g}".replace('.', ',') for valor in sorted(valores)))
        .rename('icm_lidos')
        .reset_index()
    )
    ajustes = ajustes.merge(leituras, on=COLUNAS_SERIE + ['mes'])
    ajustes['mes'] = ajustes['mes'].astype(str)

    colunas = COLUNAS_SERIE + [
        'mes', 'leituras', 'icm_lidos', 'desvio', 'icm_media', 'icm_escolhido',
        'icm_anterior', 'meses_ate_anterior', 'icm_proxima', 'meses_ate_proxima', 'icm_vizinhos',
    ]
    ajustes[colunas].round({'desvio': 4, 'icm_media': 4, 'icm_vizinhos': 4}).sort_values(
        COLUNAS_SERIE + ['mes']
    ).to_csv(ARQUIVO_AJUSTES, index=False, sep=';', decimal=',', encoding='utf-8-sig')


def resumir(series, ajustes):
    meses = [coluna for coluna in series.columns if coluna not in COLUNAS_TRECHO]
    valores = series[meses]

    leituras_por_trecho = valores.notna().sum(axis=1)
    meses_vazios = [mes for mes in meses if valores[mes].isna().all()]

    print(f"{len(series)} trechos e {len(meses)} meses ({meses[0]} a {meses[-1]}).")
    print(f"Preenchimento: {valores.notna().mean().mean():.1%} das células.")
    print(
        f"Meses com leitura por trecho: mínimo {leituras_por_trecho.min()}, "
        f"mediana {leituras_por_trecho.median():.0f}, máximo {leituras_por_trecho.max()}."
    )
    print(f"{len(meses_vazios)} meses sem nenhuma leitura: {', '.join(meses_vazios)}")
    print(
        f"{len(ajustes)} trecho-sentido-mês com leituras discrepantes (desvio > {LIMITE_DESVIO}) "
        f"usaram a leitura mais próxima dos vizinhos em vez da média."
    )


def main():
    os.makedirs(PASTA_OURO, exist_ok=True)

    df = preparar_leituras(ler_prata())
    icm_por_sentido, ajustes = calcular_icm_por_sentido(df)
    series = montar_series(calcular_icm_mensal(icm_por_sentido))

    series.to_csv(ARQUIVO_DATASET, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    salvar_ajustes(ajustes, df)
    resumir(series, ajustes)
    print(f"Dataset salvo em {ARQUIVO_DATASET}")
    print(f"Ajustes registrados em {ARQUIVO_AJUSTES}")


if __name__ == "__main__":
    main()
