import re
import os
import sys
import pandas as pd

PASTA_BRONZE = './bronze'
PASTA_PRATA = './prata'
# Fica fora da pasta prata para não ser lido junto com os dados na etapa de união
ARQUIVO_RELATORIO = './relatorio_pre_processamento.csv'

# Colunas que ficarão nos arquivos processados (todas as saídas têm este esquema)
COLUNAS_DESEJADAS = ['uf', 'rodovia', 'sentido', 'km inicial', 'km final', 'data', 'icc', 'icp', 'icm']
COLUNAS_NUMERICAS = ['km inicial', 'km final', 'icc', 'icp', 'icm']

# Mapa de tradução dos nomes de colunas das diferentes versões dos arquivos
MAPA_COLUNAS = {
    'ic': 'icc',
    'ip': 'icp',
    'km_inicial': 'km inicial',
    'km_final': 'km final',
    'data aval.': 'data',
}

# Os xlsx usam "Crescente"/"Decrescente" e os CSVs "C"/"D"
MAPA_SENTIDO = {
    'crescente': 'C',
    'decrescente': 'D',
    'c': 'C',
    'd': 'D',
}

# Colunas usadas para localizar a linha do cabeçalho, que pode estar em
# qualquer linha no começo do arquivo. Exige-se ao menos 2 delas na linha.
COLUNAS_CABECALHO = {'uf', 'rodovia', 'data', 'km inicial', 'km_inicial'}
MINIMO_COLUNAS_CABECALHO = 2

# Os arquivos possuem encodings e delimitadores diferentes
ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
DELIMITADORES = [";", ","]

MESES = {
    'janeiro': 1, 'jan': 1,
    'fevereiro': 2, 'fev': 2,
    'março': 3, 'marco': 3, 'mar': 3,
    'abril': 4, 'abr': 4,
    'maio': 5, 'mai': 5,
    'junho': 6, 'jun': 6,
    'julho': 7, 'jul': 7,
    'agosto': 8, 'ago': 8,
    'setembro': 9, 'set': 9,
    'outubro': 10, 'out': 10,
    'novembro': 11, 'nov': 11,
    'dezembro': 12, 'dez': 12,
}


def normalizar_nome_coluna(coluna):
    return str(coluna).strip(' "\r\n').lower()


def eh_cabecalho(valores):
    nomes = {normalizar_nome_coluna(valor) for valor in valores}
    return len(nomes & COLUNAS_CABECALHO) >= MINIMO_COLUNAS_CABECALHO


def extrair_data_do_nome(nome_arquivo):
    nome = os.path.splitext(nome_arquivo.lower())[0]

    # ------------------------------------------------
    # Ano em 4 dígitos e mês escrito por extenso ou abreviado.
    # O mês precisa ser uma palavra inteira, para que "set" não case
    # com "dataset", por exemplo.
    # ------------------------------------------------
    ano_match = re.search(r'(?<!\d)20\d{2}(?!\d)', nome)
    if ano_match:
        palavras = re.findall(r'[a-zç]+', nome)
        for palavra in palavras:
            if palavra in MESES:
                return pd.Timestamp(int(ano_match.group()), MESES[palavra], 1)

    # ------------------------------------------------
    # Formatos numéricos. Os lookarounds impedem que "2025_10" seja lido
    # como mês 1 (o "0?[1-9]" casaria só o "1").
    # ------------------------------------------------
    # MM_YYYY ou MM-YYYY
    match = re.search(r'(?<!\d)(1[0-2]|0?[1-9])[-_/](20\d{2})(?!\d)', nome)
    if match:
        return pd.Timestamp(int(match.group(2)), int(match.group(1)), 1)

    # YYYY_MM ou YYYY-MM
    match = re.search(r'(?<!\d)(20\d{2})[-_/](1[0-2]|0?[1-9])(?!\d)', nome)
    if match:
        return pd.Timestamp(int(match.group(1)), int(match.group(2)), 1)

    return None


def extrair_data(arquivo, df):
    data = extrair_data_do_nome(os.path.basename(arquivo))
    if data is not None:
        return data

    # ------------------------------------------------
    # Em último caso usa o mês mais frequente da coluna de data
    # ------------------------------------------------
    if 'data' in df.columns:
        meses = df['data'].dropna().dt.to_period('M')
        if not meses.empty:
            return meses.mode().iloc[0].to_timestamp()

    raise ValueError("não foi possível identificar a data do levantamento.")


def obter_cabecalho_csv(arquivo):
    for encoding in ENCODINGS:
        try:
            with open(arquivo, "r", encoding=encoding) as f:
                for numero_linha, linha in enumerate(f):
                    for delimitador in DELIMITADORES:
                        if eh_cabecalho(linha.split(delimitador)):
                            return encoding, delimitador, numero_linha
        except UnicodeDecodeError:
            continue

    raise ValueError("não foi possível identificar o cabeçalho.")


def ler_csv(arquivo):
    encoding, delimitador, linha_cabecalho = obter_cabecalho_csv(arquivo)
    return pd.read_csv(
        arquivo,
        encoding=encoding,
        sep=delimitador,
        skiprows=linha_cabecalho,
        dtype=str
    )


def ler_xlsx(arquivo):
    df = pd.read_excel(arquivo, header=None)

    for indice, linha in df.iterrows():
        if eh_cabecalho(linha.values):
            df.columns = linha.values
            return df.loc[indice + 1:].reset_index(drop=True)

    raise ValueError("não foi possível identificar o cabeçalho.")


def renomear_colunas(df):
    novos_nomes = []
    for coluna in df.columns:
        nome = normalizar_nome_coluna(coluna)
        novo = MAPA_COLUNAS.get(nome, nome)
        # Não sobrescreve uma coluna que já existe com o nome de destino
        novos_nomes.append(novo if novo not in novos_nomes else nome)
    df.columns = novos_nomes

    # Alguns arquivos repetem blocos de colunas (ex.: 09/2024); mantém o primeiro
    return df.loc[:, ~df.columns.duplicated()]


def para_numero(serie):
    texto = serie.astype(str).str.strip()
    # Valores com vírgula usam o padrão brasileiro: "1.333,8" -> "1333.8"
    com_virgula = texto.str.contains(',', regex=False)
    texto = texto.where(
        ~com_virgula,
        texto.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    )
    return pd.to_numeric(texto, errors='coerce')


def para_data(serie):
    texto = serie.astype(str).str.strip()
    data = pd.to_datetime(texto, format='%d/%m/%Y', errors='coerce')
    iso = pd.to_datetime(texto, format='ISO8601', errors='coerce')
    # Alguns CSVs trazem a data como número serial do Excel (ex.: 45182 = 13/09/2023)
    # Converte só os valores existentes, como inteiros: pd.to_datetime(unit='D')
    # sobre séries com NaN dispara FloatingPointError de forma intermitente
    serial = texto[texto.str.fullmatch(r'\d{5}')].astype('int64')
    excel = pd.Timestamp('1899-12-30') + pd.to_timedelta(serial, unit='D')
    return data.fillna(iso).fillna(excel)


def padronizar_sentido(serie):
    texto = serie.astype('string').str.strip()
    # Valores desconhecidos são mantidos como estão
    return texto.str.lower().map(MAPA_SENTIDO).fillna(texto)


def data_de_ano_mes(df):
    partes = pd.DataFrame({
        'year': para_numero(df['ano']),
        'month': para_numero(df['mes']),
        'day': 1,
    })
    return pd.to_datetime(partes, errors='coerce')


def padronizar(df):
    """Retorna o DataFrame no esquema padrão e um dicionário que descreve
    as colunas que precisaram ser geradas."""
    df = renomear_colunas(df)

    if 'uf' not in df.columns:
        raise ValueError("coluna 'uf' não encontrada.")
    df = df.dropna(subset=['uf'])

    colunas_geradas = {}

    if 'data' in df.columns:
        data = para_data(df['data'])
    else:
        data = pd.Series(pd.NaT, index=df.index)

    # Monta a data a partir de ano e mês quando ela não existir
    if data.isna().any() and {'ano', 'mes'} <= set(df.columns):
        ano_mes = data_de_ano_mes(df)
        preenchidas = data.isna() & ano_mes.notna()
        if preenchidas.any():
            data = data.fillna(ano_mes)
            colunas_geradas['data'] = f"ano/mes ({preenchidas.sum()} linhas)"

    # Garante o mesmo esquema em todas as saídas, mesmo sem alguma coluna na origem
    df = df.reindex(columns=COLUNAS_DESEJADAS)

    for coluna in COLUNAS_NUMERICAS:
        df[coluna] = para_numero(df[coluna])
    df['sentido'] = padronizar_sentido(df['sentido'])
    df['data'] = data

    return df, colunas_geradas


def processar(arquivo, destino, saidas_geradas):
    """Processa um arquivo e retorna uma linha do relatório."""
    if arquivo.lower().endswith(".xlsx"):
        df = ler_xlsx(arquivo)
    else:
        df = ler_csv(arquivo)

    df, colunas_geradas = padronizar(df)

    # Arquivos só com ICMNP (rodovias não pavimentadas) não entram na camada prata
    if df['icm'].isna().all():
        print(f'{os.path.basename(arquivo):45} → ignorado (sem ICM)')
        return {
            'arquivo_original': os.path.basename(arquivo),
            'observacoes': 'ignorado: sem coluna ICM (ICMNP é de rodovia não pavimentada)',
        }

    data = extrair_data(arquivo, df)
    observacoes = []

    # Último recurso: usa o mês do levantamento (do nome do arquivo)
    sem_data = df['data'].isna()
    if sem_data.any():
        df.loc[sem_data, 'data'] = data
        origem = f"mês do levantamento ({sem_data.sum()} linhas)"
        if 'data' in colunas_geradas:
            colunas_geradas['data'] += f" + {origem}"
        else:
            colunas_geradas['data'] = origem

    nome_saida = f"icm_{data.year:04d}_{data.month:02d}.csv"
    if nome_saida in saidas_geradas:
        # Dois arquivos do mesmo mês: não sobrescreve, salva com sufixo
        sufixo = 2
        while f"icm_{data.year:04d}_{data.month:02d}_{sufixo}.csv" in saidas_geradas:
            sufixo += 1
        aviso = f"{nome_saida} já gerado a partir de {saidas_geradas[nome_saida]}"
        print(f"AVISO: {aviso}")
        observacoes.append(aviso)
        nome_saida = f"icm_{data.year:04d}_{data.month:02d}_{sufixo}.csv"
    saidas_geradas[nome_saida] = os.path.basename(arquivo)

    caminho_saida = os.path.join(destino, nome_saida)
    df.to_csv(caminho_saida, index=False, sep=';', decimal=',', encoding="utf-8-sig", date_format='%Y-%m-%d')
    print(f'{os.path.basename(arquivo):45} → {nome_saida} ({len(df)} linhas)')

    return {
        'arquivo_original': os.path.basename(arquivo),
        'arquivo_gerado': nome_saida,
        'linhas': len(df),
        'data_minima': df['data'].min().date(),
        'data_maxima': df['data'].max().date(),
        'colunas_geradas': '; '.join(f"{coluna}: {origem}" for coluna, origem in colunas_geradas.items()),
        'colunas_vazias': ', '.join(coluna for coluna in COLUNAS_DESEJADAS if df[coluna].isna().all()),
        'observacoes': '; '.join(observacoes),
    }


def main():
    os.makedirs(PASTA_PRATA, exist_ok=True)

    arquivos = sorted(
        arquivo for arquivo in os.listdir(PASTA_BRONZE)
        if arquivo.lower().endswith((".csv", ".xlsx"))
    )
    print(f"Encontrados {len(arquivos)} arquivos.")

    saidas_geradas = {}
    relatorio = []
    falhas = []
    for arquivo in arquivos:
        try:
            relatorio.append(processar(os.path.join(PASTA_BRONZE, arquivo), PASTA_PRATA, saidas_geradas))
        except Exception as erro:
            falhas.append(arquivo)
            relatorio.append({'arquivo_original': arquivo, 'observacoes': f"ERRO: {erro}"})
            print(f"ERRO em {arquivo}: {erro}")

    pd.DataFrame(relatorio).astype({'linhas': 'Int64'}).to_csv(ARQUIVO_RELATORIO, index=False, sep=';', encoding="utf-8-sig")
    print(f"{len(saidas_geradas)} arquivos gerados, {len(falhas)} falhas. Relatório em {ARQUIVO_RELATORIO}")
    if falhas:
        sys.exit(1)


if __name__ == "__main__":
    main()
