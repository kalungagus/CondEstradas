# Previsão das condições do pavimento com redes recorrentes

Projeto da disciplina de Deep Learning para prever o valor futuro do Índice de
Condição da Manutenção (ICM) em trechos de rodovias federais. Os dados foram
obtidos em levantamentos públicos do DNIT e consolidados por trecho de 1 km e
mês.

O primeiro experimento utiliza redes recorrentes para regressão. O notebook
permite construir e comparar SimpleRNN, GRU e LSTM.

## Arquivos principais

- `dataset/dataset_icm.csv`: dataset mensal consolidado utilizado no treinamento;
- `treinamento_modelo.ipynb`: preparação, treinamento e avaliação dos modelos;
- `pre_processamento.py`: padronização dos arquivos originais;
- `gerar_dataset.py`: consolidação das leituras por trecho e mês;
- `requirements.txt`: dependências necessárias para executar o notebook.

## Ambiente recomendado

Para evitar diferenças de versões entre os integrantes, utilize **Python 3.13**
em um ambiente virtual exclusivo do projeto. O TensorFlow está fixado em uma
versão estável que possui distribuição oficial para essa versão do Python.

Antes da configuração, confirme a instalação com:

```powershell
py -3.13 --version
```

Se o comando não encontrar o interpretador, instale o Python 3.13 de 64 bits
pelo site oficial do Python e habilite o Python Launcher durante a instalação.

### Configuração automática no Windows

Depois de clonar ou baixar o repositório, abra o PowerShell na pasta do projeto
e execute:

```powershell
powershell -ExecutionPolicy Bypass -File .\configurar_ambiente.ps1
```

O script cria a pasta `.venv`, atualiza o `pip` e instala todas as dependências.

### Configuração manual

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Não é necessário ativar o ambiente para executar esses comandos.

## Seleção do kernel no VS Code

1. Abra `treinamento_modelo.ipynb`.
2. Clique no nome do kernel no canto superior direito do notebook.
3. Escolha **Select Another Kernel** e depois **Python Environments**.
4. Selecione o interpretador localizado em `.venv\Scripts\python.exe`.
5. Execute **Restart Kernel** e depois **Run All**.

Para conferir se o ambiente correto está selecionado, execute em uma célula:

```python
import sys
print(sys.executable)
```

O caminho exibido deve terminar em `.venv\Scripts\python.exe`.

## Execução

O notebook deve ser executado a partir da raiz do repositório, pois utiliza o
caminho relativo `dataset/dataset_icm.csv`.

O conjunto de teste deve permanecer reservado até a seleção final da
arquitetura e dos hiperparâmetros. A comparação entre SimpleRNN, GRU e LSTM é
feita inicialmente com o conjunto de validação.

## Observação sobre recursos

O dataset gera aproximadamente um milhão de janelas de treinamento na
configuração inicial. O treinamento pode ser mais rápido em uma máquina com
GPU, mas o código também pode ser executado em CPU, com maior tempo de espera.
