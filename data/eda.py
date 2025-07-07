import warnings                 # Importa módulo para controle de warnings/avisos
import pandas as pd             # Biblioteca para manipulação de dados
import numpy as np              # Biblioteca para operações numéricas
import matplotlib.pyplot as plt # Biblioteca para criação de gráficos
import seaborn as sns           # Biblioteca para visualizações estatísticas
import phik                     # Biblioteca para correlação Phi-K entre variáveis categóricas e numéricas

from scipy import stats                               # Módulo de estatística do SciPy
from scipy.stats import skew, kurtosis, gaussian_kde  # Funções para calcular assimetria e curtose
from ydata_profiling import ProfileReport             # Biblioteca para gerar relatórios automáticos
# from IPython.display import display                   # Funções para exibir conteúdo em notebooks
from IPython.display import display, display_markdown # Funções para exibir conteúdo em notebooks

warnings.simplefilter('ignore', Warning) # Suprimir todos os warnings de forma simples
warnings.filterwarnings('ignore', category=Warning) # Filtra e ignora warnings de categoria Warning
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned') # Ignora warning específico do polyfit

plt.style.use('default')    # Define estilo padrão para gráficos matplotlib
sns.set_palette("coolwarm") # Define paleta de cores "coolwarm" para seaborn
#######################################################################################################################################################
class Eda: # Definição da classe para Análise Exploratória de Dados (EDA)
    def __init__(self, path_train):  # Method construtor da classe
        self.path_train = path_train # Armazena o caminho do arquivo do dataset
        self.target = None           # Inicializa a variável target como None
        self.df_L0 = None            # DataFrame original (Level 0) - dados brutos
        self.df_L1 = None            # DataFrame transformado (Level 1) - dados processados
        self.columns = None          # Lista das colunas do dataset

    ###################################################################################################################################################
    def load_dataset(self): # Method para carregar o dataset
        try: # Inicia bloco try para capturar exceções
            self.df_L0 = pd.read_csv(self.path_train) # Carrega o arquivo CSV no DataFrame L0
            self.df_L1 = self.df_L0.copy()            # Cria uma cópia dos dados originais em L1
            self.columns = self.df_L0.columns         # Armazena os nomes das colunas
            return False # Retorna False indicando sucesso no carregamento
        except FileNotFoundError as e: # Captura erro se arquivo não for encontrado
            print(f"\33[91m{e}\nPor favor, verifique o arquivo e tente novamente.\n") # Exibe mensagem de erro
            return True  # Retorna True indicando falha no carregamento

    ###################################################################################################################################################
    @staticmethod
    def report(df, text): # Method para gerar relatório completo
        profile = ProfileReport(df, title=str(text), html={"style":{"full_width":True}}, # Cria objeto ProfileReport
                                correlations={"auto":{"calculate":False,"warn_high_correlations":False},   # Desativa correlações automáticas
                                              "phi_k":{"calculate":True,"warn_high_correlations":True},    # Ativa correlação Phi-K
                                              "pearson":{"calculate":True,"warn_high_correlations":True},  # Ativa correlação de Pearson
                                              "spearman":{"calculate":True,"warn_high_correlations":True}, # Ativa correlação de Spearman
                                              "kendall":{"calculate":True,"warn_high_correlations":True},  # Ativa correlação de Kendall
                                              "cramers":{"calculate":True,"warn_high_correlations":True}}, # Ativa correlação de Cramer's V
                                dataset={"description": "The Pokemon with Stats dataset is for use in data science education.", # Descrição do dataset
                                         "author": "Myles O'Neill",    # Nome do autor
                                         "copyright_holder": "Kaggle", # Detentor dos direitos autorais
                                         "copyright_year": 2016,       # Ano do copyright
                                         "url": "https://www.kaggle.com/datasets/abcsds/pokemon/data"}, # URL do dataset
                                variables={"descriptions":{"#":"ID for each pokemon", # Descrição das colunas
                                           "Name":"Name of each pokemon",
                                           "Type 1":"Each pokemon has a type, this determines weakness/resistance to attacks",
                                           "Type 2":"Some pokemon are dual type and have 2",
                                           "Total":"Sum of all stats that come after this, a general guide to how strong a pokemon is",
                                           "HP":"Hit points, or health, defines how much damage a pokemon can withstand before fainting",
                                           "Attack":"The base modifier for normal attacks (eg. Scratch, Punch)",
                                           "Defense":"The base damage resistance against normal attacks",
                                           "SP Atk":"Special attack, the base modifier for special attacks (e.g. fire blast, bubble beam)",
                                           "SP Def": "The base damage resistance against special attacks",
                                           "Speed": "Determines which pokemon attacks first each round",
                                           "Generation": "Term used to group Pokémon media released during a certain time period",
                                           "Legendary": "Are a type of very rare and powerful type of Pokémon", }})
        profile.to_file(f"{text}.html") # Salva o relatório como arquivo HTML

    ###################################################################################################################################################
    def statistics(self, df, section, lv): # Method para exibir estatísticas descritivas
        print() # Imprime linha em branco para espaçamento
        display_markdown(f"### {section}. ESTATÍSTICAS DESCRITIVAS BÁSICAS - {lv}", raw=True) # Exibe título da seção
        # print(f"{section}. ESTATÍSTICAS DESCRITIVAS BÁSICAS - {lv}")
        print(f"Dimensões do Dataset de Treino: {df.shape}\n") # Exibe dimensões do dataset
        self.print_statistics(df) # Chama method para imprimir estatísticas detalhadas

    ###################################################################################################################################################
    @staticmethod
    def print_statistics(df): # Method que exibe várias estatísticas do dataset
        display(df.info(show_counts=False)) # Exibe informações gerais do DataFrame
        # display_markdown("#### [Primeiras Linhas]", raw=True) # Exibe subtítulo para primeiras linhas
        print("\n[Primeiras Linhas]")
        display(df.head()) # Exibe as 5 primeiras linhas
        # display_markdown("#### [Últimas Linhas]", raw=True) # Exibe subtítulo para últimas linhas
        print("\n[Últimas Linhas]")
        display(df.tail()) # Exibe as 5 últimas linhas
        numeric_cols = df.select_dtypes(include=[np.number]).columns # Seleciona apenas colunas numéricas
        # display_markdown("#### [Features Numéricas]", raw=True) # Exibe subtítulo para features numéricas
        print("\n[Features Numéricas]")
        display(df[numeric_cols].describe()) # Exibe estatísticas descritivas das colunas numéricas
        # display_markdown("#### [Features Categóricas]", raw=True) # Exibe subtítulo para features categóricas
        print("\n[Features Categóricas]")
        display(df.describe(include=['object', 'category', 'bool'])) # Exibe estatísticas das colunas categóricas
        # display_markdown("#### [Resumo Estatístico]", raw=True) # Exibe subtítulo para resumo estatístico
        print("\n[Resumo Estatístico]")
        display(df.describe().T) # Exibe resumo estatístico transposto

    ###################################################################################################################################################
    def set_target(self, target): # Method para definir a variável target
        self.target = target      # Atribui o nome da coluna target à propriedade da classe

    ###################################################################################################################################################
    def analysis_target(self): # Method para analisar a variável target
        # display_markdown(f"### 2. ANÁLISE DA VARIÁVEL TARGET - [{self.target}]", raw=True) # Exibe título da análise
        print(f"2. ANÁLISE DA VARIÁVEL TARGET - [{self.target}]\n")
        train_target = False # Inicializa variável de controle como False
        if self.target not in self.df_L0.columns: # Verifica se a coluna target existe no dataset
            print(f"\33[91mVariável [{self.target}] não encontrada no Dataset de Treino.\n") # Mensagem de erro
            train_target = True # Define como True indicando erro
        else: # Se a coluna target existe
            self.stats_target() # Chama method para calcular estatísticas do target
        return train_target # Retorna status da operação

    ###################################################################################################################################################
    def stats_target(self): # Method para calcular estatísticas da variável target
        target_data = self.df_L0[self.target] # Extrai dados da variável target

        print(f"Tipo da variável: {target_data.dtype}") # Exibe tipo de dado da variável
        print(f"Número de categorias únicas: {target_data.nunique()}") # Conta valores únicos
        print(f"Valores nulos: {target_data.isnull().sum()}") # Conta valores nulos

        if target_data.dtype in ['object', 'category', 'bool']: # Verifica se variável é categórica
            # display_markdown("#### ESTATÍSTICAS CATEGÓRICAS", raw=True) # Subtítulo para estatísticas categóricas
            print("ESTATÍSTICAS CATEGÓRICAS")
            print(f"Valores únicos: {list(target_data.unique())}") # Lista valores únicos
            print(f"Moda: {target_data.mode().iloc[0] if not target_data.mode().empty else 'N/A'}") # Calcula moda
            print(f"\nDistribuição de frequências: {target_data.value_counts()}") # Distribuição de frequências
        else: # Se variável é numérica
            # display_markdown("#### ESTATÍSTICAS NUMÉRICAS", raw=True) # Subtítulo para estatísticas numéricas
            print("ESTATÍSTICAS NUMÉRICAS")
            print(f"Média: {target_data.mean():,.2f}") # Calcula e exibe média
            print(f"Mediana: {target_data.median():,.2f}") # Calcula e exibe mediana
            print(f"Desvio Padrão: {target_data.std():,.2f}") # Calcula e exibe desvio padrão
            print(f"Mínimo: {target_data.min():,.2f}") # Encontra e exibe valor mínimo
            print(f"Máximo: {target_data.max():,.2f}") # Encontra e exibe valor máximo
            print(f"Assimetria (Skewness): {skew(target_data):.3f}") # Calcula assimetria
            print(f"Curtose (Kurtosis): {kurtosis(target_data):.3f}") # Calcula curtose

    ###################################################################################################################################################
    def transformation(self): # Method para aplicar transformações nos dados
        categorical_fill = {'Type 2': 'None'} # Dicionário para conversão de valores nulos em Type 2
        self.df_L1.fillna(categorical_fill, inplace=True) # Preenche valores nulos conforme dicionário
        self.df_L1.drop(['#'], axis=1, inplace=True) # Remove coluna '#' (ID) do dataset
        self.df_L1.set_index('Name', inplace=True) # Define coluna 'Name' como índice
        self.df_L1.index = self.df_L1.index.str.replace(".*(?=Mega)", "", regex=True) # Remove prefixos antes de "Mega"
        self.df_L1.bfill(inplace=True) # Preenche valores nulos com backward fill
        self.df_L1.drop_duplicates(inplace=True) # Remove linhas duplicadas do dataset

        convert_type = {'Total':'int16', 'HP':'int16', 'Attack':'int16', 'Defense':'int16', # Dicionário para conversão de tipos das colunas de stats
                        'Sp. Atk':'int16', 'Sp. Def':'int16', 'Speed':'int16', 'Generation': 'int8'}
        self.df_L1 = self.df_L1.astype(convert_type) # Aplica a conversão de tipos
        self.columns = self.df_L1.columns # Atualiza lista de colunas após transformações

    ###################################################################################################################################################
    def summary(self, df): # Method para exibir resumo do dataset
        # display_markdown("### 4. RESUMO DO DATASET", raw=True) # Título da seção de resumo
        print("4. RESUMO DO DATASET\n")
        print(f"Formato: {df.shape}") # Exibe dimensões do dataset (linhas, colunas)
        print(f"Variável Target: {self.target}") # Exibe nome da variável target
        print(f"Variáveis para Comparação: {self.columns}") # Exibe lista de colunas disponíveis
        print(f"Valores Nulos Totais: {df.isnull().sum().sum()}") # Conta total de valores nulos
        print(f"Duplicatas: {df.duplicated().sum()}\n") # Conta número de linhas duplicadas

    ###################################################################################################################################################
    def preview_target(self, df): # Method para visualizar a variável target
        if df[self.target].dtype in ['object', 'category', 'bool']: # Verifica se target é categórico
            fig, axes = plt.subplots(1, 2, figsize=(10, 6)) # Cria figura com 2 subplots
            fig.suptitle(f'Análise da Variável Target Categórica - {self.target}', fontsize=16) # Título da figura

            value_counts = df[self.target].value_counts() # Conta frequência de cada categoria

            axes[0].bar(value_counts.index, value_counts.values, color='skyblue', edgecolor='black') # Gráfico de barras
            axes[0].set_title(f'Distribution - {self.target}') # Título do primeiro subplot
            axes[0].set_xlabel(f'{self.target}') # Rótulo do eixo x
            axes[0].set_ylabel('Frequency') # Rótulo do eixo y
            axes[0].tick_params(axis='x', rotation=45) # Rotaciona rótulos do eixo x

            axes[1].axis('off')  # Desativa eixos do segundo subplot
            axes[1].axis('auto') # Define escala automática
            table_data = [[category, frequency, f"{frequency / len(df) * 100:.2f}%"] for category, frequency in value_counts.items()] # Dados da tabela
            table = axes[1].table(cellText=table_data, colLabels=['Category', 'Frequency', 'Percentage'], cellLoc='center', loc='center') # Cria tabela
            table.auto_set_font_size(False) # Desativa ajuste automático da fonte
            table.set_fontsize(12) # Define tamanho da fonte
            table.scale(1, 1.5) # Escala da tabela (largura, altura)
            table.auto_set_column_width([0, 1, 2]) # Ajusta largura das colunas
        else: # Se target é numérico
            fig, axes = plt.subplots(2, 2, figsize=(12, 8)) # Cria figura com 4 subplots (2x2)
            fig.suptitle(f'Análise da Variável Target Numérica - {self.target}', fontsize=16) # Título da figura

            axes[0, 0].hist(df[self.target], bins=50, alpha=0.7, color='skyblue', edgecolor='black') # Histograma

            try: # Tenta adicionar curva de densidade
                density = gaussian_kde(df[self.target].dropna()) # Calcula densidade kernel gaussiana
                xs = np.linspace(df[self.target].min(), df[self.target].max(),200) # Cria pontos para curva
                axes[0, 0].plot(xs, density(xs), 'r-', label='Density') # Plota curva de densidade
                axes[0, 0].legend() # Adiciona legenda
            except Exception: # Ignora erros na criação da curva
                pass # Não faz nada se der erro

            axes[0, 0].set_title(f'Distribution - {self.target}') # Título do histograma
            axes[0, 0].set_xlabel(f'{self.target}') # Rótulo do eixo x
            axes[0, 0].set_ylabel('Frequency') # Rótulo do eixo y

            bp = axes[0, 1].boxplot(df[self.target].dropna(), patch_artist=True) # Boxplot para identificar outliers
            bp['boxes'][0].set_facecolor('lightblue') # Define cor da caixa do boxplot
            axes[0, 1].set_title(f'Box Plot - {self.target}') # Título do boxplot
            axes[0, 1].set_ylabel(f'{self.target}') # Rótulo do eixo y

            stats.probplot(df[self.target], dist="norm", plot=axes[1, 0]) # Q-Q plot para testar normalidade
            axes[1, 0].set_title('Q-Q Plot - Normality') # Título do Q-Q plot

            log_prices = np.log(df[self.target]) # Aplica transformação logarítmica
            axes[1, 1].hist(log_prices, bins=50, alpha=0.7, color='lightgreen', edgecolor='black') # Histograma transformado
            axes[1, 1].set_title('Log-transformed distribution') # Título do histograma transformado
            axes[1, 1].set_xlabel(f'Log({self.target})') # Rótulo do eixo x
            axes[1, 1].set_ylabel('Frequency') # Rótulo do eixo y

        plt.tight_layout() # Ajusta espaçamento entre subplots
        plt.show() # Exibe a figura

    ###################################################################################################################################################
    def analysis_columns(self, df): # Method para analisar relação entre colunas e target
        # display_markdown("### 5. ANÁLISE DE VARIÁVEIS PARA COMPARAÇÃO", raw=True) # Título da seção
        print("5. ANÁLISE DE VARIÁVEIS PARA COMPARAÇÃO")

        if df[self.target].dtype in ['object', 'category', 'bool']: # Verifica se target é categórico
            for col in self.columns: # Itera sobre todas as colunas
                if col != self.target: # Pula a própria coluna target
                    self.categorical_target_analysis(df, col) # Chama análise para target categórico
        else: # Se target é numérico
            for col in self.columns: # Itera sobre todas as colunas
                if col != self.target: # Pula a própria coluna target
                    self.numeric_target_analysis(df, col) # Chama análise para target numérico

    ###################################################################################################################################################
    def categorical_target_analysis(self, df, col): # Method para analisar colunas vs target categórico
        fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # Cria figura com 2 subplots lado a lado
        sns.set_style("whitegrid") # Define estilo dos gráficos com grade

        if not df[col].dtype in ['object', 'category', 'bool']: # Se coluna é numérica vs target categórico
            sns.boxplot(data=df, x=self.target, y=col, ax=axes[0]) # Cria boxplot
            axes[0].set_title(f'Boxplot of {col} vs. {self.target}', fontsize=14) # Título do boxplot
            axes[0].tick_params(axis='x', rotation=45) # Rotaciona rótulos do eixo x
            axes[0].set_xlabel(self.target) # Rótulo do eixo x
            axes[0].set_ylabel(col) # Rótulo do eixo y
        else: # Se ambas são categóricas
            try: # Tenta criar gráfico de contagem
                sns.countplot(data=df, x=df[col], hue=df[self.target], ax=axes[0]) # Gráfico de contagem com agrupamento
                axes[0].set_title(f'Distribution: {self.target} by {col}', fontsize=14) # Título do gráfico
            except Exception: # Se falhar, cria gráfico simples
                value_counts = df[col].value_counts() # Conta valores únicos
                axes[0].bar(value_counts.index, value_counts.values) # Gráfico de barras simples
                axes[0].set_title(f'Distribution: {col}', fontsize=14) # Título do gráfico
            axes[0].tick_params(axis='x', rotation=45) # Rotaciona rótulos do eixo x
            axes[0].set_xlabel(col) # Rótulo do eixo x

        if not df[col].dtype in ['object', 'category', 'bool']: # Se coluna é numérica
            if col == 'Generation': # Caso especial para Generation
                value_counts = df[col].value_counts() # Conta valores únicos
                axes[1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%') # Gráfico de pizza
                axes[1].set_title(f'Distribution of {col}', fontsize=14) # Título do gráfico
            else: # Para outras variáveis numéricas
                axes[1].hist(df[col].dropna(), bins=30, alpha=0.7, color='lightgreen', edgecolor='black') # Histograma
                axes[1].set_title(f'Distribution: {col}', fontsize=14) # Título do histograma
                axes[1].set_ylabel('Frequency') # Rótulo do eixo y
        else: # Se coluna é categórica
            value_counts = df[col].value_counts() # Conta valores únicos
            if df[col].dtype == 'bool': # Se é booleana
                explode = (0.1, 0) # Define explosão para gráfico de pizza
                axes[1].pie(value_counts.values, labels=value_counts.index, explode=explode, autopct='%1.1f%%') # Pizza com explosão
                axes[1].set_title(f'Distribution of {col}', fontsize=14) # Título do gráfico
            else: # Para outras categóricas
                axes[1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%') # Gráfico de pizza normal
                axes[1].set_title(f'Distribution of {col}', fontsize=14) # Título do gráfico

        plt.tight_layout() # Ajusta espaçamento entre subplots
        plt.show() # Exibe a figura

    ###################################################################################################################################################
    def numeric_target_analysis(self, df, col): # Method para analisar colunas vs target numérico
        fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # Cria figura com 2 subplots lado a lado
        sns.set_style("whitegrid") # Define estilo dos gráficos com grade

        valid_data = df[[col, self.target]].dropna() # Remove linhas com valores nulos

        if df[col].dtype in ['object', 'category', 'bool']: # Se coluna é categórica
            sns.boxplot(data=df, x=col, y=self.target, ax=axes[0]) # Boxplot categórica vs numérica
            axes[0].set_title(f'Boxplot of {self.target} by {col}', fontsize=14) # Título do boxplot
            axes[0].tick_params(axis='x', rotation=45) # Rotaciona rótulos do eixo x
        else: # Se coluna é numérica
            if len(valid_data) >= 10: # Verifica se há dados suficientes
                try: # Tenta criar gráfico de regressão
                    sns.regplot(data=df, x=col, y=self.target, scatter_kws={"alpha": 0.4}, line_kws={"color": "green", "lw": 2, "alpha": 0.9},
                                order=2, ax=axes[0]) # Regressão polinomial de ordem 2
                    axes[0].set_title(f'Regplot: {col} vs. {self.target}', fontsize=14) # Título do regplot

                    z = np.polyfit(df[col].dropna(), df[self.target][df[col].notna()], 1) # Calcula coeficientes do ajuste linear
                    p = np.poly1d(z) # Cria função polinomial
                    axes[0].plot(df[col], p(df[col]), "r--", alpha=0.8) # Adiciona linha de tendência tracejada
                except np.linalg.LinAlgError: # Se der erro de álgebra linear
                    sns.scatterplot(data=valid_data, x=col, y=self.target, ax=axes[0]) # Scatter plot simples
                    axes[0].set_title(f'Scatter: {col} vs. {self.target}', fontsize=14) # Título do scatter
            else: # Se não há dados suficientes
                sns.scatterplot(data=valid_data, x=col, y=self.target, ax=axes[0]) # Scatter plot simples
                axes[0].set_title(f'Scatter: {col} vs. {self.target}', fontsize=14) # Título do scatter

        axes[0].set_xlabel(col) # Rótulo do eixo x
        axes[0].set_ylabel(self.target) # Rótulo do eixo y

        if df[col].dtype in ['object', 'category', 'bool']: # Se coluna é categórica
            sns.violinplot(data=df, x=col, y=self.target, ax=axes[1]) # Cria violin plot para mostrar distribuição do target por categoria
            axes[1].set_title(f'Violin Plot: {self.target} by {col}', fontsize=14) # Define título do violin plot
            axes[1].tick_params(axis='x', rotation=45) # Rotaciona rótulos do eixo x em 45 graus
        else: # Se coluna é numérica
            if len(valid_data) >= 10:  # Verifica se há dados suficientes
                try: # Tenta criar histograma 2D
                    hist2d = axes[1].hist2d(valid_data[col], valid_data[self.target], bins=20,
                                            cmap='Blues') # Cria histograma 2D com 20 bins usando colormap Blues
                    plt.colorbar(hist2d[3], ax=axes[1], shrink=0.8) # Adiciona barra de cores ao histograma 2D
                    axes[1].set_title(f'Density 2D: {col} vs. {self.target}', fontsize=14) # Define título do histograma 2D
                except ValueError: # Se der erro ao criar histograma 2D
                    axes[1].scatter(valid_data[col], valid_data[self.target], alpha=0.6, c='blue') # Cria scatter plot com transparência e cor azul
                    axes[1].set_title(f'Scatter: {col} vs. {self.target}', fontsize=14) # Define título do scatter plot
            else: # Se não há dados suficientes
                axes[1].scatter(valid_data[col], valid_data[self.target], alpha=0.6) # Cria scatter plot simples com transparência
                axes[1].set_title(f'Scatter: {col} vs. {self.target}', fontsize=14) # Define título do scatter plot
            axes[1].set_xlabel(col) # Rótulo do eixo x
            axes[1].set_ylabel(self.target) # Rótulo do eixo y

        plt.tight_layout() # Ajusta espaçamento entre subplots
        plt.show() # Exibe a figura

    ###################################################################################################################################################
    def correlation_heatmap(self, df): # Method para criar matriz de correlação completa
        # display_markdown("### 6. MATRIZ DE CORRELAÇÃO", raw=True) # Título da seção
        print("6. MATRIZ DE CORRELAÇÃO")
        numeric_cols = df.select_dtypes(include=[np.number]).columns # Seleciona apenas colunas numéricas

        if len(numeric_cols) < 2: # Verifica se há pelo menos 2 variáveis numéricas
            print("\33[91mDados insuficientes para matriz de correlação (menos de 2 variáveis numéricas).\n") # Mensagem de erro
            return # Sai da função

        plt.figure(figsize=(10, 6)) # Cria figura para o heatmap
        corr_matrix = phik.phik_matrix(df, interval_cols=numeric_cols) # Calcula matriz de correlação Phi-K

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Cria máscara triangular superior

        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, vmin=0.2, vmax=1, square=True,
                    cbar_kws={"shrink": .8})  # Heatmap da matriz de correlação
        plt.title('Correlação Phik (φk)', fontsize=16, fontweight='bold') # Título do gráfico
        plt.tight_layout() # Ajusta espaçamento
        plt.show() # Exibe a figura

        target_corr = corr_matrix[self.target].abs().sort_values(ascending=False) # Ordena correlações com target por valor absoluto
        print(f"\nCorrelações mais altas com [{self.target}]:\n") # Título da lista de correlações
        for var, corr in target_corr.items(): # Itera sobre as correlações
            if var != self.target: # Exclui correlação da variável consigo mesma
                strength = self.correlation_strength(abs(corr)) # Classifica força da correlação
                print(f"{var}: {corr:.3f} ({strength})") # Exibe nome, valor e classificação da correlação

    ###################################################################################################################################################
    @staticmethod
    def correlation_strength(corr_value): # Method para classificar força da correlação
        """Classifica a força da correlação""" # Docstring explicativa
        if corr_value >= 0.7: # Se correlação >= 0.7
            return "Forte"
        elif corr_value >= 0.5: # Se correlação >= 0.5
            return "Moderada"
        elif corr_value >= 0.3: # Se correlação >= 0.3
            return "Fraca"
        else: # Se correlação < 0.3
            return "Muito fraca"