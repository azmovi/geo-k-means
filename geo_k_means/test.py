import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Criando um DataFrame de exemplo com sexo e idade
np.random.seed(0)
data = {
    'Sexo': np.random.choice(['Masculino', 'Feminino'], size=100),
    'Idade': np.random.randint(1, 101, size=100)
}
df = pd.DataFrame(data)

# Definindo as faixas etárias
bins = [0, 15, 30, 45, 60, 75, 90, 105]
faixas_etarias = ['1-15', '16-30', '31-45', '46-60', '61-75', '76-90', '91-100']

# Adicionando uma coluna para faixa etária
df['Faixa Etária'] = pd.cut(df['Idade'], bins=bins, labels=faixas_etarias, right=False)

# Agrupando por faixa etária e sexo e contando o número de homens e mulheres em cada grupo
counts = df.groupby(['Faixa Etária', 'Sexo']).size().unstack(fill_value=0)

# Criando o gráfico de barras
counts.plot(kind='bar', stacked=True)

# Adicionando rótulos aos eixos
plt.xlabel('Faixa Etária')
plt.ylabel('Contagem')

# Exibindo o gráfico
plt.show()
