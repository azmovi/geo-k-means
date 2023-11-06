# Desempenho

### Conceito:

- Testa de eficiência entre o algoritmo desenvolvido por mim
- Comparação entre algoritmo prontos no scikit-learn
- Resultado esperado é que o algoritmo da biblioteca scikit-learn tenha um melhor
desempenho devido a programação paralela implementada no algoritmo

### Metodos utilizados:

- Rand index

### Rand index:

- Medir a similaridade entre os dados presentes nas classes, desconsiderando permutações.
- Representa a precisão se um dado pertence ou não a um cluster.
- Tem um valor entre $0$ e $1$.
    - Sendo $0$ o agrupamento de dados não se assemelham.
    - Sendo $1$ o agrupamento de dados são iguais.
- Sendo $C$ a clusterização correta com $K$ classes, deve-se definir $a$ e $b$:
    - $a$: o número de pares de elementos que estão no **mesmo** conjunto em $C$ e no mesmo conjunto em $K$ 
    - $b$: o número de pares de elementos que estão em conjuntos **diferentes** em $C$ e em diferentes conjuntos em $K$ 

##### unadjusted Rand index:

- Não garante que atribuições aleatórias de rótulos obterá um valor próximo de zero
- Formulação matemática:
$$ RI = \frac{a + b}{C^{n}_{2}} $$

- Sendo $C^{n}_{2}$ o número total de pares possíveis no conjunto de dados

##### adjusted Rand index:

- Garante que atribuições aleatórias de rótulos obterá um valor próximo de zero ou negativo
- Formulação matemática:
$$ ARI = \frac{RI - E[RI]}{max(RI) - E[RI]} $$


### Referências:

[clustering-performance-evaluation
](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)

