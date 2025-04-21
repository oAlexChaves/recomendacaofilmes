
# 🎬 IMDb Top 1000 - Análise Interativa com Streamlit

Este é um projeto de **análise interativa** dos 1000 melhores filmes do IMDb, criado com **Python**, **Pandas**, **Plotly** e **Streamlit**. A aplicação permite explorar os dados de forma visual, bem como receber **recomendações de filmes** com base em similaridade de conteúdo.

---

## 🔧 Funcionalidades

- 📊 **Distribuição do Meta Score por Gênero**  
  Veja como as notas da crítica (Meta Score) se distribuem entre os diferentes gêneros.

- 💸 **Receita Bruta vs Nota**  
  Explore a relação entre a receita arrecadada e as notas do IMDb ou Meta Score.

- 📈 **Evolução das Notas com o Tempo**  
  Analise como as avaliações mudaram ao longo das décadas.

- 💰 **Receita Média por Gênero**  
  Descubra quais gêneros lucraram mais, em média.

- 🎯 **Sistema de Recomendação de Filmes**  
  Receba sugestões de filmes semelhantes com base em gêneros, diretor e elenco usando **TF-IDF** e **similaridade de cosseno**.

---

## 📁 Estrutura do Projeto

```
📂 imdb_app/
├── imdb_top_1000.csv         # Dataset principal
├── app.py                    # Código principal da aplicação Streamlit
└── README.md                 # Este arquivo
```

---

## ▶️ Como Executar

1. **Clone o repositório:**
```bash
git clone https://github.com/seu-usuario/imdb-top1000-streamlit.git
cd imdb-top1000-streamlit
```

2. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

3. **Execute a aplicação:**
```bash
streamlit run app.py
```

4. Acesse no navegador:  
   👉 http://localhost:8501

---

## 📦 Requisitos

- Python 3.8+
- Bibliotecas:
  - `pandas`
  - `plotly`
  - `streamlit`
  - `scikit-learn`

**Exemplo de `requirements.txt`:**
```txt
streamlit
pandas
plotly
scikit-learn
```

---

## 📊 Sobre o Dataset

O dataset **IMDb Top 1000** contém informações dos filmes mais bem avaliados do IMDb, incluindo:

- Título (`Series_Title`)
- Ano de lançamento (`Released_Year`)
- Diretor e estrelas principais
- Gênero (`Genre`)
- Nota IMDb e Meta Score
- Receita bruta (`Gross`)

Fonte: Dataset público disponível no Kaggle ([IMDb Top 1000 Movies](https://www.kaggle.com/datasets))

---

## 🧠 Recomendação de Filmes - Como Funciona?

O sistema de recomendação utiliza:

- **TF-IDF Vectorizer:** para transformar os textos combinados (gênero, diretor e elenco) em vetores.
- **Cosseno Similarity:** para calcular a similaridade entre os filmes com base nesses vetores.

Com isso, o usuário pode escolher um filme e receber recomendações de outros títulos similares.

---

## 📄 Licença

Este projeto é livre para uso acadêmico e educacional. Verifique a [licença do dataset original](https://www.kaggle.com/datasets).
