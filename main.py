import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")
st.title("🎬 Análise Interativa - IMDb Top 1000")

# --- CARREGAMENTO E LIMPEZA ---
@st.cache_data
def carregar_dados():
    df = pd.read_csv("imdb_top_1000.csv")

    # Limpeza da coluna Meta_score
    df = df[df["Meta_score"].notna()]
    df = df[df["Meta_score"].astype(str).str.strip() != ""]
    df["Meta_score"] = pd.to_numeric(df["Meta_score"], errors="coerce")
    df = df[df["Meta_score"].notna()]

    # Limpeza da coluna Gross
    df = df[df["Gross"].notna()]
    df["Gross"] = df["Gross"].astype(str).str.replace(r"[\$,]", "", regex=True)
    df["Gross"] = pd.to_numeric(df["Gross"], errors="coerce")

    # Converte IMDB_Rating
    df["IMDB_Rating"] = pd.to_numeric(df["IMDB_Rating"], errors="coerce")

    df = df.dropna()

    return df

df = carregar_dados()

# --- FUNÇÕES DE VISUALIZAÇÃO ---
def exibir_boxplot_meta_score_por_genero(df):
    st.subheader("📊 Distribuição do Meta Score por Gênero")
    if "Meta_score" in df.columns and "Genre" in df.columns:
        df_exploded = df.assign(Genre=df["Genre"].str.split(',')).explode("Genre")
        df_exploded["Genre"] = df_exploded["Genre"].str.strip()

        unique_genres = df_exploded["Genre"].value_counts().index.tolist()
        selected_genres = st.multiselect("🎭 Selecione os gêneros:", unique_genres, default=unique_genres[:6])

        filtered_df = df_exploded[df_exploded["Genre"].isin(selected_genres)]

        if not filtered_df.empty:
            fig = px.box(
                filtered_df,
                x="Genre",
                y="Meta_score",
                color="Genre",
                points="all",
                title="Distribuição do Meta Score por Gênero"
            )
            fig.update_layout(xaxis_title="Gênero", yaxis_title="Meta Score")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Nenhum dado disponível para os gêneros selecionados.")
    else:
        st.warning("O dataset precisa conter as colunas 'Meta_score' e 'Genre'.")

def exibir_relacao_receita_vs_nota(df):
    st.subheader("💸 Receita Bruta vs Nota")
    nota_tipo = st.selectbox("Escolha o tipo de nota:", ["IMDB_Rating", "Meta_score"])

    df_validos = df[df["Gross"].notna() & df[nota_tipo].notna()]

    if not df_validos.empty:
        fig = px.scatter(
            df_validos,
            x="Gross",
            y=nota_tipo,
            hover_name="Series_Title",
            title=f"Relação entre Receita Bruta e {nota_tipo}",
            labels={"Gross": "Receita Bruta (US$)", nota_tipo: nota_tipo},
            opacity=0.7,
            trendline="ols"
        )
        fig.update_layout(xaxis_title="Receita Bruta (US$)", yaxis_title=f"{nota_tipo}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dados insuficientes para o gráfico.")

def exibir_evolucao_das_notas(df):
    st.subheader("📈 Evolução das Notas com o Tempo")
    nota_tipo = st.selectbox("Selecione o tipo de nota:", ["IMDB_Rating", "Meta_score"], key="evolucao_select")

    df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors="coerce")
    df = df[df["Released_Year"].notna() & df[nota_tipo].notna()]

    df_grouped = df.groupby("Released_Year")[nota_tipo].mean().reset_index()
    df_grouped = df_grouped.sort_values("Released_Year")

    fig = px.line(
        df_grouped,
        x="Released_Year",
        y=nota_tipo,
        title=f"Evolução da nota média ({nota_tipo}) ao longo do tempo",
        labels={"Released_Year": "Ano de Lançamento", nota_tipo: "Nota Média"},
        markers=True
    )
    fig.update_layout(xaxis_title="Ano", yaxis_title="Nota Média", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

def exibir_receita_por_genero(df):
    st.subheader("💰 Receita Média por Gênero")

    if "Gross" in df.columns and "Genre" in df.columns:
        df = df[df["Gross"].notna()]
        df["Gross"] = df["Gross"].replace('[\$,]', '', regex=True).astype(float)

        df_exploded = df.assign(Genre=df["Genre"].str.split(',')).explode("Genre")
        df_exploded["Genre"] = df_exploded["Genre"].str.strip()

        df_grouped = df_exploded.groupby("Genre")["Gross"].mean().reset_index()
        df_grouped = df_grouped.sort_values("Gross", ascending=False)

        fig = px.bar(
            df_grouped,
            x="Genre",
            y="Gross",
            title="Receita Média por Gênero",
            labels={"Genre": "Gênero", "Gross": "Receita Média (USD)"},
            text_auto='.2s',
            color="Gross",
            color_continuous_scale="blues"
        )
        fig.update_layout(xaxis_title="Gênero", yaxis_title="Receita Média", xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("O dataset precisa conter as colunas 'Gross' e 'Genre'.")

# --- MENU LATERAL ---
opcao = st.sidebar.selectbox(
    "📊 Selecione a visualização:",
    [
        "Distribuição do Meta Score por Gênero",
        "Receita Bruta vs Nota",
        "Evolução das Notas com o Tempo",
        "Receita Média por Gênero",
        "🎞️ Recomendador de Filmes"
    ]
)
def exibir_recomendacoes(df):
    st.subheader("🎯 Recomendação de Filmes")

    colunas_necessarias = ["Genre", "Director", "Star1", "Star2", "Star3", "Star4", "Series_Title", "Released_Year"]
    if all(col in df.columns for col in colunas_necessarias):

        def combinar_infos(row):
            # Coletamos os atores em uma lista e removemos duplicatas com set
            atores = list({row['Star1'], row['Star2'], row['Star3'], row['Star4']})
            atores_str = " ".join(ator for ator in atores if pd.notnull(ator))
            
            genero = row['Genre'] if pd.notnull(row['Genre']) else ""
            diretor = row['Director'] if pd.notnull(row['Director']) else ""

            return f"{genero} {diretor} {atores_str}"


        df["tags"] = df.apply(combinar_infos, axis=1)

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df["tags"])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        def recomendar_filmes(titulo_filme, top_n=5):
            try:
                idx = df[df["Series_Title"] == titulo_filme].index[0]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
                recomendados_idx = [i[0] for i in sim_scores]
                recomendados = df.iloc[recomendados_idx][
                    ["Series_Title", "Released_Year", "Director", "Star1"]
                ].copy()
                recomendados["Similaridade"] = [i[1] for i in sim_scores]
                return recomendados
            except IndexError:
                return pd.DataFrame()

        filmes_disponiveis = df["Series_Title"].dropna().unique().tolist()
        filme_escolhido = st.selectbox("🎬 Escolha um filme para receber recomendações:", sorted(filmes_disponiveis))

        if filme_escolhido:
            recomendacoes = recomendar_filmes(filme_escolhido)

            if not recomendacoes.empty:
                st.markdown("**Filmes recomendados:**")
                for i, row in recomendacoes.iterrows():
                    st.write(
                        f"**{row['Series_Title']}** ({int(row['Released_Year'])}) — 🎬 *{row['Director']}* | "
                        f"⭐ {row['Star1']} | 📈 Similaridade: `{row['Similaridade']:.2f}`")
            else:
                st.warning("Não foi possível gerar recomendações para esse filme.")
    else:
        st.error("O dataset precisa conter as colunas: Genre, Director, Star1, Star2, Star3, Star4, Series_Title e Released_Year.")



# --- CHAMADAS DINÂMICAS ---
if opcao == "Distribuição do Meta Score por Gênero":
    exibir_boxplot_meta_score_por_genero(df)
elif opcao == "Receita Bruta vs Nota":
    exibir_relacao_receita_vs_nota(df)
elif opcao == "Evolução das Notas com o Tempo":
    exibir_evolucao_das_notas(df)
elif opcao == "Receita Média por Gênero":
    exibir_receita_por_genero(df)
elif opcao == "🎞️ Recomendador de Filmes":
    exibir_recomendacoes(df)
