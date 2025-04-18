import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings
from wordcloud import WordCloud
import streamlit.components.v1 as components
import base64

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.preprocessing import StandardScaler

import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist

import pyLDAvis
import pyLDAvis.lda_model
from bertopic import BERTopic

# Try importing rpy2
rpy2_available = True
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
    import logging

    pandas2ri.activate()
    rpy2_logger.setLevel(logging.ERROR)
except ImportError:
    rpy2_available = False

warnings.filterwarnings("ignore", category=FutureWarning, message=".*np.matrix.*")

st.set_page_config(page_title="Konu Modelleme Çalışması", layout="wide")
st.title("🧠 Kapsamlı Konu Modelleme Uygulaması")

st.sidebar.header("Yükleme ve Ayarlar")
uploaded_file = st.sidebar.file_uploader("CSV veya Excel dosyanızı yükleyin", type=["csv", "xlsx"])
n_topics = st.sidebar.slider("Konu Sayısı", 2, 20, 9)
model_options = ["LDA", "NMF", "BERTopic"]
if rpy2_available:
    model_options.append("STM (R)")
model_choice = st.sidebar.selectbox("Konu Modelleme Yöntemini Seçin", model_options)

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        return None

if uploaded_file:
    df = load_data(uploaded_file)
    st.success("Dosya başarıyla yüklendi!")
else:
    st.info("ÖN TANIMLI VERİLER YÜKLENDİ")
    # st.stop()
    df = pd.read_csv("processed_data.csv")

st.subheader("📄 Veri Önizleme")
st.dataframe(df)

text_column = st.selectbox("Modelleme için metin sütununu seçin", index=len(df.columns)-1, options=df.columns)
texts = df[text_column].astype(str).tolist()
date_column = st.selectbox("Zaman analizi için tarih sütununu seçin (isteğe bağlı)", index=2, options= [None] + list(df.columns))

if date_column:
    df[date_column] = pd.to_datetime(df[date_column], format='%Y', errors='coerce')
    df = df.dropna(subset=[date_column])
    df['year'] = df[date_column].dt.year

    st.subheader("📅 Yıl Dağıtımı")
    st.bar_chart(df['year'].value_counts().sort_index())

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)

# --- Export Button ---
def generate_download_link(df, filename="topic_distribution.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📅 Konu dağıtımı CSV dosyasını indirin</a>'
    return href

# --- Visualization Style Enhancements ---
def styled_line_chart(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("husl", n_colors=df.shape[1])
    df.plot(ax=ax, linewidth=2, color=palette)
    plt.title("Zaman İçinde Ortalama Konu Dağılımı", fontsize=16)
    plt.xlabel("Yıl")
    plt.ylabel("Ortalama Konu Ağırlığı")
    plt.grid(True)
    st.pyplot(fig)
    
if model_choice == "EDA":
    st.subheader("📊 Keşifsel Veri Analizi")

    # Basic info
    st.markdown("**Veri Kümesi Boyutları:**")
    st.write(f"Satırlar: {df.shape[0]}, 📚 Sütunlar: {df.shape[1]}")

    # Text Lengths
    df["text_length"] = df[text_column].apply(lambda x: len(str(x).split()))
    st.markdown("**Metin Uzunluğu İstatistikleri:**")
    st.write(df["text_length"].describe())

    # Histogram of text lengths
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df["text_length"], bins=30, kde=True, color="teal")
    ax.set_title("Belge Uzunluklarının Dağılımı")
    ax.set_xlabel("Kelime Sayısı")
    ax.set_ylabel("Frekans")
    st.pyplot(fig)

    # Most common words
    from collections import Counter
    all_words = [word for text in texts for word in text.split()]
    word_freq = Counter(all_words)
    common_words = word_freq.most_common(20)

    st.markdown("**En Yaygın 20 Kelime:**")
    common_df = pd.DataFrame(common_words, columns=["Word", "Frequency"])
    st.dataframe(common_df)

    # Word cloud
    st.markdown("**Sık Kullanılan Kelimelerin Kelime Bulutu:**")
    wordcloud = WordCloud(width=1000, height=400, background_color='white').generate_from_frequencies(word_freq)
    fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    # If date column exists, show number of documents over time
    if date_column:
        st.markdown("**🗓️ Yılda Belge Sayısı:**")
        year_counts = df['year'].value_counts().sort_index()
        st.bar_chart(year_counts)

    st.markdown("---")
    st.markdown("✅ Verileri inceledikten sonra artık kenar çubuğundan bir konu modelleme algoritması seçip çalıştırabilirsiniz.")

if model_choice == "LDA":
    st.subheader("🔍 Gizli Dirichlet Tahsis Konu Modellemesi")
    st.write("Latent Dirichlet Allocation Topic Modeling (LDA)")
    # vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_output = lda_model.fit_transform(X)
    topic_df = pd.DataFrame(lda_output, columns=[f"T{i+1}" for i in range(n_topics)])
    st.dataframe(topic_df.style.background_gradient(cmap="Blues"))
    st.markdown(generate_download_link(topic_df), unsafe_allow_html=True)

    # lda_model.fit(doc_term_matrix)
    vocab = vectorizer.get_feature_names_out()
    topic_distributions = lda_model.transform(doc_term_matrix)

    st.subheader("🔠 LDA Kelime Bulutları ve Anahtar Sözcükler")
    for idx, topic in enumerate(lda_model.components_):
        with st.expander(f"Topic {idx}", expanded=False):
            words = [vocab[i] for i in topic.argsort()[:-11:-1]]
            st.write(", ".join(words))
            word_freq = {vocab[i]: topic[i] for i in topic.argsort()[:-11:-1]}
            wc = WordCloud(width=600, height=300, background_color='white').generate_from_frequencies(word_freq)
            st.image(wc.to_array())

    st.subheader("📊 Belgeye Göre Konu Dağılımı")
    st.write(pd.DataFrame(topic_distributions, columns=[f"T{i+1}" for i in range(n_topics)]))

    if date_column:
        if lda_output.shape[0] == len(df):
            df['topic_distr'] = lda_output.tolist()
            yearly_topics = df.groupby('year')['topic_distr'].apply(lambda x: np.mean(x.tolist(), axis=0))
            yearly_df = pd.DataFrame(yearly_topics.tolist(), index=yearly_topics.index, columns=[f"T{i+1}" for i in range(n_topics)])
            st.subheader("📆 Zaman İçinde Ortalama Konu Dağılımı")
            styled_line_chart(yearly_df)
        else:
            st.warning("⚠️ Konu dağılım satırları ile belge sayısı arasında uyumsuzluk var. Yıl bazında toplamayı yapamadı.")

        st.subheader("📊 Zaman İçinde Konu Dağılımının Yığılmış Çubuk Grafiği")
        fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
        yearly_df.plot(kind='bar', stacked=True, ax=ax_bar, colormap='tab20')
        ax_bar.set_ylabel("Ortalama Konu Oranı")
        ax_bar.set_xlabel("Yıl")
        ax_bar.set_title("Yıla Göre Yığılmış Ortalama Konu Dağılımı")
        st.pyplot(fig_bar)

    st.subheader("pyLDAVis Etkileşimli Görselleştirme")
    with st.spinner("pyLDAvis Oluşturuluyor..."):
        try:
            vis_data = pyLDAvis.lda_model.prepare(lda_model, doc_term_matrix, vectorizer, mds='pcoa')
            html_string = pyLDAvis.prepared_data_to_html(vis_data)
            components.html(html_string, width=1300, height=800, scrolling=True)
        except Exception as e:
            st.error(f"pyLDAVis görselleştirmesi oluşturulamadı. Hata: {e}")

    st.subheader("📈 Konu Korelasyon Isı Haritası")
    corr_matrix = np.corrcoef(topic_distributions.T)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f", xticklabels=[f"T{i+1}" for i in range(n_topics)], yticklabels=[f"T{i+1}" for i in range(n_topics)], ax=ax)
    st.pyplot(fig)

    st.subheader("🔗 Konu Korelasyon Ağı Grafiği")
    threshold = st.slider("Kenar Korelasyon Eşiği", 0.0, 1.0, 0.1, 0.01)
    G = nx.Graph()
    for i in range(n_topics):
        G.add_node(f"T{i+1}")
        for j in range(i + 1, n_topics):
            if abs(corr_matrix[i, j]) >= threshold:
                G.add_edge(f"T{i+1}", f"T{j+1}", weight=round(corr_matrix[i, j], 3))
    fig_net, ax_net = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]['weight'] for u, v in G.edges]

    if weights:
        nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", edge_color=weights, edge_cmap=plt.cm.Blues, width=2, ax=ax_net)
        # Draw edge weights
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
        sm.set_array([])
        fig_net.colorbar(sm, ax=ax_net, label='Korelasyon Ağırlığı')
        st.pyplot(fig_net)
    else:
        st.warning("⚠️ Seçilen eşiğin üzerinde konu korelasyonu yok. Eşiği düşürmeyi deneyin.")

    st.subheader("🧬 Konuların Hiyerarşik Kümelenmesi")
    distance_metric = st.selectbox("Mesafe Metriği", ["euclidean", "cosine"], index=1)
    linkage_method = st.selectbox("Bağlantı Yöntemi", ["ward", "complete", "average", "single"], index=2)
    fig_dendro, ax_dendro = plt.subplots(figsize=(10, 5))
    dist_matrix = pdist(topic_distributions.T, metric=distance_metric)
    linkage_matrix = sch.linkage(dist_matrix, method=linkage_method)
    sch.dendrogram(linkage_matrix, labels=[f"T{i+1}" for i in range(n_topics)], ax=ax_dendro)
    st.pyplot(fig_dendro)

elif model_choice == "NMF":
    st.subheader("🔍 Negatif Olmayan Matris Faktörizasyonu (NMF) Konu Modelleme")
    st.write("Non-negative Matrix Factorization (NMF) Topic Modeling")
    nmf_model = NMF(n_components=n_topics, random_state=42)
    nmf_output = nmf_model.fit_transform(X)
    topic_df = pd.DataFrame(nmf_output, columns=[f"T{i+1}" for i in range(n_topics)])
    st.dataframe(topic_df.style.background_gradient(cmap="Purples"))
    st.markdown(generate_download_link(topic_df), unsafe_allow_html=True)

    tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(texts)
    nmf = NMF(n_components=n_topics, random_state=42)
    nmf.fit(tfidf_matrix)
    vocab = tfidf.get_feature_names_out()
    topic_distributions = nmf.transform(tfidf_matrix)

    for idx, topic in enumerate(nmf.components_):
        with st.expander(f"Topic {idx}", expanded=False):
            words = [vocab[i] for i in topic.argsort()[:-11:-1]]
            st.write(", ".join(words))
            word_freq = {vocab[i]: topic[i] for i in topic.argsort()[:-11:-1]}
            wc = WordCloud(width=600, height=300, background_color='white').generate_from_frequencies(word_freq)
            st.image(wc.to_array())

    st.subheader("📊 Belgeye Göre Konu Dağılımı")
    st.write(pd.DataFrame(topic_distributions, columns=[f"T{i+1}" for i in range(n_topics)]))

    if date_column:
        if nmf_output.shape[0] == len(df):
            df['topic_distr'] = nmf_output.tolist()
            yearly_topics = df.groupby('year')['topic_distr'].apply(lambda x: np.mean(x.tolist(), axis=0))
            yearly_df = pd.DataFrame(yearly_topics.tolist(), index=yearly_topics.index, columns=[f"T{i+1}" for i in range(n_topics)])
            st.subheader("📆 Zaman İçinde Ortalama Konu Dağılımı")
            styled_line_chart(yearly_df)
        else:
            st.warning("⚠️ Konu dağılım satırları ile belge sayısı arasında uyumsuzluk var. Yıl bazında toplamayı yapamadı.")

elif model_choice == "BERTopic":
    st.subheader("🔍 BERTopic")
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(texts)
    topic_model.reduce_topics(texts, nr_topics=n_topics)

    st.write(topic_model.get_topic_info().head(n_topics))
    fig = topic_model.visualize_barchart(top_n_topics=n_topics)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔠 BERTopic Kelime Bulutları ve Anahtar Sözcükler")
    for topic_id in topic_model.get_topic_freq().head(n_topics)['Topic']:
        if topic_id == -1:
            continue
        with st.expander(f"Topic {topic_id}", expanded=False):
            words_scores = topic_model.get_topic(topic_id)
            keywords = [word for word, _ in words_scores[:10]]
            st.markdown(f"**Top Keywords:** {', '.join(keywords)}")
            word_freq = {word: score for word, score in words_scores[:30]}
            wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
            st.image(wc.to_array(), use_container_width=True)

    # st.subheader("📊 Topic Distribution by Document")
    topic_distr = topic_model.transform(texts)[1]
    # topic_df = pd.DataFrame(topic_distr, columns=[f"T{i+1}" for i in range(topic_distr[0])])
    # st.dataframe(topic_df.style.background_gradient(cmap="Oranges"))
    #st.markdown(generate_download_link(topic_df), unsafe_allow_html=True)

    if date_column:
        if topic_distr.shape[0] == len(df):
            topics_over_time = topic_model.topics_over_time(texts, df["year"])
            fig = topic_model.visualize_topics_over_time(topics_over_time)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Konu dağılım satırları ile belge sayısı arasında uyumsuzluk var. Yıl bazında toplamayı yapamadı.")

    st.subheader("📌 BERTopic Konulararası Mesafe Haritası")
    st.plotly_chart(topic_model.visualize_topics(), use_container_width=True)

    st.subheader("🌿 BERTopic Hiyerarşi Görselleştirmesi")
    hierarchical_topics = topic_model.hierarchical_topics(texts)
    st.plotly_chart(topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics), use_container_width=True)

    st.subheader("🌿 BERTopic Isı Haritası Görselleştirmesi")
    fig = topic_model.visualize_heatmap()
    st.plotly_chart(fig, use_container_width=True)

# --- STM INTEGRATION (R-based) ---
if model_choice == "STM (R)":
    if not rpy2_available:
        st.error("rpy2 is not available. Please install it to run STM.")
        st.stop()

    df[text_column].to_csv("stm_input.csv", index=False)
    if date_column:
        df[[date_column]].to_csv("stm_meta.csv", index=False)

    st.info("Running Structured Topic Modeling (STM) using R...")

    try:
        robjects.r('''
        library(stm)
        library(tm)
        library(SnowballC)
        library(ggplot2)
        library(quanteda)

        texts <- read.csv("stm_input.csv", stringsAsFactors = FALSE)$text
        metadata <- read.csv("stm_meta.csv", stringsAsFactors = FALSE)

        processed <- textProcessor(texts, metadata = metadata)
        out <- prepDocuments(processed$documents, processed$vocab, processed$meta)

        model <- stm(out$documents, out$vocab, K=9, data=out$meta, init.type="Spectral")

        png("stm_summary.png", width=1000, height=800)
        plot(model, type="summary")
        dev.off()
        ''')

        st.subheader("📈 STM Summary Plot (from R)")
        st.image("stm_summary.png")
    except Exception as e:
        st.error(f"STM modeling failed: {e}")
    st.stop()
# --- END STM HANDLING ---

st.markdown("---")
st.markdown("© Powered by Streamlit and Scikit-learn")
