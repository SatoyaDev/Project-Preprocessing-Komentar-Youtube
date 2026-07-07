import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from youtube_comment_downloader import YoutubeCommentDownloader
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# 1. KONFIGURASI HALAMAN WEB
st.set_page_config(page_title="PREPROCESSING KOMENTAR YOUTUBE", layout="wide", page_icon="🎥")
st.title("🎥 Aplikasi Text Preprocessing & Analisis Komentar YouTube")
st.markdown("**Proyek UAS Desain dan Analisis Algoritma (DAA) BY KELOMPOK 2**")
st.markdown("---")

# 2. INISIASI MESIN
@st.cache_resource
def load_nlp_engines():
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    # Membaca Stopwords Tambahan
    with open('data/tambahan_stopwords.txt', 'r') as f:
        custom_stopwords = set(line.strip().lower() for line in f if line.strip())
    stemmer = StemmerFactory().create_stemmer()
    downloader = YoutubeCommentDownloader()
    return stopword_remover, custom_stopwords, stemmer, downloader

stopword_remover, custom_stopwords, stemmer, downloader = load_nlp_engines()

@st.cache_data
def load_kamus():
    # Membaca langsung dari folder data
    df_kamus = pd.read_csv('data/kamus_slang.csv')
    return dict(zip(df_kamus['singkatan'], df_kamus['asli']))

kamus_slang = load_kamus()

# 3. FUNGSI ALGORITMA PREPROCESSING
def proses_teks_lengkap(teks):
    teks = str(teks).lower()
    teks = re.sub(r'http\S+|www\S+|https\S+', '', teks, flags=re.MULTILINE)
    teks = re.sub(r'[^a-z\s]', '', teks)
    
    words = teks.split()
    # Normalisasi Slang
    words_baku = [kamus_slang[kata] if kata in kamus_slang else kata for kata in words]
    
    # Hapus Custom Stopwords
    words_tanpa_custom = [w for w in words_baku if w not in custom_stopwords]
    
    # Hapus Stopwords Sastrawi & Stemming
    teks_gabungan = " ".join(words_tanpa_custom)
    teks_stopword = stopword_remover.remove(teks_gabungan)
    teks_final = stemmer.stem(teks_stopword)
    
    return teks_final

# 4. ANTARMUKA PENGGUNA (UI)
st.subheader("1. Data Acquisition")
url_input = st.text_input("🔗 Masukkan Link Video YouTube:")
jumlah_komentar = st.slider("Jumlah komentar:", 10, 200, 100)

if st.button("🚀 Mulai Analisis"):
    if url_input:
        try:
            with st.spinner("Mengambil data..."):
                take_comments = downloader.get_comments_from_url(url_input)
                all_comments = [c['text'].strip() for i, c in enumerate(take_comments) if i < jumlah_komentar and c['text'].strip()]
                df = pd.DataFrame({'text_mentah': all_comments})
            
            with st.spinner("Preprocessing..."):
                df['teks_bersih'] = df['text_mentah'].apply(proses_teks_lengkap)
            
            st.success("Selesai!")
            st.dataframe(df[['text_mentah', 'teks_bersih']])
            
            # Visualisasi
            semua_kata = " ".join(df['teks_bersih'].tolist())
            col1, col2 = st.columns(2)
            
            with col1:
                wc = WordCloud(width=800, height=400, background_color='white').generate(semua_kata)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                
            with col2:
                top_10 = Counter(semua_kata.split()).most_common(10)
                df_top = pd.DataFrame(top_10, columns=['Kata', 'Jumlah'])
                fig, ax = plt.subplots()
                ax.bar(df_top['Kata'], df_top['Jumlah'])
                plt.xticks(rotation=45)
                st.pyplot(fig)

            # Download Hanya Kolom Teks Bersih
            csv = df[['teks_bersih']].to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 Download Clean Dataset", csv, 'dataset_youtube.csv', 'text/csv')
            
        except Exception as e:
            st.error(f"Error: {e}")