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

# 2. INISIASI MESIN (Menggunakan cache agar tidak perlu diload berulang kali)
@st.cache_resource
def load_nlp_engines():
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    # Stopwords Tambahan
    with open('data/tambahan_stopwords.txt', 'r') as f:
        custom_stopwords = set(line.strip().lower() for line in f if line.strip())
    stemmer = StemmerFactory().create_stemmer()
    downloader = YoutubeCommentDownloader()
    return stopword_remover, custom_stopwords , stemmer, downloader

stopword_remover, custom_stopwords, stemmer, downloader = load_nlp_engines()

@st.cache_data
def load_kamus():
    try:
        df_kamus = pd.read_csv('data/kamus_slang.csv')
        return dict(zip(df_kamus['singkatan'], df_kamus['asli']))
    except:
        return {} # Jika kamus tidak ditemukan, kembalikan dictionary kosong

kamus_slang = load_kamus()

# 3. FUNGSI ALGORITMA PREPROCESSING
def proses_teks_lengkap(teks):
    teks = str(teks).lower()
    teks = re.sub(r'http\S+|www\S+|https\S+', '', teks, flags=re.MULTILINE)
    teks = re.sub(r'[^a-z\s]', '', teks)
    
    words = teks.split()
    words_baku = [kamus_slang[kata] if kata in kamus_slang else kata for kata in words]
    
    # 1. Hapus Custom Stopwords terlebih dahulu
    words_tanpa_custom = [w for w in words_baku if w not in custom_stopwords]
    
    # 2. Gabungkan menjadi teks untuk diproses Sastrawi
    teks_gabungan = " ".join(words_tanpa_custom)
    
    # 3. Hapus Stopword bawaan Sastrawi
    teks_stopword = stopword_remover.remove(teks_gabungan)
    
    # 4. Stemming
    teks_final = stemmer.stem(teks_stopword)
    
    return teks_final

# 4. ANTARMUKA PENGGUNA (UI) UTAMA
st.subheader("1. Data Acquisition (Pengambilan Data)")
url_input = st.text_input("🔗 Masukkan Link Video YouTube:", placeholder="Contoh: https://www.youtube.com/watch?v=jpSuhrdhsKI")
jumlah_komentar = st.slider("Jumlah komentar yang ingin diambil (Maksimal 200 untuk efisiensi):", 10, 200, 100)

# Tombol Eksekusi Utama
if st.button("🚀 Mulai Analisis (Scraping ➔ Preprocessing ➔ Visualisasi)", type="primary"):
    
    if url_input:
        try:
            # --- TAHAP A: SCRAPING ---
            with st.spinner("Sedang mengambil data komentar dari YouTube... (Membutuhkan koneksi internet)"):
                take_comments = downloader.get_comments_from_url(url_input)
                all_comments = []
                for i, comment in enumerate(take_comments):
                    if i >= jumlah_komentar:
                        break
                    text = comment['text'].strip()
                    if text:
                        all_comments.append(text)
                
                df = pd.DataFrame({'text_mentah': all_comments})
            
            st.success(f"Berhasil mengambil {len(df)} komentar!")
            st.dataframe(df.head(), use_container_width=True)
            
            # --- TAHAP B: PREPROCESSING ---
            st.markdown("---")
            st.subheader("2. Text Preprocessing (Algoritma Inti DAA)")
            with st.spinner("Mesin NLP sedang membersihkan teks kata demi kata..."):
                df['teks_bersih'] = df['text_mentah'].apply(proses_teks_lengkap)
                
            st.success("Preprocessing Selesai!")
            st.dataframe(df[['text_mentah', 'teks_bersih']], use_container_width=True)
            
            # --- TAHAP C: VISUALISASI ---
            st.markdown("---")
            st.subheader("3. Hasil Analisis Visual")
            
            # Menggabungkan semua teks bersih menjadi satu string panjang
            semua_kata = " ".join(df['teks_bersih'].tolist())
            
            if semua_kata.strip(): # Pastikan ada kata yang tersisa
                col1, col2 = st.columns(2)
                
                # Visualisasi 1: WordCloud (Kiri)
                with col1:
                    st.markdown("**☁️ Word Cloud (Kata Terpopuler)**")
                    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(semua_kata)
                    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                    ax_wc.imshow(wordcloud, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
                
                # Visualisasi 2: Bar Chart (Kanan)
                with col2:
                    st.markdown("**📊 Top 10 Kata Terbanyak (Bar Chart)**")
                    # Menghitung frekuensi kata pakai struktur data Counter
                    kata_list = semua_kata.split()
                    frekuensi = Counter(kata_list)
                    top_10 = frekuensi.most_common(10)
                    
                    df_top = pd.DataFrame(top_10, columns=['Kata', 'Jumlah'])
                    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
                    ax_bar.bar(df_top['Kata'], df_top['Jumlah'], color='skyblue')
                    ax_bar.set_ylabel("Frekuensi Kemunculan")
                    plt.xticks(rotation=45)
                    st.pyplot(fig_bar)
            else:
                st.warning("Teks terlalu pendek atau habis terhapus oleh Stopword/Regex.")
            
            # --- TAHAP D: DOWNLOAD ---
            st.markdown("---")
            # Memilih hanya kolom 'teks_bersih' sebelum diekspor
            csv = df[['teks_bersih']].to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download Clean Dataset (.csv)",
                data=csv,
                file_name='dataset_youtube_terbaru.csv',
                mime='text/csv',
            )
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses link: {e}")
    else:
        st.warning("⚠️ Silakan masukkan link YouTube terlebih dahulu!")