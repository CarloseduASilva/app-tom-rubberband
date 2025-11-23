import streamlit as st
import tempfile
import os
import numpy as np
import time

# --- Importações com tratamento de erro ---
try:
    import soundfile as sf
except ImportError:
    st.error("Biblioteca 'soundfile' faltando. Instale: pip install soundfile")
    st.stop()

try:
    import pyrubberband as rb
except ImportError:
    st.error("Biblioteca 'pyrubberband' faltando. Instale: pip install pyrubberband")
    st.stop()

# Importação condicional do librosa 
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# --- Funções Auxiliares ---
@st.cache_data
def obter_dicionario_notas():
    notas = ["C (Dó)", "C# / Db", "D (Ré)", "D# / Eb", "E (Mi)", 
             "F (Fá)", "F# / Gb", "G (Sol)", "G# / Ab", "A (Lá)", "A# / Bb", "B (Si)"]
    return {nota: i for i, nota in enumerate(notas)}

def calcular_semitons(origem, destino, oitavas):
    mapa = obter_dicionario_notas()
    return (mapa[destino] - mapa[origem]) + (oitavas * 12)

def detectar_tonica(y, sr):
    if not LIBROSA_AVAILABLE:
        return 0
    y_harmonic = librosa.effects.harmonic(y)
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    chroma_vals = np.sum(chromagram, axis=1)
    return int(np.argmax(chroma_vals))

# --- Configuração da Página ---
st.set_page_config(page_title="Alterador de Tom", page_icon="🎵", layout="centered")
st.title("Alterador de Tom com Preview")

# --- Opções de Upload ---
col_up_1, col_up_2 = st.columns([3, 1])

with col_up_1:
    arquivo = st.file_uploader("Carregue sua música", type=["mp3", "wav", "ogg", "flac"])

with col_up_2:
    st.write("") 
    st.write("") 
    # Checkbox para ativar a detecção
    detectar_auto = st.checkbox(
        "Detectar tom?", 
        value=False, 
        help="Beta: Tenta descobrir o tom da música. Pode demorar alguns segundos e não é 100% preciso."
    )

if arquivo:
    # --- Lógica de Carregamento ---
    novo_arquivo = "arquivo_atual" not in st.session_state or st.session_state.arquivo_atual != arquivo.name
    mudou_opcao_detectar = "detectar_ativo" not in st.session_state or st.session_state.detectar_ativo != detectar_auto

    if novo_arquivo or (mudou_opcao_detectar and detectar_auto):
        
        texto_loading = "Carregando áudio..."
        if detectar_auto:
            texto_loading = "Carregando e analisando tom (isso pode demorar um pouco)..."

        with st.spinner(texto_loading):
            if novo_arquivo:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(arquivo.getvalue())
                    st.session_state.path_original = tmp.name
                
                st.session_state.arquivo_atual = arquivo.name
                # Lê o áudio para memória
                y, sr = sf.read(st.session_state.path_original)
                st.session_state.y = y
                st.session_state.sr = sr

                # Prepara o slice de preview (15s)
                duration = len(y) / sr
                start_preview = min(30, duration / 2)
                end_preview = min(start_preview + 15, duration)
                st.session_state.y_preview = y[int(start_preview*sr):int(end_preview*sr)]

            # Lógica de Detecção de Tom 
            if detectar_auto:
                if not LIBROSA_AVAILABLE:
                    st.session_state.tom_detectado_idx = 0
                else:
                    y_analise = st.session_state.y[:st.session_state.sr * 60] # Primeiros 60s
                    
                    if len(y_analise.shape) > 1:
                        y_analise = np.mean(y_analise, axis=1) # Mono
                        
                    idx = detectar_tonica(y_analise, st.session_state.sr)
                    st.session_state.tom_detectado_idx = idx
                    st.toast(f"Tom estimado detectado.", icon="🎹")
            else:
                if "tom_detectado_idx" not in st.session_state:
                     st.session_state.tom_detectado_idx = 0 # C

            st.session_state.detectar_ativo = detectar_auto

    st.write("---")
    
    # --- Controles ---
    col1, col2, col3 = st.columns([2, 2, 1.5])
    notas_lista = list(obter_dicionario_notas().keys())
    
    with col1:
        # Define o índice padrão baseado na detecção (se houve)
        idx_padrao = st.session_state.get("tom_detectado_idx", 0)
        
        # Se a detecção não foi usada, o índice será 0 (C)
        label_tom = "Tom Original"
        if detectar_auto:
            label_tom = "Tom Original (Estimado)"

        de = st.selectbox(label_tom, notas_lista, index=idx_padrao)
        
    with col2:
        para = st.selectbox("Tom Desejado", notas_lista, index=0) # Padrão C
        
    with col3:
        oitava = st.number_input("Oitavas", -2, 2, 0)

    # --- Feedback de Detecção (Apenas se ativado) ---
    if detectar_auto:
        st.caption("⚠️ A detecção automática pode falhar em músicas complexas. Ajuste o 'Tom Original' se necessário.")

    # --- Lógica de Cálculo e Preview ---
    semitons = calcular_semitons(de, para, oitava)
    
    if semitons != 0:
        st.info(f"Mudança: **{semitons:+} semitons**")
    else:
        st.info("Tom original mantido.")

    st.markdown("Preview (Trecho de 15s)")
    
    if semitons == 0:
        path_preview = "preview_original.wav"
        sf.write(path_preview, st.session_state.y_preview, st.session_state.sr)
        st.audio(path_preview)
    else:
        with st.spinner("Gerando preview..."):
            try:
                y_prev_shifted = rb.pitch_shift(
                    st.session_state.y_preview, 
                    st.session_state.sr, 
                    n_steps=semitons,
                    rbargs={'--realtime': '', '--formant': ''}
                )
                
                max_val = np.max(np.abs(y_prev_shifted))
                if max_val > 1.0: y_prev_shifted = y_prev_shifted / max_val * 0.99
                
                sf.write("preview_temp.wav", y_prev_shifted, st.session_state.sr)
                st.audio("preview_temp.wav")
            except Exception as e:
                st.error(f"Erro no preview: {e}")

    st.write("---")

    # --- Processamento Final ---
    st.markdown("### 💾 Finalizar e Baixar")
    
    if st.button("Processar Música Inteira", type="primary"):
        progresso = st.progress(0)
        status = st.empty()
        
        try:
            status.text("Processando áudio completo...")
            
            rb_args_final = {'--formant': '', '--fine': ''}
            
            y_final = rb.pitch_shift(
                st.session_state.y,
                st.session_state.sr, 
                n_steps=semitons,
                rbargs=rb_args_final
            )
            progresso.progress(80)
            
            status.text("Finalizando arquivo...")
            max_val = np.max(np.abs(y_final))
            if max_val > 1.0: y_final = y_final / max_val * 0.99
            
            output_filename = f"musica_final_{semitons}st.wav"
            sf.write(output_filename, y_final, st.session_state.sr)
            
            progresso.progress(100)
            status.success("Concluído!")
            
            with open(output_filename, "rb") as f:
                st.download_button(
                    label="⬇️ Baixar WAV",
                    data=f,
                    file_name=output_filename,
                    mime="audio/wav"
                )
                
        except Exception as e:
            st.error(f"Erro no processamento final: {e}")
