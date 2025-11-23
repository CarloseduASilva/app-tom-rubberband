import streamlit as st
import tempfile
import os
import numpy as np

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

# --- Configuração da Página ---
st.set_page_config(page_title="Alterador de Tom", page_icon="🎵", layout="centered")

# --- Funções Auxiliares ---
@st.cache_data
def obter_dicionario_notas():
    notas = ["C (Dó)", "C# / Db", "D (Ré)", "D# / Eb", "E (Mi)", 
             "F (Fá)", "F# / Gb", "G (Sol)", "G# / Ab", "A (Lá)", "A# / Bb", "B (Si)"]
    return {nota: i for i, nota in enumerate(notas)}

def calcular_semitons(origem, destino, oitavas):
    mapa = obter_dicionario_notas()
    return (mapa[destino] - mapa[origem]) + (oitavas * 12)

# --- Interface ---
st.title("Alterador de Tom PRO 2025")
st.markdown("### Com Rubberband")

arquivo = st.file_uploader("Escolha sua música (MP3, WAV, OGG, FLAC)", 
                          type=["mp3", "wav", "ogg", "flac", "m4a"])

if arquivo:
    extensao = os.path.splitext(arquivo.name)[1].lower()
    if not extensao: extensao = ".wav"
    
    # Salva input
    with tempfile.NamedTemporaryFile(delete=False, suffix=extensao) as tmp_in:
        tmp_in.write(arquivo.getvalue())
        input_path = tmp_in.name

    st.audio(input_path)
    st.write("---")
    
    # Layout de controles
    col1, col2, col3 = st.columns([2, 2, 1.5])
    notas = list(obter_dicionario_notas().keys())
    
    with col1:
        de = st.selectbox("Tom original", notas, index=9)  # A
    with col2:
        para = st.selectbox("Tom desejado", notas, index=0)  # C
    with col3:
        oitava = st.number_input("Oitavas", -2, 2, 0)
    
    # Opções Avançadas
    with st.expander("⚙️ Configurações Avançadas", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            preservar_formantes = st.checkbox("Preservar voz natural (Formantes)", value=True)
        with col_b:
            # Simplificado: Removemos o modo "Alta Qualidade" que era muito lento
            qualidade = st.radio("Modo de Processamento", ["Padrão (Recomendado)", "Rápido (Preview)"], index=0)

    semitons = calcular_semitons(de, para, oitava)

    # Feedback Visual
    if semitons == 0:
        st.info("O tom de origem e destino são iguais.")
    else:
        direcao = "Subindo ⬆️" if semitons > 0 else "Descendo ⬇️"
        st.success(f"{direcao} **{abs(semitons)} semitons**")

        if st.button("Processar Áudio 🚀", type="primary", use_container_width=True):
            
            progresso = st.progress(0)
            status = st.empty()
            
            try:
                # 1. Carregar
                status.text("Carregando áudio...")
                y, sr = sf.read(input_path, always_2d=True)
                progresso.progress(25)
                
                # 2. Configurar Argumentos do Rubberband
                argumentos_rb = {}
                
                if preservar_formantes:
                    argumentos_rb['--formant'] = ''
                
              
                if "Rápido" in qualidade:
                    argumentos_rb['--realtime'] = ''
                
                # 3. Processar
                status.text(f"Processando (Modo {qualidade.split(' ')[0]})...")
                
                y_shifted = rb.pitch_shift(
                    y,
                    sr,
                    n_steps=semitons,
                    rbargs=argumentos_rb
                )
                progresso.progress(75)

                # 4. Pós-processamento 
                max_val = np.max(np.abs(y_shifted))
                if max_val > 1.0:
                    y_shifted = y_shifted / max_val * 0.99
                
                # 5. Salvar
                status.text("Salvando resultado...")
                suffix_out = ".wav"
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix_out) as tmp_out:
                    output_path = tmp_out.name
                
                sf.write(output_path, y_shifted, sr)
                
                progresso.progress(100)
                status.empty()
                st.balloons()

                # 6. Download
                col_res_1, col_res_2 = st.columns([3, 1])
                with col_res_1:
                    st.markdown("### Resultado Final:")
                    st.audio(output_path)
                
                with col_res_2:
                    st.write("") 
                    st.write("")
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Baixar WAV",
                            data=f,
                            file_name=f"musica_{semitons:+}st.wav",
                            mime="audio/wav"
                        )
                        
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")
                st.warning("Verifique se o 'rubberband.exe' e as DLLs estão na pasta Scripts do .venv")
            
            finally:

                pass
