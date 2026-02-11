import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Predição de Mercado - Portfólio ML", layout="wide", initial_sidebar_state="expanded")

# --- ESTILOS CSS PREMIUM (GLASSMORPHISM & NEON) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    /* Dark Mode Global */
    .main {
        background: #050505 !important;
        color: #ffffff !important;
    }
    
    .stApp {
        background: #050505 !important;
        color: #ffffff !important;
    }

    /* Forcing white color on all standard elements */
    p, span, div, label, li, .stMarkdown {
        color: #ffffff !important;
    }

    /* Top Control Bar */
    .control-panel {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
    }

    /* Headers - VISUAL IMPACT RED */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #ff3131 !important;
        background: none !important;
        -webkit-text-fill-color: initial !important;
        text-shadow: 0 0 20px rgba(255, 49, 49, 0.6);
        font-weight: 900 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #b30000, #ff3131) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(255, 49, 49, 0.4) !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #ffffff !important; /* Mudei para branco para melhor leitura contra o título vermelho */
        font-family: 'Orbitron', sans-serif;
        text-shadow: 0 0 10px rgba(255,255,255,0.2);
    }
    [data-testid="stMetricLabel"] {
        color: #ff3131 !important;
        font-weight: bold !important;
    }

    /* Data Table Styling */
    .stTable td, .stTable th {
        color: #ffffff !important;
        background-color: #111111 !important;
        border-bottom: 1px solid #333333 !important;
    }
    .stTable {
        background-color: #111111 !important;
        border: 1px solid #333333 !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid rgba(255, 49, 49, 0.2);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    /* Custom Sidebar Branding Box */
    .sidebar-branding {
        text-align: center; 
        padding: 15px; 
        border: 2px solid #ff3131; 
        border-radius: 12px; 
        background: rgba(255, 49, 49, 0.1);
        box-shadow: 0 0 15px rgba(255, 49, 49, 0.2);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNÇÕES DE GERAÇÃO E MODELAGEM (MANTIDAS) ---
def gerar_dados_ficticios(dias=120):
    np.random.seed(st.session_state.get('seed', 42))
    datas = pd.date_range(end=datetime.datetime.now(), periods=dias)
    t = np.arange(dias)
    # Variando a tendência e ruído para cada geração
    noise_lvl = st.session_state.get('noise_lvl', 5)
    tendencia = (np.random.uniform(0.2, 0.8)) * t + 10 
    sazonalidade = 12 * np.sin(2 * np.pi * t / 25) 
    ruido = np.random.normal(0, noise_lvl, dias)
    precos = tendencia + sazonalidade + ruido
    
    # Gerando Volume de Vendas (Algo entre 500 e 2000 com variação)
    volumes = np.random.randint(500, 2500, size=dias) + (precos * 0.5).astype(int)
    
    return pd.DataFrame({'Data': datas, 'Preco': precos, 'Volume': volumes})

def treinar_modelo_polinomial(df, grau=3, dias_previsao=15):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Preco'].values
    poly = PolynomialFeatures(degree=grau)
    X_poly = poly.fit_transform(X)
    modelo = LinearRegression()
    modelo.fit(X_poly, y)
    y_pred = modelo.predict(X_poly)
    X_futuro = np.arange(len(df), len(df) + dias_previsao).reshape(-1, 1)
    y_futuro = modelo.predict(poly.transform(X_futuro))
    datas_futuras = pd.date_range(start=df['Data'].iloc[-1] + datetime.timedelta(days=1), periods=dias_previsao)
    df_futuro = pd.DataFrame({'Data': datas_futuras, 'Preco_Pred': y_futuro})
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    return y_pred, df_futuro, rmse, mae

# --- LAYOUT PRINCIPAL REORGANIZADO ---
st.title("💎 Market Predictive Analytics Pro")

# Barra de Controles Superior
with st.container():
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    
    with c1:
        if st.button('🔄 Gerar Novos Dados'):
            st.session_state.seed = np.random.randint(0, 10000)
            st.session_state.noise_lvl = np.random.randint(3, 10)
    
    with c2:
        st.markdown("**🎚️ Complexidade do Modelo (Grau Polinomial)**")
        grau_polinomio = st.select_slider(
            "Slide para ajustar",
            options=[1, 2, 3, 4, 5],
            value=3,
            label_visibility="collapsed"
        )
    
    with c3:
        st.write("") # Espaçador
        st.markdown(f"**Status:** Modelando grau {grau_polinomio}")
    st.markdown('</div>', unsafe_allow_html=True)

# Lógica de Dados
df = gerar_dados_ficticios()
y_pred, df_futuro, rmse, mae = treinar_modelo_polinomial(df, grau=grau_polinomio)

# Área de Visualização
# Tabela Compacta - agora com estilo branco
with st.expander("📊 Ver Dados Brutos (Últimos dias)", expanded=False):
    df_display = df.tail(10).copy()
    df_display['Data'] = df_display['Data'].dt.strftime('%d/%m/%Y')
    df_display['Preco'] = df_display['Preco'].map('R$ {:,.2f}'.format)
    df_display['Volume'] = df_display['Volume'].map('{:,.0f}'.format).str.replace(',', '.')
    # Renomeando para exibição
    df_display.columns = ['Data', 'Preço Unitário', 'Volume de Vendas']
    st.table(df_display)

# Gráfico Principal - Configuração de cores forçada para branco
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Data'], y=df['Preco'], mode='lines', name='Histórico', line=dict(color='#00d1ff', width=2)))
fig.add_trace(go.Scatter(x=df['Data'], y=y_pred, mode='lines', name='Tendência', line=dict(color='#ff3131', width=1, dash='dot'))) # Vermelho para tendência
fig.add_trace(go.Scatter(x=df_futuro['Data'], y=df_futuro['Preco_Pred'], mode='lines', name='Projeção', line=dict(color='#00ff88', width=4)))

fig.update_layout(
    template="plotly_dark",
    height=500,
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color="#ffffff"), # Texto do gráfico em branco
    xaxis=dict(
        showgrid=True, 
        gridcolor='rgba(255,255,255,0.05)', 
        tickfont=dict(color='#ffffff') # Eixos em branco
    ),
    yaxis=dict(
        showgrid=True, 
        gridcolor='rgba(255,255,255,0.05)', 
        tickprefix="R$ ", 
        tickfont=dict(color='#ffffff') # Eixos em branco
    ),
    legend=dict(orientation="h", y=1.1, font=dict(color="#ffffff"))
)
st.plotly_chart(fig, use_container_width=True)

# Métricas e Análise
st.markdown("### 📊 INTELIGÊNCIA DO MODELO")
m1, m2, m3 = st.columns(3)
with m1: st.metric("RMSE", f"{rmse:.2f}")
with m2: st.metric("MAE", f"{mae:.2f}")
with m3: 
    trend = "ALTA" if df_futuro['Preco_Pred'].iloc[-1] > df_futuro['Preco_Pred'].iloc[0] else "BAIXA"
    st.metric("Tendência Projetada", trend)

st.divider()

# Formatando valores para o padrão brasileiro (vírgula como decimal) nos insights
mae_br = f"{mae:.2f}".replace('.', ',')
rmse_br = f"{rmse:.2f}".replace('.', ',')

st.markdown(f"""
### 📓 INSIGHTS ESTRATÉGICOS
A aplicação da **Regressão de Grau {grau_polinomio}** revela um comportamento de **{trend}**.
- **Ajuste:** O erro médio absoluto ({mae_br}) indica que o modelo ignora flutuações irrelevantes para focar na trajetória estrutural.
- **Projeção:** A curvatura atual sugere uma {'aceleração' if grau_polinomio > 1 else 'continuidade'} dos preços para o próximo quindênio.
""")

# --- SIDEBAR MARKETING & BRANDING ---
st.sidebar.markdown(f"""
<div class="sidebar-branding">
    <h2 style="color: #ff3131; margin-bottom: 0; font-size: 1.4rem;">TECSOLUTIONS</h2>
    <p style="color: #ffffff; font-size: 0.85rem; font-weight: bold; margin-top: 5px;">Inovação em Inteligência de Dados</p>
</div>

---

### 🚀 Nossos Serviços
- **🤖 Modelagem de ML Sênior**: Algoritmos customizados para predição de alta precisão.
- **📊 Business Intelligence**: Dashboards interativos e visualização de dados dinâmica.
- **💻 Engenharia de Software**: Desenvolvimento de aplicações web escaláveis e seguras.
- **📐 Consultoria Matemática**: Modelos matemáticos avançados para problemas complexos.

---

### 💼 Por que a TecSolutions?
Aliamos rigor matemático com design de ponta para entregar soluções que não apenas funcionam, mas impressionam. Este projeto é uma pequena amostra do nosso compromisso com a **excelência técnica**.

---

<p style="text-align: center; color: #ff3131; font-weight: bold; font-size: 1.1rem;">
    🚀 Desenvolvido por TecSolutions
</p>
""", unsafe_allow_html=True)

st.sidebar.info("Este é apenas um aplicativo de demonstração. Os dados apresentados são simulados aleatoriamente; em uma situação real, os dados seriam extraídos diretamente de planilhas ou bancos de dados do cliente.")
