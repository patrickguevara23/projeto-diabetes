"""
Passo 2 — Interface web do modelo SVC.
Rodar localmente:  streamlit run app.py
"""
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title='Avaliação de Risco de Diabetes',
    page_icon='🩺',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ---------------------------------------------------------------------------
# Rótulos amigáveis para os campos.
# Ajuste as chaves para os nomes EXATOS das colunas do seu df_limpo.
# Colunas que não estiverem aqui aparecem com o nome original formatado.
# ---------------------------------------------------------------------------
ROTULOS = {
    # Variáveis Base
    'Sexo': ('Sexo', '-', 'Gênero do paciente'),
    'Diabetes_familiar': ('Histórico Familiar de Diabetes', '-', 'Presença de diabetes na família'),
    'Pressão_alta': ('Pressão Alta', '-', 'Diagnóstico prévio de hipertensão'),
    'IMC': ('IMC', 'kg/m²', 'Índice de Massa Corporal'),
    'Fumante': ('Fumante', '-', 'Hábito de fumar'),
    'Ingere_alcool': ('Consumo de Álcool', '-', 'Hábito de ingerir bebidas alcoólicas'),
    'Medicamentos_regulares': ('Medicamentos Regulares', '-', 'Uso regular de medicamentos'),
    'Gradivez': ('Gravidez', '-', 'Histórico de gestação (grafia original: Gradivez)'),
    'Diabetes': ('Diabetes', '-', 'Diagnóstico de diabetes (Variável Alvo)'),

    # Faixa Etária (Idade)
    'Idade_less than 40': ('Idade: Menos de 40', '-', 'Faixa etária abaixo de 40 anos'),
    'Idade_40-49': ('Idade: 40 a 49', '-', 'Faixa etária entre 40 e 49 anos'),
    'Idade_50-59': ('Idade: 50 a 59', '-', 'Faixa etária entre 50 e 59 anos'),
    'Idade_60 or older': ('Idade: 60 ou mais', '-', 'Faixa etária de 60 anos ou superior'),

    # Atividade Física
    'Ativo_fisicamente_none': ('Atividade Física: Nenhuma', '-', 'Não pratica atividade física'),
    'Ativo_fisicamente_less than half an hr': ('Atividade Física: < 30 min', '-', 'Pratica menos de 30 minutos de exercício'),
    'Ativo_fisicamente_more than half an hr': ('Atividade Física: > 30 min', '-', 'Pratica mais de 30 minutos de exercício'),
    'Ativo_fisicamente_one hr or more': ('Atividade Física: 1h+', '-', 'Pratica uma hora ou mais de exercício'),

    # Alimentação (Junk Food)
    'JunkFood_occasionally': ('Alimentos Ultraprocessados: Ocasionalmente', '-', 'Consome fast food ocasionalmente'),
    'JunkFood_often': ('Alimentos Ultraprocessados: Frequentemente', '-', 'Consome fast food frequentemente'),
    'JunkFood_very often': ('Alimentos Ultraprocessados: Muito Frequentemente', '-', 'Consome fast food com muita frequência'),
    'JunkFood_always': ('Alimentos Ultraprocessados: Sempre', '-', 'Consome fast food sempre'),

    # Estresse
    'Stress_not at all': ('Estresse: Nenhum', '-', 'Nenhum nível de estresse relatado'),
    'Stress_sometimes': ('Estresse: Às vezes', '-', 'Nível de estresse ocasional'),
    'Stress_very often': ('Estresse: Muito Frequente', '-', 'Nível de estresse elevado e constante'),
    'Stress_always': ('Estresse: Sempre', '-', 'Nível de estresse contínuo'),

    # Pressão Sanguínea
    'Nível_pressão_sanguínea_low': ('Pressão: Baixa', '-', 'Nível de pressão arterial categorizado como baixo'),
    'Nível_pressão_sanguínea_normal': ('Pressão: Normal', '-', 'Nível de pressão arterial categorizado como normal'),
    'Nível_pressão_sanguínea_high': ('Pressão: Alta', '-', 'Nível de pressão arterial categorizado como alto'),

    # Pré-diabetes
    'Pre_diabetes_0': ('Pré-diabetes: Não', '-', 'Sem diagnóstico de pré-diabetes'),
    'Pre_diabetes_1': ('Pré-diabetes: Sim', '-', 'Com diagnóstico de pré-diabetes'),

    # Frequência Urinária
    'Frequência_urinária_not much': ('Frequência Urinária: Normal', '-', 'Frequência de urinar normal ou baixa'),
    'Frequência_urinária_quite often': ('Frequência Urinária: Alta', '-', 'Urinar com muita frequência (possível sintoma)'),
}

# ---------------------------------------------------------------------------
# Campos exibidos como opções de texto (Sim/Não, Feminino/Masculino).
# O texto aparece só na tela; o modelo continua recebendo o número (0 ou 1).
# ---------------------------------------------------------------------------
SIM_NAO = {'Não': 0, 'Sim': 1}

OPCOES = {
    # ATENÇÃO: confirme no seu df_limpo qual número representa cada sexo
    'Sexo': {'Feminino': 0, 'Masculino': 1},
    'Diabetes_familiar': SIM_NAO,
    'Pressão_alta': SIM_NAO,
    'Fumante': SIM_NAO,
    'Ingere_alcool': SIM_NAO,
    'Medicamentos_regulares': SIM_NAO,
}

# Variáveis que o one-hot dividiu em várias colunas.
# Na tela viram UMA pergunta; a resposta escolhida recebe 1 e as demais colunas, 0.
# 'padrao' (opcional) é a opção que já vem selecionada; sem ele, vem a primeira.
GRUPOS = {
    'Idade': {
        'rotulo': ('Faixa etária', 'Idade do paciente'),
        'opcoes': {
            'Menos de 40 anos': 'Idade_less than 40',
            '40 a 49 anos': 'Idade_40-49',
            '50 a 59 anos': 'Idade_50-59',
            '60 anos ou mais': 'Idade_60 or older',
        },
    },
    'Ativo_fisicamente': {
        'rotulo': ('Atividade física diária', 'Tempo de exercício físico por dia'),
        'opcoes': {
            'Nenhuma': 'Ativo_fisicamente_none',
            'Menos de 30 min': 'Ativo_fisicamente_less than half an hr',
            'Mais de 30 min': 'Ativo_fisicamente_more than half an hr',
            '1 hora ou mais': 'Ativo_fisicamente_one hr or more',
        },
    },
    'JunkFood': {
        'rotulo': ('Consumo de ultraprocessados', 'Frequência de consumo de fast food / ultraprocessados'),
        'opcoes': {
            'Ocasionalmente': 'JunkFood_occasionally',
            'Frequentemente': 'JunkFood_often',
            'Muito frequentemente': 'JunkFood_very often',
            'Sempre': 'JunkFood_always',
        },
    },
    'Stress': {
        'rotulo': ('Nível de estresse', 'Com que frequência o paciente se sente estressado'),
        'opcoes': {
            'Nenhum': 'Stress_not at all',
            'Às vezes': 'Stress_sometimes',
            'Muito frequente': 'Stress_very often',
            'Sempre': 'Stress_always',
        },
    },
    'Nível_pressão_sanguínea': {
        'rotulo': ('Nível de pressão arterial', 'Pressão arterial medida, em categorias'),
        'opcoes': {
            'Baixa': 'Nível_pressão_sanguínea_low',
            'Normal': 'Nível_pressão_sanguínea_normal',
            'Alta': 'Nível_pressão_sanguínea_high',
        },
        'padrao': 'Normal',
    },
    'Pre_diabetes': {
        'rotulo': ('Pré-diabetes', 'Diagnóstico prévio de pré-diabetes'),
        'opcoes': {'Não': 'Pre_diabetes_0', 'Sim': 'Pre_diabetes_1'},
    },
    'Frequência_urinária': {
        'rotulo': ('Frequência urinária', 'Com que frequência o paciente urina (urinar muito é um possível sintoma)'),
        'opcoes': {
            'Normal': 'Frequência_urinária_not much',
            'Alta': 'Frequência_urinária_quite often',
        },
    },
}
# Regra de negócio: Gravidez não se aplica ao sexo masculino
CAMPO_SEXO = 'Sexo'
CAMPO_GRAVIDEZ = 'Gradivez'

COLUNA_PARA_GRUPO = {col: g for g, d in GRUPOS.items() for col in d['opcoes'].values()}


def rotulo(col):
    if col in ROTULOS:
        nome, unidade, ajuda = ROTULOS[col]
        return nome, ('' if unidade == '-' else unidade), ajuda
    return (col.replace('_', ' ').capitalize(), '', '')


# ---------------------------------------------------------------------------
# Estilo
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero h1 {font-size: 2.1rem; margin-bottom: .25rem;}
.hero p  {color: #64748b; font-size: 1.05rem; margin-top: 0;}

.card {
    border: 1px solid rgba(100,116,139,.25);
    border-radius: 16px; padding: 1.5rem 1.6rem;
    background: rgba(148,163,184,.06);
}
.resultado {border-width: 2px;}
.resultado.alto  {border-color: #ef4444; background: rgba(239,68,68,.07);}
.resultado.baixo {border-color: #10b981; background: rgba(16,185,129,.07);}

.selo {display:inline-block; padding:.3rem .8rem; border-radius:999px;
       font-weight:600; font-size:.85rem; letter-spacing:.02em;}
.selo.alto  {background:#fee2e2; color:#b91c1c;}
.selo.baixo {background:#d1fae5; color:#047857;}

.titulo-resultado {font-size:1.6rem; font-weight:700; margin:.7rem 0 .2rem;}
.sub {color:#64748b; margin:0;}

.barra {height:14px; border-radius:999px; background:rgba(100,116,139,.18);
        overflow:hidden; margin:1.2rem 0 .4rem; position:relative;}
.barra > div {height:100%; border-radius:999px;}
.limiar {position:absolute; top:-4px; bottom:-4px; width:2px; background:#334155;}
.escala {display:flex; justify-content:space-between; color:#94a3b8; font-size:.8rem;}

.vazio {text-align:center; color:#94a3b8; padding:3rem 1rem;}
.vazio .icone {font-size:2.6rem;}
.aviso {font-size:.82rem; color:#94a3b8; margin-top:1rem;}

/* Esconde a dica "Press Enter to apply" que o Streamlit mostra nos campos */
div[data-testid="InputInstructions"] {display: none;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Carregamento do modelo
# ---------------------------------------------------------------------------
@st.cache_resource
def carregar():
    modelo = joblib.load('modelo.joblib')
    with open('info_modelo.json', encoding='utf-8') as f:
        info = json.load(f)
    return modelo, info


try:
    modelo, info = carregar()
except FileNotFoundError:
    st.error('Arquivos do modelo não encontrados. Rode `salvar_modelo.py` antes de abrir o app.')
    st.stop()

m = info['metricas']
limiar = info['limiar']

# ---------------------------------------------------------------------------
# Barra lateral: transparência sobre o modelo
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('### 🧠 Sobre o modelo')
    st.write(f"**{info['modelo']}** com padronização (StandardScaler).")
    c1, c2 = st.columns(2)
    c1.metric('Acurácia', f"{m['acuracia']:.0%}",
              f"{(m['acuracia'] - m['acuracia_dummy']) * 100:+.0f} p.p. vs. baseline")
    c2.metric('Recall', f"{m['recall']:.0%}",
              help='Dos pacientes com diabetes no teste, quantos o modelo identificou.')
    st.metric('Precisão', f"{m['precisao']:.0%}",
              help='Dos pacientes que o modelo apontou como risco, quantos de fato tinham diabetes.')
    st.caption(f"Treino: {info['n_treino']} pacientes · Teste: {info['n_teste']} pacientes")
    st.divider()
    st.markdown('### ℹ️ Como usar')
    st.write('1. Preencha os dados clínicos.\n2. Clique em **Avaliar risco**.\n'
             '3. Veja a probabilidade estimada pelo modelo.')
    st.divider()
    st.caption('Não substitui diagnóstico ou orientação médica.')

# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🩺 Avaliação de Risco de Diabetes</h1>
  <p>Informe os dados clínicos do paciente e receba a estimativa de risco gerada por Machine Learning.</p>
</div>
""", unsafe_allow_html=True)
st.write('')

col_form, col_res = st.columns([1.15, 1], gap='large')

# ---------------------------------------------------------------------------
# Formulário
# ---------------------------------------------------------------------------
with col_form:
    st.markdown('#### Dados clínicos')
    # Container com borda no lugar do st.form: dentro de um form a tela só atualiza
    # ao enviar, e o campo Gravidez precisa reagir na hora à escolha do Sexo.
    with st.container(border=True):
        valores = {}
        campos = st.columns(2)
        grupos_exibidos = set()
        n = 0  # posição do campo na grade de 2 colunas
        for col, s in info['colunas'].items():
            grupo = COLUNA_PARA_GRUPO.get(col)
            if grupo in grupos_exibidos:
                continue  # a pergunta deste grupo já foi exibida
            nome, unidade, ajuda = rotulo(col)
            label = f'{nome} ({unidade})' if unidade else nome
            faixa = f"Faixa no treino: {s['min']:g} a {s['max']:g}"
            with campos[n % 2]:
                if col == CAMPO_GRAVIDEZ and st.session_state.get(f'campo_{CAMPO_SEXO}') == 'Masculino':
                    # Masculino: Gravidez fica travada, mostra "Não se aplica" e o modelo recebe 0
                    st.text_input(label, value='Não se aplica', disabled=True,
                                  help='Preenchido automaticamente (valor 0) para o sexo masculino.')
                    valores[col] = 0
                elif grupo:
                    g = GRUPOS[grupo]
                    nome_g, ajuda_g = g['rotulo']
                    lista = list(g['opcoes'])
                    padrao = lista.index(g['padrao']) if 'padrao' in g else 0
                    if len(lista) <= 2:  # 2 opções: botões lado a lado
                        escolha = st.radio(nome_g, lista, index=padrao, horizontal=True, help=ajuda_g)
                    else:                # 3+ opções: lista suspensa, ocupa menos espaço
                        escolha = st.selectbox(nome_g, lista, index=padrao, help=ajuda_g)
                    # One-hot: a coluna da opção escolhida recebe 1, as demais recebem 0
                    for opcao, coluna in g['opcoes'].items():
                        valores[coluna] = int(opcao == escolha)
                    grupos_exibidos.add(grupo)
                elif col in OPCOES:
                    opcoes = OPCOES[col]
                    mediana = round(s['mediana'])
                    padrao = list(opcoes.values()).index(mediana) if mediana in opcoes.values() else 0
                    # key permite consultar a escolha em outros campos (ex.: Sexo -> Gravidez)
                    escolha = st.radio(nome, list(opcoes), index=padrao, horizontal=True, help=ajuda,
                                       key=f'campo_{col}')
                    valores[col] = opcoes[escolha]  # texto da tela -> número para o modelo
                elif s['inteiro']:
                    valores[col] = st.number_input(
                        label, min_value=0, max_value=int(s['max'] * 2),
                        value=int(round(s['mediana'])), step=1,
                        help=f'{ajuda}. {faixa}'.strip('. '),
                    )
                else:
                    valores[col] = st.number_input(
                        label, min_value=0.0, max_value=float(s['max'] * 2),
                        value=float(s['mediana']), step=0.1, format='%.2f',
                        help=f'{ajuda}. {faixa}'.strip('. '),
                    )
            n += 1
        enviar = st.button('Avaliar risco', type='primary', use_container_width=True)
    st.caption('Os campos numéricos começam preenchidos com a mediana dos dados de treino.')

# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------
with col_res:
    st.markdown('#### Resultado')

    if enviar:
        entrada = pd.DataFrame([valores])[list(info['colunas'])]  # mesma ordem do treino
        prob = float(modelo.predict_proba(entrada)[0, 1])
        alto = prob >= limiar
        classe = 'alto' if alto else 'baixo'
        cor = '#ef4444' if alto else '#10b981'
        selo = 'RISCO ELEVADO' if alto else 'RISCO BAIXO'
        titulo = 'Indícios de diabetes' if alto else 'Sem indícios de diabetes'
        texto = ('O perfil informado se parece com o de pacientes diagnosticados com diabetes. '
                 'Recomenda-se avaliação médica e exames complementares.'
                 if alto else
                 'O perfil informado se parece com o de pacientes sem diabetes. '
                 'Mantenha o acompanhamento de rotina.')

        st.markdown(f"""
        <div class="card resultado {classe}">
          <span class="selo {classe}">{selo}</span>
          <div class="titulo-resultado">{titulo}</div>
          <p class="sub">{texto}</p>
          <div class="barra">
            <div style="width:{prob * 100:.1f}%; background:{cor};"></div>
            <span class="limiar" style="left:{limiar * 100:.0f}%;"></span>
          </div>
          <div class="escala"><span>0%</span><span>limiar {limiar:.0%}</span><span>100%</span></div>
          <p style="margin-top:1rem; font-size:1.05rem;">
            Probabilidade estimada de diabetes: <b>{prob:.1%}</b>
          </p>
        </div>
        """, unsafe_allow_html=True)

        # Valores fora da faixa de treino reduzem a confiabilidade da previsão
        fora = [rotulo(c)[0] for c, v in valores.items()
                if c in info['colunas'] and c not in COLUNA_PARA_GRUPO and c not in OPCOES
                and not info['colunas'][c]['min'] <= v <= info['colunas'][c]['max']]
        if fora:
            st.warning('Valores fora da faixa vista no treino: ' + ', '.join(fora) +
                       '. A previsão pode ser menos confiável.')

        with st.expander('Ver dados enviados ao modelo'):
            st.dataframe(
                entrada.rename(columns=lambda c: rotulo(c)[0]).T.rename(columns={0: 'Valor'}),
                use_container_width=True,
            )
    else:
        st.markdown("""
        <div class="card vazio">
          <div class="icone">📋</div>
          <p>Preencha o formulário e clique em <b>Avaliar risco</b><br>para ver o resultado aqui.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="aviso">⚠️ATENÇÃO: O resultado é uma estimativa estatística '
                'e não substitui diagnóstico médico.</p>', unsafe_allow_html=True)