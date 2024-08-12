import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import streamlit as st
import plotly.graph_objs as go


st.set_page_config(page_title='Sistema de Inferência Nebulosa para Central de Peças', layout='wide')

# Função para tratamento de erros
def safe_compute(simulation):
    try:
        simulation.compute()
        return True
    except Exception as e:
        st.error(f"Erro ao calcular a saída: {e}")
        return False

# Definindo as variáveis linguísticas e suas funções de pertinência
# Variável de entrada: Tempo de espera (m)
m = ctrl.Antecedent(np.arange(0, 1.2, 0.001), 'tempo_espera')
m['muito_pequeno'] = fuzz.trapmf(m.universe, [0, 0, 0.1, 0.3])
m['pequeno'] = fuzz.trimf(m.universe, [0.1, 0.3, 0.5])
m['medio'] = fuzz.trapmf(m.universe, [0.4, 0.6, 0.7, 0.7])

# Variável de entrada: Fator de utilização de reparo da central (p)
p = ctrl.Antecedent(np.arange(0, 1.2, 0.001), 'fator_utilizacao')
p['baixo'] = fuzz.trapmf(p.universe, [0, 0, 0.2, 0.4])
p['medio'] = fuzz.trimf(p.universe, [0.3, 0.5, 0.7])
p['alto'] = fuzz.trapmf(p.universe, [0.6, 0.8, 1, 1])

# Variável de entrada: Número de funcionários (s)
s = ctrl.Antecedent(np.arange(0, 1.2, 0.001), 'numero_funcionarios')
s['pequeno'] = fuzz.trapmf(s.universe, [0, 0, 0.4, 0.6])
s['medio'] = fuzz.trimf(s.universe, [0.4, 0.6, 0.8])
s['grande'] = fuzz.trapmf(s.universe, [0.6, 0.8, 1, 1])

# Variável de saída: Número de peças extras (n)
n = ctrl.Consequent(np.arange(0, 1.2, 0.001), 'numero_pecas')
n['muito_pequeno'] = fuzz.trapmf(n.universe, [0, 0, 0.1, 0.3])
n['pequeno'] = fuzz.trimf(n.universe, [0, 0.2, 0.4])
n['pouco_pequeno'] = fuzz.trimf(n.universe, [0.25, 0.35, 0.45])
n['medio'] = fuzz.trimf(n.universe, [0.3, 0.5, 0.7])  # Adicionando a linha mediana para 'medio'
n['pouco_grande'] = fuzz.trimf(n.universe, [0.55, 0.65, 0.75])
n['grande'] = fuzz.trimf(n.universe, [0.6, 0.8, 1])
n['muito_grande'] = fuzz.trapmf(n.universe, [0.7, 0.9, 1, 1])

# Definindo as regras de inferência
rule1 = ctrl.Rule(m['muito_pequeno'] & s['pequeno'], n['muito_grande'])
rule2 = ctrl.Rule(m['pequeno'] & s['grande'], n['pequeno'])
rule3 = ctrl.Rule(p['baixo'], n['pequeno'])
rule4 = ctrl.Rule(p['alto'], n['grande'])

# Criando o sistema de controle
system = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
sim = ctrl.ControlSystemSimulation(system)

# Interface com Streamlit
st.title('Sistema de Inferência Nebulosa para Central de Peças')

# Sliders para entrada de dados
tempo_espera = st.slider('Tempo de Espera (0.0 a 1.0)', 0.0, 1.0, 0.3, step=0.01)
fator_utilizacao = st.slider('Fator de Utilização (0.0 a 1.0)', 0.0, 1.0, 0.3, step=0.01)
numero_funcionarios = st.slider('Número de Funcionários (0.0 a 1.0)', 0.0, 1.0, 0.3, step=0.01)

# Passando as entradas para o sistema
sim.input['tempo_espera'] = tempo_espera
sim.input['fator_utilizacao'] = fator_utilizacao
sim.input['numero_funcionarios'] = numero_funcionarios

# Realizando a computação
if safe_compute(sim):
    st.write(f'Número de peças extras recomendadas: {sim.output["numero_pecas"]:.2f}')

# Função para plotar as funções de pertinência com Plotly e marcar os pontos
def plot_fuzzy_var(var, var_name, input_value=None, output_value=None, medians=[]):
    traces = []
    for label in var.terms:
        trace = go.Scatter(
            x=var.universe,
            y=var[label].mf,
            mode='lines',
            name=label
        )
        traces.append(trace)

    layout = go.Layout(
        title=f'Função de Pertinência - {var_name}',
        xaxis=dict(
            title=var_name,
            tickvals=np.arange(0, 1.05, 0.05),  # Define os ticks em 0.05
            ticktext=[f'{i:.2f}' for i in np.arange(0, 1.05, 0.05)]  # Mostra os ticks como texto formatado
        ),
        yaxis=dict(title='Pertinência'),
    )
    
    fig = go.Figure(data=traces, layout=layout)

    # Adicionando linhas medianas
    for median in medians:
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=median, y0=0,
                x1=median, y1=1,
                line=dict(color="LightGray", dash="dash"),
            )
        )

    # Adicionando labels
    for label, x in zip(var.terms, medians):
        fig.add_annotation(
            x=x,
            y=1.05,
            text=label,
            showarrow=False,
            xanchor="center",
            yanchor="bottom"
        )

    if input_value is not None:
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=input_value, y0=0,
                x1=input_value, y1=1,
                line=dict(color="Red", dash="dashdot"),
            )
        )

    if output_value is not None:
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=output_value, y0=0,
                x1=output_value, y1=1,
                line=dict(color="Blue", dash="dashdot"),
            )
        )

    return fig

# Exibindo gráficos das funções de pertinência e marcando os pontos
st.subheader('Funções de Pertinência')
st.plotly_chart(plot_fuzzy_var(m, 'Tempo de Espera', input_value=tempo_espera, medians=[0.1, 0.3, 0.6]))
st.plotly_chart(plot_fuzzy_var(p, 'Fator de Utilização', input_value=fator_utilizacao, medians=[0.2, 0.5, 0.8]))
st.plotly_chart(plot_fuzzy_var(s, 'Número de Funcionários', input_value=numero_funcionarios, medians=[0.4, 0.6, 0.8]))
st.plotly_chart(plot_fuzzy_var(n, 'Número de Peças Extras', output_value=sim.output["numero_pecas"], medians=[0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9]))
