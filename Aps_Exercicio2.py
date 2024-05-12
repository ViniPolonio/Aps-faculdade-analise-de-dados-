from pulp import LpMaximize, LpProblem, LpVariable, LpStatus, value

# Problema de Maximização do Retorno
problema = LpProblem(name="Maximizar_Retorno", sense=LpMaximize)

# Variáveis de decisão
x = LpVariable("Acoes", lowBound=0, cat="Integer")
y = LpVariable("Titulos", lowBound=0, cat="Integer")
z = LpVariable("Imoveis", lowBound=0, cat="Integer")

# Função Objetivo
problema += 0.10 * x + 0.08 * y + 0.06 * z, "Maximizar_Retorno"

# Restrições
problema += x + y + z == 1, "Capital_Total"
problema += x <= 0.5, "Limite_Acoes"
problema += y <= 0.5, "Limite_Titulos"
problema += z <= 0.3, "Limite_Imoveis"
problema += 0.10 * x + 0.08 * y + 0.06 * z >= 0.08, "Meta_Retorno_Minimo"
problema += x >= 0.1, "Diversificacao_Acoes"
problema += y >= 0.1, "Diversificacao_Titulos"
problema += z >= 0.1, "Diversificacao_Imoveis"

# Resolve o problema
problema.solve()

# Resultados
print("Status:", LpStatus[problema.status])
print(f"Proporção de capital investido em ações: {x.varValue * 100:.2f}%")
print(f"Proporção de capital investido em títulos: {y.varValue * 100:.2f}%")
print(f"Proporção de capital investido em imóveis: {z.varValue * 100:.2f}%")
print(f"Retorno esperado total da carteira de investimentos: {value(problema.objective):.2f}%")