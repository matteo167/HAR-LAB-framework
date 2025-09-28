import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Carregar dados
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Criar explainer LIME
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=['Maligno', 'Benigno'],
    mode='classification',
    random_state=42
)

# Explicar uma instância específica
instance_idx = 0
instance = X_test[instance_idx]

exp = explainer.explain_instance(
    instance, 
    model.predict_proba, 
    num_features=10
)

# Visualizar explicação no terminal
print("=" * 70)
print("EXPLICAÇÃO LIME - CÂNCER DE MAMA")
print("=" * 70)

print(f"\n📊 INSTÂNCIA ANALISADA: {instance_idx}")
print(f"🔮 Predição do modelo: {model.predict([instance])[0]} ({'Benigno' if model.predict([instance])[0] == 1 else 'Maligno'})")
print(f"📈 Probabilidades: [Maligno: {model.predict_proba([instance])[0][0]:.3f}, Benigno: {model.predict_proba([instance])[0][1]:.3f}]")

print(f"\n📋 VALORES DA INSTÂNCIA:")
for i, (feature, value) in enumerate(zip(feature_names, instance)):
    if i < 10:  # Mostrar apenas as primeiras 10 features para não poluir
        print(f"  {feature}: {value:.4f}")

print(f"\n🔍 EXPLICAÇÃO DETALHADA:")
print("-" * 50)

# Mostrar features que contribuem para cada classe
print(f"\n✅ FATORES que APOIAM a classificação como BENIGNO:")
features_benigno = []
for feature, weight in exp.as_list():
    if weight > 0:  # Contribui para classe Benigno
        features_benigno.append((feature, weight))

# Ordenar por peso (maior primeiro)
features_benigno.sort(key=lambda x: x[1], reverse=True)
for feature, weight in features_benigno:
    print(f"  ➕ {feature}: {weight:+.4f}")

print(f"\n❌ FATORES que APOIAM a classificação como MALIGNO:")
features_maligno = []
for feature, weight in exp.as_list():
    if weight < 0:  # Contribui para classe Maligno
        features_maligno.append((feature, weight))

# Ordenar por peso (menor primeiro)
features_maligno.sort(key=lambda x: x[1])
for feature, weight in features_maligno:
    print(f"  ➖ {feature}: {weight:+.4f}")

print(f"\n📊 TODAS AS FEATURES (ordenadas por importância absoluta):")
print("-" * 50)
for feature, weight in exp.as_list():
    simbolo = "🟢" if weight > 0 else "🔴" if weight < 0 else "⚪"
    print(f"  {simbolo} {feature}: {weight:+.4f}")

# Informações adicionais do modelo local
print(f"\n📐 INFORMAÇÕES DO MODELO LOCAL:")
print(f"   Score do modelo explicativo: {exp.score:.4f}")
print(f"   Intercept: {exp.intercept[1]:.4f}")

print(f"\n💡 INTERPRETAÇÃO:")
print("   Valores POSITIVOS → Contribuem para a classe BENIGNO")
print("   Valores NEGATIVOS → Contribuem para a classe MALIGNO")
print("   Quanto maior o valor absoluto, mais importante a feature")

print("\n" + "=" * 70)