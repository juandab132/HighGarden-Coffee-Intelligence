preguntas = [
    "What coffee type has the highest demand?",
    "What is the expected consumption for 2023?",
    "How reliable is the model for business decisions?"
]

for p in preguntas:
    print(f"\n{'='*60}")
    print(f"PREGUNTA: {p}")
    print('='*60)
    print(consultar(p))