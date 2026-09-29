# Explicaciones locales (SHAP)

Valor base (log-odds medio del modelo): -2.456 → probabilidad 0.079. Umbral operativo: 0.050.

## Verdadero positivo (fila 2477 del test)

- Cliente: 53 años, ocupación `unknown`, estado civil `married`, educación `primary`, saldo 732, hipoteca `no`, préstamo `no`, contacto `cellular`, mes `oct`, contactos en campaña 2, resultado previo `unknown`.
- Probabilidad predicha: **0.284** (umbral 0.050) → llamar. Resultado real: **convirtió**.
- Factores que más **aumentan** la probabilidad: Mes = oct (+1.36), Nº de préstamos = 0 (+0.13), Tipo de contacto = cellular (+0.11).
- Factores que más la **reducen**: Educación = primary (-0.13), Estado civil = married (-0.11), Ocupación = unknown (-0.07).

## Falso positivo (fila 657 del test)

- Cliente: 64 años, ocupación `retired`, estado civil `married`, educación `primary`, saldo 43, hipoteca `no`, préstamo `no`, contacto `cellular`, mes `mar`, contactos en campaña 1, resultado previo `success`.
- Probabilidad predicha: **0.814** (umbral 0.050) → llamar. Resultado real: **no convirtió**.
- Factores que más **aumentan** la probabilidad: Éxito previo = 1 (+1.23), Mes = mar (+0.82), Resultado previo = success (+0.50).
- Factores que más la **reducen**: Educación = primary (-0.11), Estado civil = married (-0.04), Grupo de edad = 55-64 (-0.01).

## Falso negativo (fila 1135 del test)

- Cliente: 57 años, ocupación `retired`, estado civil `married`, educación `tertiary`, saldo 0, hipoteca `yes`, préstamo `yes`, contacto `unknown`, mes `may`, contactos en campaña 1, resultado previo `unknown`.
- Probabilidad predicha: **0.021** (umbral 0.050) → no llamar. Resultado real: **convirtió**.
- Factores que más **aumentan** la probabilidad: Contactos en campaña = 1.0 (+0.08), Educación = tertiary (+0.04), Estación = spring (+0.02).
- Factores que más la **reducen**: Tipo de contacto = unknown (-0.34), Canal conocido = 0 (-0.33), Saldo (log) = 0.0 (-0.20).
