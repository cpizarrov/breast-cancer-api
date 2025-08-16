# API para la detección de cáncer: Manual del usuario

## Descripción del dataset utilizado

Se desarrolló un modelo de regresión logística para la detección de cáncer de mamas, a partir del dataset [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). En éste se recopilan imágenes de distintos núcleos celulares, y los casos son clasificados como Benignos (B) o Malignos (M). 

El dataset cuenta con:
- 569 casos: 357 Benignos y 212 Malignos 
- Variable dependiente categórica: Diagnosis (B o M) 
- 30 variables independientes numéricas (continuas)

Por cada núcleo celular se extraen la media (mean), el error estándar (error) y el peor valor (worst) de estos 10 atributos: radio, textura, perímetro, área, suavidad, compacidad, concavidad, puntos cóncavos, simetría y dimensión fractal.

*Nota: Para efectos del modelo, Benigno (B) = 0 y Maligno (M) = 1.*

## Realizando consultas a la API a través de Python

Las siguientes instrucciones han sido ajustadas para poder ser ejecutadas en la siguiente [planilla de google collab](https://colab.research.google.com/drive/1gNcbb-4dmE9XKaFZqeL5Ff4rmWcoZyaW?usp=sharing), la cual puedes utilizar sólo ajustando los datos del payload. 

### URL de la API

```bash
https://breast-cancer-api-37mk.onrender.com
```

### Prepara las dependencias

La única dependedencia que se necesita instalar para realizar consultas a la API es **requests**. 

```bash
pip install requests
```

También necesitaremos importar las librerías *time* y *json*.

```bash
import requests
import time
import json
```

### Prepara la URL y define una función de warm-up

Dado que estamos utilizando un plan gratuito de render, necesitamos "despertar" la instancia inactiva. Para ello, definimos una función para que nuestras consultas no arrojen error mientras la instancia se prepara. 

Puedes preparar la URL y la instancia utilizando el siguiente código: 

```bash
BASE = "https://breast-cancer-api-37mk.onrender.com"
HEALTH = f"{BASE}/health"
PRED   = f"{BASE}/predict"
FEATS  = f"{BASE}/features"

def warm_up(max_wait_s=420, initial_read_to=40, max_read_to=180, sleep_s=2):
    start = time.time(); attempt = 0; read_to = initial_read_to; connect_to = 6
    while True:
        attempt += 1
        try:
            r = requests.get(HEALTH, timeout=(connect_to, read_to))
            ok = (r.status_code == 200 and r.json().get("status") == "ok")
            print(f"[warm-up {attempt}] HTTP {r.status_code} | elapsed={r.elapsed.total_seconds():.1f}s | ok={ok}")
            if ok:
                print("Servicio listo ✅"); return
        except requests.exceptions.RequestException as e:
            print(f"[warm-up {attempt}] {type(e).__name__}: {e}")
        if time.time() - start > max_wait_s:
            raise TimeoutError("El servicio no quedó listo durante el warm-up.")
        read_to = min(read_to + 20, max_read_to); time.sleep(sleep_s)

# Warm-up
warm_up()
```

### Estructura de los datos de entrada

Debes entregar un archivo JSON que contenga la clave **features**. Al interior de ésta, deben estar contenidas 30 claves que coincidan exactamente con los nombres de las variables de **/features**, y cuyos valores asociados sean de tipo flotante.

Aquí tienes un payload de ejemplo, con los nombres correspondientes de las variables: 

```bash
payload_ejemplo = {
  "features": {
    "mean radius": 17.99,
    "mean texture": 10.38,
    "mean perimeter": 122.8,
    "mean area": 1001.0,
    "mean smoothness": 0.1184,
    "mean compactness": 0.2776,
    "mean concavity": 0.3001,
    "mean concave points": 0.1471,
    "mean symmetry": 0.2419,
    "mean fractal dimension": 0.07871,
    "radius error": 1.095,
    "texture error": 0.9053,
    "perimeter error": 8.589,
    "area error": 153.4,
    "smoothness error": 0.006399,
    "compactness error": 0.04904,
    "concavity error": 0.05373,
    "concave points error": 0.01587,
    "symmetry error": 0.03003,
    "fractal dimension error": 0.006193,
    "worst radius": 25.38,
    "worst texture": 17.33,
    "worst perimeter": 184.6,
    "worst area": 2019.0,
    "worst smoothness": 0.1622,
    "worst compactness": 0.6656,
    "worst concavity": 0.7119,
    "worst concave points": 0.2654,
    "worst symmetry": 0.4601,
    "worst fractal dimension": 0.1189
  }
}
```

### Solicita tus predicciones

Utiliza el siguiente código para enviar tu consulta

```bash
resp = requests.post(PRED, json=payload_ejemplo, timeout=(6, 180))
print("HTTP", resp.status_code)
print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
```

Obtendrás una respuesta de este estilo: 

```bash
HTTP 200
{
  "predicted_class": 1,
  "prob_maligno": 0.9999572315866951,
  "label": "M",
  "threshold_used": 0.5721941718081877,
  "missing_features": [],
  "extra_features": []
}
```
Donde:
- **predicted_class:** 0 = Benigno (B) y 1 = Maligno (M).
- **prob_maligno:** Probabilidad de que sea "Maligno". 
- **label:** Etiqueta de la clase predicha ("M" o "B").
- **threshold_used:** Umbral de decisión utilizado.

### Errores típicos: 

Si ingresas un string como valor, se te solicitará ingresar un valor numérico válido. Al mismo tiempo, se te indicará cuál es el valor incorrecto que incorporaste en el payload. 

```bash
HTTP 422
{
  "detail": [
    {
      "type": "float_parsing",
      "loc": [
        "body",
        "features",
        "mean radius"
      ],
      "msg": "Input should be a valid number, unable to parse string as a number",
      "input": "asd"
    }
  ]
}
```

Si ingresas mal el nombre de una variable, se te indicará que el nombre de las variables no coinciden con las esperadas. Al mismo tiempo, si falta o sobra alguna variable, se te indicará dentro de los campos **missing_features** o **extra_features**.

```bash
HTTP 422
{
  "detail": {
    "msg": "Las variables no coinciden con las esperadas (usa /features).",
    "missing_features": [
      "worst fractal dimension"
    ],
    "extra_features": [], (...)
```
