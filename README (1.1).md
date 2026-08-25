Enlace a WOKWI
https://wokwi.com/projects/473343017327443969
# Detección de Carro y Moto de Juguete con YOLO + ESP32

Sistema de visión artificial que detecta en tiempo real un **carro de juguete** y una **moto de juguete** usando el modelo **YOLO**, y enciende un **LED rojo** o un **LED verde** conectados a una **ESP32** según el objeto detectado.

| Objeto detectado | Salida física |
|---|---|
| 🚗 Carro de juguete | LED **rojo** encendido |
| 🏍️ Moto de juguete | LED **verde** encendido |
| Nada / objeto desconocido | Ambos LEDs apagados |

**Herramientas utilizadas:**

| Herramienta | Lenguaje | Dónde se ejecuta | Función |
|---|---|---|---|
| **Thonny** | MicroPython | ESP32 | Control de los LEDs |
| **Visual Studio Code** | Python | PC | Detección con YOLO |

---

## 1. ¿Qué es YOLO?

**YOLO** (*You Only Look Once* — "solo miras una vez") es una familia de modelos de **detección de objetos** basada en redes neuronales convolucionales (CNN). Fue propuesta por Joseph Redmon en 2015 y hoy va por versiones como YOLOv5, YOLOv8 y YOLOv11 (Ultralytics).

A diferencia de un clasificador normal, que solo responde *"¿qué hay en la imagen?"*, YOLO responde tres cosas al mismo tiempo:

1. **Qué** objeto hay (clase: carro, moto, persona…).
2. **Dónde** está (caja delimitadora o *bounding box*: x, y, ancho, alto).
3. **Qué tan seguro está** de esa predicción (*confidence score*, de 0 a 1).

### ¿Por qué se llama "You Only Look Once"?

Los detectores anteriores (R-CNN, Fast R-CNN) funcionaban en **dos etapas**: primero proponían cientos o miles de regiones candidatas y luego clasificaban cada región por separado. Eso significaba pasar la imagen por la red **muchas veces** → muy lento.

YOLO replantea el problema como una **regresión única**: la imagen completa entra **una sola vez** a la red y de esa única pasada salen directamente todas las cajas y todas las clases. Por eso alcanza velocidades de **30 a 150+ FPS**, lo que lo hace ideal para tiempo real (que es justamente lo que necesitamos para encender un LED al instante).

### ¿Cómo funciona internamente?

**Paso 1 — División en cuadrícula (grid)**
La imagen de entrada se redimensiona (típicamente a 640×640 px) y se divide conceptualmente en una cuadrícula de S×S celdas.

```
┌────┬────┬────┬────┐
│    │    │    │    │
├────┼────┼────┼────┤
│    │ 🚗 │    │    │   ← la celda donde cae el centro del objeto
├────┼────┼────┼────┤      es la responsable de detectarlo
│    │    │    │ 🏍️ │
├────┼────┼────┼────┤
│    │    │    │    │
└────┴────┴────┴────┘
```

**Paso 2 — Extracción de características (backbone)**
Una CNN (backbone, p. ej. CSPDarknet) recorre la imagen y extrae mapas de características: bordes, texturas, formas, ruedas, contornos, etc. Las capas iniciales detectan detalles simples y las profundas, formas complejas.

**Paso 3 — Fusión multiescala (neck)**
Una estructura tipo FPN/PAN combina información de varias resoluciones. Esto permite detectar tanto objetos **grandes** (moto cerca de la cámara) como **pequeños** (carro al fondo de la mesa).

**Paso 4 — Predicción (head)**
Cada celda predice varias cajas. Por cada caja se obtiene:

```
[ x, y, w, h, confianza_objeto, p(clase_1), p(clase_2), ... ]
```

- `x, y` → centro de la caja
- `w, h` → ancho y alto
- `confianza` → probabilidad de que ahí exista un objeto real
- `p(clase_i)` → probabilidad de cada clase

La puntuación final de una detección es: `score = confianza × p(clase)`

**Paso 5 — Filtrado por umbral de confianza**
Se descartan todas las cajas con score menor al umbral (p. ej. 0.5). Así se elimina el ruido.

**Paso 6 — Non-Maximum Suppression (NMS)**
Como varias celdas vecinas suelen detectar el mismo objeto, quedan cajas duplicadas. El NMS:
1. Toma la caja con mayor score.
2. Calcula el **IoU** (*Intersection over Union*) contra las demás.
3. Elimina las que se solapan por encima de un umbral (p. ej. IoU > 0.45).
4. Repite hasta dejar **una sola caja por objeto**.

```
IoU = área de intersección / área de unión
```

**Resultado final:** una lista de detecciones limpias con clase, coordenadas y confianza — que es exactamente lo que nuestro programa usa para decidir qué LED encender.

### Ventajas y limitaciones

| Ventajas | Limitaciones |
|---|---|
| Muy rápido (tiempo real) | Menor precisión en objetos muy pequeños |
| Ve el contexto global de la imagen | Dificultad con objetos muy juntos o superpuestos |
| Arquitectura simple, un solo modelo | Necesita dataset propio para objetos no estándar |
| Fácil de entrenar con datos propios | Sensible a iluminación y ángulos no vistos en entrenamiento |

---

## 2. Arquitectura del sistema

La ESP32 **no ejecuta YOLO**: no tiene memoria ni potencia de cómputo suficiente para una CNN de este tamaño. El diseño correcto reparte el trabajo así:

```
   ┌──────────────┐        ┌───────────────────────────┐        ┌──────────────┐
   │   CÁMARA     │  video │   PC — VISUAL STUDIO CODE │  WiFi  │ ESP32—THONNY │
   │ Webcam o     ├───────►│  YOLO: detecta y clasifica├───────►│  Recibe orden│
   │ ESP32-CAM    │        │  Decide: carro / moto     │  HTTP  │  Enciende LED│
   └──────────────┘        └───────────────────────────┘        └──────┬───────┘
                                                                       │
                                                            ┌──────────┴──────────┐
                                                            │  LED rojo  LED verde │
                                                            └──────────────────────┘
```

- **PC (Visual Studio Code)** → hace el trabajo pesado: la inferencia de YOLO.
- **ESP32 (Thonny / MicroPython)** → actúa como **actuador**: recibe una orden simple (`carro`, `moto`, `ninguno`) y controla los LEDs.
- **Comunicación** → HTTP sobre WiFi. Ambos dispositivos deben estar en la **misma red de 2.4 GHz**.

---

## 3. Materiales y conexiones

| Cantidad | Componente |
|---|---|
| 1 | Placa ESP32 (DevKit V1) con firmware **MicroPython** |
| 1 | LED rojo de 5 mm |
| 1 | LED verde de 5 mm |
| 2 | Resistencias de 220 Ω |
| 1 | Protoboard y cables jumper |
| 1 | Cable micro-USB |
| 1 | Cámara web |
| 1 | Carro de juguete y moto de juguete |

### Conexiones

| ESP32 | Componente |
|---|---|
| GPIO 25 | Ánodo (+) LED rojo → resistencia 220 Ω → GND |
| GPIO 26 | Ánodo (+) LED verde → resistencia 220 Ω → GND |
| GND | Cátodo (−) de ambos LEDs |

```
GPIO25 ──[220Ω]──►|── GND     (LED ROJO  - carro)
GPIO26 ──[220Ω]──►|── GND     (LED VERDE - moto)
```

> ⚠️ La ESP32 trabaja a **3.3 V**. Nunca alimentes los LEDs desde 5 V directo a un GPIO.

---

## 4. CÓDIGO DE THONNY — ESP32 (MicroPython)

> **Archivo:** `main.py` → se guarda **dentro de la ESP32** (en Thonny: *Archivo → Guardar como… → MicroPython device → `main.py`*). Así el programa arranca solo cada vez que la placa se energiza.

Este código conecta la ESP32 al WiFi, levanta un pequeño servidor y espera órdenes del tipo:
`http://192.168.1.50/led?objeto=carro`

```python
"""
THONNY - MicroPython - ESP32
Recibe la clase detectada por YOLO y enciende el LED correspondiente.
  carro   -> LED ROJO  (GPIO 25)
  moto    -> LED VERDE (GPIO 26)
  ninguno -> ambos apagados
"""

import network
import socket
import time
from machine import Pin

# ==================== CONFIGURACIÓN ====================
SSID     = "NOMBRE_DE_TU_RED"
PASSWORD = "CONTRASENA_DE_TU_RED"

led_rojo  = Pin(25, Pin.OUT)   # Carro
led_verde = Pin(26, Pin.OUT)   # Moto

TIMEOUT = 3000                 # ms sin mensajes -> apagar LEDs por seguridad
estado_actual = "ninguno"
ultimo_mensaje = time.ticks_ms()


# ==================== CONTROL DE LEDS ====================
def apagar_todo():
    led_rojo.value(0)
    led_verde.value(0)

def encender_rojo():
    led_rojo.value(1)
    led_verde.value(0)

def encender_verde():
    led_rojo.value(0)
    led_verde.value(1)

def parpadeo_inicio():
    for _ in range(3):
        led_rojo.value(1); led_verde.value(1)
        time.sleep(0.15)
        apagar_todo()
        time.sleep(0.15)


# ==================== CONEXIÓN WIFI ====================
def conectar_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print("Conectando a", SSID, "...")
        wlan.connect(SSID, PASSWORD)
        while not wlan.isconnected():
            time.sleep(0.5)
            print(".", end="")
    ip = wlan.ifconfig()[0]
    print("\n¡WiFi conectado!")
    print(">>> IP DE LA ESP32:", ip)   # <-- copiar esta IP al código de Visual Studio Code
    return ip


# ==================== SERVIDOR ====================
def extraer_objeto(peticion):
    """Obtiene el valor de ?objeto=... de la petición HTTP."""
    if "objeto=" in peticion:
        valor = peticion.split("objeto=")[1]
        valor = valor.split(" ")[0].split("&")[0]
        return valor.strip()
    return None


def procesar(objeto):
    global estado_actual, ultimo_mensaje
    ultimo_mensaje = time.ticks_ms()
    estado_actual = objeto

    if objeto == "carro":
        encender_rojo()
        print("[ESP32] CARRO detectado -> LED ROJO")
    elif objeto == "moto":
        encender_verde()
        print("[ESP32] MOTO detectada -> LED VERDE")
    else:
        apagar_todo()
        print("[ESP32] Sin deteccion -> LEDs apagados")


# ==================== PROGRAMA PRINCIPAL ====================
apagar_todo()
conectar_wifi()
parpadeo_inicio()

servidor = socket.socket()
servidor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
servidor.bind(('', 80))
servidor.listen(1)
servidor.settimeout(0.5)          # permite revisar el timeout de seguridad
print("Servidor iniciado. Esperando detecciones...")

while True:
    try:
        cliente, direccion = servidor.accept()
    except OSError:
        # No llegó ninguna petición: verificar timeout de seguridad
        if estado_actual != "ninguno":
            if time.ticks_diff(time.ticks_ms(), ultimo_mensaje) > TIMEOUT:
                apagar_todo()
                estado_actual = "ninguno"
                print("[ESP32] Timeout -> LEDs apagados")
        continue

    try:
        peticion = cliente.recv(1024).decode()
        objeto = extraer_objeto(peticion)

        if objeto is not None:
            procesar(objeto)
            respuesta = "OK:" + objeto
        else:
            respuesta = "Uso: /led?objeto=carro | moto | ninguno"

        cliente.send("HTTP/1.1 200 OK\r\n")
        cliente.send("Content-Type: text/plain\r\n")
        cliente.send("Connection: close\r\n\r\n")
        cliente.send(respuesta)
    except Exception as e:
        print("Error:", e)
    finally:
        cliente.close()
```

---

## 5. CÓDIGO DE VISUAL STUDIO CODE — PC (Python + YOLO)

> **Archivo:** `deteccion_yolo.py` → se guarda **en el PC** y se ejecuta desde la terminal de VS Code.
> Antes de ejecutarlo: `pip install ultralytics opencv-python requests`

```python
"""
VISUAL STUDIO CODE - Python - PC
Detección de carro y moto de juguete con YOLO.
Envía la clase detectada a la ESP32 por HTTP para encender el LED correspondiente.
"""

import cv2
import time
import requests
from ultralytics import YOLO

# ==================== CONFIGURACIÓN ====================
ESP32_IP   = "192.168.1.50"                  # IP que imprime Thonny en la consola
URL_ESP32  = f"http://{ESP32_IP}/led"

MODELO     = "yolov8n.pt"   # o "runs/detect/train/weights/best.pt" si entrenas el tuyo
FUENTE     = 0              # 0 = webcam del PC

UMBRAL_CONF = 0.55          # confianza mínima para aceptar una detección
INTERVALO   = 0.30          # segundos mínimos entre envíos a la ESP32

# Traduce los nombres del modelo a las dos categorías del proyecto.
# COCO (modelo preentrenado) usa 'car' y 'motorcycle'.
MAPA_CLASES = {
    "car":            "carro",
    "truck":          "carro",
    "motorcycle":     "moto",
    "carro_juguete":  "carro",
    "moto_juguete":   "moto",
}

COLORES = {"carro": (0, 0, 255), "moto": (0, 255, 0)}   # BGR


# ==================== COMUNICACIÓN ====================
def enviar_a_esp32(objeto: str) -> None:
    """Envía la orden a la ESP32. Si falla, no rompe el programa."""
    try:
        requests.get(URL_ESP32, params={"objeto": objeto}, timeout=0.6)
    except requests.exceptions.RequestException:
        print("[WARN] No se pudo comunicar con la ESP32")


# ==================== PROGRAMA PRINCIPAL ====================
def main() -> None:
    print("[INFO] Cargando modelo YOLO...")
    modelo = YOLO(MODELO)

    cap = cv2.VideoCapture(FUENTE)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara")
        return

    estado_anterior = None
    ultimo_envio = 0.0

    print("[INFO] Detección iniciada. Presiona 'q' para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] No se recibió imagen de la cámara")
            break

        # ---- INFERENCIA YOLO (una sola pasada por la red) ----
        resultados = modelo(frame, conf=UMBRAL_CONF, verbose=False)[0]

        detectado = "ninguno"
        mejor_conf = 0.0

        # ---- RECORRER LAS DETECCIONES ----
        for caja in resultados.boxes:
            id_clase   = int(caja.cls[0])
            nombre_raw = modelo.names[id_clase]
            confianza  = float(caja.conf[0])

            objeto = MAPA_CLASES.get(nombre_raw)
            if objeto is None:
                continue    # clase que no nos interesa (persona, silla, etc.)

            # Dibujar la caja en pantalla
            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            color = COLORES[objeto]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{objeto} {confianza:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Nos quedamos con la detección más confiable del frame
            if confianza > mejor_conf:
                mejor_conf = confianza
                detectado = objeto

        # ---- ENVIAR A LA ESP32 ----
        ahora = time.time()
        if detectado != estado_anterior or (ahora - ultimo_envio) > 1.0:
            if (ahora - ultimo_envio) > INTERVALO:
                enviar_a_esp32(detectado)
                print(f"[YOLO] {detectado.upper():8s}  conf={mejor_conf:.2f}")
                estado_anterior = detectado
                ultimo_envio = ahora

        # ---- INTERFAZ ----
        texto_led = {"carro": "LED ROJO ON", "moto": "LED VERDE ON"}.get(detectado, "LEDs OFF")
        cv2.putText(frame, texto_led, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    COLORES.get(detectado, (200, 200, 200)), 2)

        cv2.imshow("Deteccion YOLO - Carro / Moto", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    enviar_a_esp32("ninguno")   # apagar LEDs al cerrar
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Programa finalizado")


if __name__ == "__main__":
    main()
```
