[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

> Implementación de un juego Pong en C++ con paddle controlado por una red neuronal.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)
---

### Datos generales

* **Tema**: Juego Pong con paddle IA con Redes Neuronales en C++
* **Grupo**: `pongsasos`
* **Integrantes**:

  * Saldarriaga Núñez, Annemarie Alejandra – 202410265 (Alumno A)
  * Bonilla Sarmiento, Martin Jesús – 202410303 (Alumno B)
  * Lazón Meza, María Fernanda – 202410320 (Alumno C)
  * Anaya Manzo, Matias Javier – 202410238 (Alumno D)

---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior
2. **Dependencias**:

   * CMake 3.18+
   * Eigen 3.4
   * Raylib
3. **IDEs**: CLion, VSCode o similares
4. **Instalación**:

   ```bash 
   git clone https://github.com/CS1103/projecto-final-pongsasos.git
   cd pongsasos
   
   # Para macOS solo se ejecuta el siguiente comando
   brew install raylib # En macOS
   
   # Instalar raylib para Window y Linux
   git clone https://github.com/microsoft/vcpkg.git
   cd vcpkg
   
   # Windows (instala raylib para MSVC de 64 bits)
   ./bootstrap-vcpkg.bat
   .\vcpkg\vcpkg install raylib:x64-windows 
   
   # Linux (instala raylib para tripleta nativa)
   ./bootstrap-vcpkg.sh
   ./vcpkg/vcpkg install raylib
   
   # Solo una vez por sistema
   .\vcpkg\vcpkg integrate install
   ```
Luego de instalar la librería Raylib, se es necesario una última configuración:
   * Para CLion:
     * Ve a: `File > Settings > Build, Execution, Deployment > CMake`
     * Añadir la siguiente línea en **CMake options**:
      ```ini
        -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake
     ```
   * Para VSCode (usando la extensión CMake Tools):
     * En `.vscode/settings.json`, agregar:
     ```json
     "cmake.configureArgs": [
     "-DCMAKE_TOOLCHAIN_FILE=${workspaceFolder}/vcpkg/scripts/buildsystems/vcpkg.cmake
     ]
     ```
- Extra: Para ejecución manual
````bash
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build .
````
---

### 1. Investigación teórica

* **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.
* **Contenido de ejemplo**:

  1. Historia y evolución de las NNs.
  2. Principales arquitecturas: MLP, CNN, RNN.
  3. Algoritmos de entrenamiento: backpropagation, optimizadores.

---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

* **Patrones de diseño**: ejemplo: No se he utilizado ningún patrón de diseño.
* **Estructura de carpetas**:

  ```
  pongsasos/
  ├── nn/
  │   ├── network.h
  │   ├── tensor.h
  ├── main.cpp
  ├── test_neural_network.cpp
  ├── README.md
  └── CMakeLists.txt
  ```

#### 2.2 Manual de uso y casos de prueba

* **Cómo ejecutar** (en Git Bash): `cd ./ruta_al_proyecto/cmake-build-debug && ./nombre_del_proyecto`
* **Casos de prueba**:

  * Test unitario para la función de pérdida de la red.

---

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `docs/demo.mp4`.
> Pasos:
>
> 1. Preparar datos de entrenamiento.
> 2. Presionar tecla 'T' para empezar el entrenamiento'.
> 3. Presionar tecla 'F' para acelerar el entrenamiento.
> 4. La red neuronal se entrena con los datos.
> 5. El AIPaddle está lista para etrenar.

---

### 4. Análisis del rendimiento

* **Métricas de ejemplo**:

  * Iteraciones: 100  épocas.
  * Tiempo total de entrenamiento: 12 min.
* **Ventajas/Desventajas**:

  * Código ligero y dependencias mínimas.
  * – Sin paralelización, rendimiento limitado.
* **Mejoras futuras**:

  * Uso de CUDA para interfaz gráfica (Justificación).
  * Paralelizar entrenamiento por lotes (Esto ayudaría a disminuir el tiempo de entrenamiento).

---

### 5. Trabajo en equipo

| Tarea                                 | Miembro(s)            | 
|---------------------------------------|-----------------------|
| Investigación teórica                 | Alumno D              |
| Implementación de la interfaz gráfica | Alumno A y C          |
| Implementación de AIPaddle            | Alumno B              |
| Video de presentación del proyecto    | Todos los integrantes |
| Edición del video                     | Alumno B              |

---

### 6. Conclusiones

* **Logros**: Implementar NN desde cero, validar en dataset de ejemplo.
* **Evaluación**: Calidad y rendimiento adecuados para propósito académico.
* **Aprendizajes**: Profundización en backpropagation, optimización e implementación de interfaz gráfica con librería Raylib.
* **Recomendaciones**: Escalar a datasets más grandes y optimizar memoria.

---

### 7. Bibliografía

> [1] T. Esmael, “Activation Functions in Neural Networks,” International Journal of Engineering and Advanced Scientific Technology (IJEAST), vol. 4, no. 12, pp. 310–316, 2022. [En línea]. Disponible en: https://d1wqtxts1xzle7.cloudfront.net/89662883/310-316_Tesma412_IJEAST-libre.pdf
> 
> [2] M. Nielsen, Neural Networks and Deep Learning, cap. 1. [En línea]. Disponible en: https://neuralnetworksanddeeplearning.com/chap1.html
> 
> [3] S. Haykin, Neural Networks and Learning Machines, 3rd ed. Upper Saddle River, NJ, USA: Pearson Education, 2009.
> 
> [4] C. M. Bishop, Pattern Recognition and Machine Learning. New York, NY, USA: Springer, 2006.

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
