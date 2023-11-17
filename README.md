# Proyecto de Reconocimiento de Emociones mediante Machine Learning

## Descripción del Proyecto 🚀

Las emociones desempeñan un papel crucial en la interacción humana, y el reconocimiento de emociones en imágenes se ha vuelto esencial para diversas aplicaciones. Este proyecto aborda el reconocimiento de emociones en distintas imágenes mediante un modelo de Deep Learning. El objetivo principal es diseñar y entrenar una red neuronal profunda capaz de diferenciar entre las emociones de **enojo**, **felicidad** y **tristeza**, modelando así la expresión emocional humana a través de imágenes.

## Conjunto de Datos 📊

El conjunto de datos utilizado en este proyecto es esencial para la correcta formación y evaluación del modelo de reconocimiento de emociones. Está organizado jerárquicamente en una carpeta principal denominada **"data"**, que contiene tres subcarpetas representando cada emoción específica: **"Happy"**, **"Sad"** y **"Angry"**. Cada imagen está etiquetada correctamente gracias a su path con la emoción correspondiente.

- **Happy:** Imágenes que representan expresiones asociadas con la felicidad, capturando momentos felices con gestos evidentes de alegría.
- **Sad:** Imágenes que reflejan expresiones vinculadas a la tristeza, con manifestaciones más pronunciadas de pensamiento y colores grises y oscuros.
- **Angry:** Imágenes que representan emociones de enojo, con colores fuertes y gestos intensos.

El conjunto de datos se distribuye de la siguiente manera:

- **Train:** 90% de los datos.
- **Validation:** 10% de los datos.
- **Test:** 20% del conjunto de entrenamiento.

Las imágenes tienen un formato de 224x224 píxeles con 3 canales (RGB) para garantizar una entrada coherente al modelo.

## Arquitectura del Modelo 🧠

La arquitectura inicial del modelo se basó en la red neuronal preentrenada **VGG16** mediante **Transfer Learning** con ImageNet. Se creó un modelo secuencial para agregar capas adicionales adaptadas al problema específico. La arquitectura consta de:

1. Capas Convolutivas de **VGG16**.
2. Capa **Flatten** para convertir la salida tridimensional en un vector unidimensional.
3. Capa Densa con **128 neuronas** y activación **ReLU**.
4. Capa de **Dropout del 40%** para prevenir el sobreajuste.
5. Capa Densa de salida con **3 neuronas** y activación **softmax** para las predicciones de las 3 clases.

## Predicciones y Despliegue 🚀

El modelo entrenado se guarda para realizar predicciones. El archivo **"userTest.ipynb"** proporciona una manera interactiva de hacer predicciones con el modelo en nuevas imágenes. Además, el archivo **"app.py"** crea un servidor en **Flask** que despliega una interfaz gráfica para facilitar la carga de imágenes y realizar predicciones.

¡Gracias por explorar nuestro proyecto de reconocimiento de emociones mediante Machine Learning! 🌈

