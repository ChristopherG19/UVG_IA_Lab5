# UVG_IA_Lab5

## Análisis de resultados
- Task 1.1
  - En cuanto a los resultados obtenidos para el modelo del KNN la mejor implementación fue la que utiliza librerías. Esto porque dada la matriz de confusión se observa que tiene más precisión al momento de calcular True positives y True negatives. Sin embargo cabe resaltar que el modelo realizado sin librerías cuenta con una precisión y recall idénticos a los del modelo con librerías (0.87 y 0.81 respectivamente) indicando que es bastante bueno, sin embargo la precisión para calcular True positives y True negatives disminuye un poco.
 
  <p align="center">
    <img src="https://user-images.githubusercontent.com/60325784/224112555-77fd0dc7-c3aa-43e9-8422-f63fb257105a.png"/>
  </p>

  <p align="center">Primera matriz y reporte: Modelo sin librerías</br>Segunda matriz y reporte: Modelo con librerías</p>
  
- Task 1.2 
  - En cuanto a los resultados obtenidos para el modelo del SVM la mejor implementación fue la que utiliza librerías. Especificamente el SVM con función radial fue el utilizado. Esto porque la implementación realizada sin librerías no presenta un puntaje final, para las predicciones que hace, muy alto lo que nos indica que no es tan exacto con las respuestas que presenta. La precision que esta presenta es del 0.55 y la que presenta el modelo con librerías es de 0.61, talvez esta diferencia no es tan grande pero tomando otros valores como el recall para los datos es posible observar que se mantiene más balanceado respecto a los datos presentes y no hay un sesgo hacía uno de los valores como en el modelo sin librerías.
  
  <p align="center">
    <img src="https://user-images.githubusercontent.com/60325784/224113284-85259aeb-d22d-4bbf-b4c6-1c8822ddfb1a.png"/>
  </p>
  
  <p align="center">Primera matriz y reporte: Modelo sin librerías</br>Segunda matriz y reporte: Modelo con librerías</p>
  
- Comparación
  - ¿Cómo difirieron los grupos creados por ambos modelos?: 
  - ¿Cuál de los modelos fue más rápido?: El modelo más rápido fue el SVM pero esto debido a que el modelo del KNN se ejecuta varias veces hasta encontrar el mejor k para retornar una respuesta más precisa.
  - ¿Qué modelo usarían?: Entre los modelos sin librerías, el modelo KNN sería el indicado porque es más preciso y la respuesta hacia predicciones sobre identificación de sitios es más acertada que la que retornaría el modelo SVM. De igual forma entre los modelos con librerías, se recomendaría utilizar el KNN para realizar predicciones porque sus respuestas serán más acertadas y permitirán la detección de sitios dudosos.
