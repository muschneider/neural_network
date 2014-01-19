package com.github.com.muschneider.rna.backpropagation

import com.github.com.muschneider.rna.base.Random.random
import com.github.com.muschneider.rna.base.ActivateFunction.sigmoid

class NeuralNetwork(val structure: List[Int], val learnRate:Double = 0.7) {

   // Weights for Hidden layer
   var hiddenLayer: List[List[Double]] = List.fill(structure(1))(List.fill(structure(0) + 1)(random(-0.5, 0.5)))

   // Weights for Output layer
   var outputLayer: List[List[Double]] = List.fill(structure(2))(List.fill(structure(1) + 1)(random(-0.5, 0.5)))

   var mse: List[Double] = Nil

   def train(epochs: Int, pattern: List[List[Double]], acc: Double) = {
      var errorSum: Double = 9.9
      for (epoch <- 1 to epochs; if errorSum == 0 || errorSum > acc) {
         errorSum = 0
         for (p <- pattern) {
          
            val inputPattern = p.take(structure(0))
            val outputPattern = p.takeRight(structure(2))

            //get network output
            val (outputNeuralNetwork, outputHiddenLayer) = feedForward(inputPattern)

            //compute error
            val errors = (outputPattern, outputNeuralNetwork).zipped.map((f: Double, b: Double) => (f - b))
            val deltak = (errors, outputNeuralNetwork).zipped.map((netErro: Double, netOut: Double) => netErro * netOut * (1 - netOut))
            val deltah = (outputHiddenLayer.init, outputLayer.transpose.map((deltak, _).zipped.map(_ * _).sum)).zipped.map((oh: Double, dkw: Double) => oh * (1 - oh) * dkw)

            //update weights
            hiddenLayer = (hiddenLayer, deltah).zipped.map((weight: List[Double], delta: Double) => (weight, inputPattern ::: List(1.0)).zipped.map((w: Double, i: Double) => w + (delta * i * learnRate)))
            outputLayer = (outputLayer, deltak).zipped.map((weight: List[Double], delta: Double) => (weight, outputHiddenLayer).zipped.map((w: Double, i: Double) => w + (delta * i * learnRate)))

            errorSum += scala.math.pow(errors.sum, 2)
         }
         val epochMSE = errorSum / (2 * pattern.length)
         mse ::= epochMSE
      }
   }

   def feedForward(inputPattern: List[Double]) = {
      val outputHiddenLayer = hiddenLayer.map(weights => sigmoid((weights, inputPattern ::: List(1.0)).zipped.map(_ * _).sum)) ::: List(1.0)
      val outputNeuralNetwork = outputLayer.map(weights => sigmoid((weights, outputHiddenLayer).zipped.map(_ * _).sum))
      (outputNeuralNetwork, outputHiddenLayer)
   }

}

