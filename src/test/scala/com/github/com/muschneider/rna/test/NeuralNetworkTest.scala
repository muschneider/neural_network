package com.github.com.muschneider.rna.test

import org.scalatest._
import org.scalatest.matchers._

import com.github.com.muschneider.rna.backpropagation.NeuralNetwork;

class NeuralNetworkTest extends FlatSpec with ShouldMatchers {

    val xorPattern: List[List[Double]] = List(
        List(0, 0, 0),
        List(0, 1, 1),
        List(1, 0, 1),
        List(1, 1, 0)
    )
    
    val nnXor = new NeuralNetwork( structure = List(2, 4, 1) )
    nnXor.train(15000, xorPattern, 0.005)

    "The XOR ( 0 - 0) test with Neural Network" should " be around 0" in {
        val (nnOut, _) = nnXor.feedForward( List(0, 0) )
        nnOut(0) should (be >= (0.01d) and be <= (0.09d))
    }
    
    "The XOR ( 0 - 1) test with Neural Network" should " be around 1" in {
        val (nnOut, _) = nnXor.feedForward( List(0, 1) )
        nnOut(0) should (be >= (0.9d) and be <= (1d))
    }
    
    "The XOR ( 1 - 0) test with Neural Network" should " be around  1" in {
        val (nnOut, _) = nnXor.feedForward( List(1, 0) )
        nnOut(0) should (be >= (0.9d) and be <= (1d))
    }
    
    "The XOR ( 1 - 1) test with Neural Network" should " be around 0" in {
        val (nnOut, _) = nnXor.feedForward( List(1, 1) )
        nnOut(0) should (be >= (0.01d) and be <= (0.09d))
    }
    

}
