package com.github.com.muschneider.rna.base

object ActivateFunction {

    // Activation Function
    def sigmoid(value: Double) = (1 / (1 + math.exp(-value)))

}