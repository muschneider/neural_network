package com.github.com.muschneider.rna.base

object Random {
  
  // Generate Double random number
  def random(min:Double, max:Double) = min + scala.util.Random.nextDouble * (max - min)
  
  // Generate Int random number
  def random(min:Int, max:Int) = (min + scala.util.Random.nextDouble * (max - min)).asInstanceOf[Int]

}