package scalann

import breeze.linalg.DenseVector
import org.scalatest._

class NetworkTest extends FlatSpec with Matchers {
  "A network" should "initialize with zero weights and biases" in {
    val n = new Network(Seq(1,1))
    n.weights.size should be (1)
    n.weights(0)(0,0) should be (0d)
    n.biases(0)(0) should be (0d)
  }

  it should "feed forward" in {
    val n = new Network(Seq(1,1))
    n.weights(0)(0,0) = 0d
    n.biases(0)(0) = 0d
    n.feedForward(DenseVector(0d)) should be (DenseVector(0.5d))
    n.feedForward(DenseVector(1d)) should be (DenseVector(0.5d))
  }

  it should "learn NOT" in {
    val n = new Network(Seq(1, 1, 1))
    n.randomize()
    val r = new scala.util.Random()
    val trainingData = Seq.fill(10000) {
      val i = r.nextInt(2).toDouble
      (DenseVector(i), DenseVector(1 - i))
    }
    n.stochasticGradientDescent(trainingData, 10, 100, 8.0)
    n.feedForward(DenseVector(1.0d))(0) should be < 0.1
    n.feedForward(DenseVector(0.0d))(0) should be > 0.9
  }

  it should "learn AND" in {
    shouldLearnBooleanFunction2(_ & _)
  }

  it should "learn XOR" in {
    shouldLearnBooleanFunction2(_ ^ _, hiddenLayers = Seq(2,2))
  }

  it should "learn NAND" in {
    shouldLearnBooleanFunction2((a, b) => !(a & b))
  }

  def shouldLearnBooleanFunction2(
    f: (Boolean, Boolean) => Boolean,
    hiddenLayers: Seq[Int] = Seq(1),
    epochs: Int = 10
  ): Unit = {
    implicit def boolToDouble(b: Boolean): Double = if (b) 1.0 else 0.0
    val n = new Network(2 +: hiddenLayers :+ 1)
    n.randomize()
    val r = new scala.util.Random()
    val trainingData = Seq.fill(10000) {
      val a = r.nextBoolean()
      val b = r.nextBoolean()
      (DenseVector(a.toDouble, b.toDouble), DenseVector(f(a, b).toDouble))
    }
    n.stochasticGradientDescent(trainingData, epochs, 100, 30.0)
    for (a <- Seq(false, true); b <- Seq(false, true)) {
      val c = n.feedForward(DenseVector(a, b))(0)
      if (f(a, b))
        c should be > 0.9
      else
        c should be < 0.1
    }
  }
}
