package scalann

import breeze.linalg._
import breeze.numerics.sigmoid

class Network(sizes: Seq[Int]) {
  assert(sizes.size >= 2, s"Must have at least an input and an output layer")

  val weights: Seq[DenseMatrix[Double]] =
    sizes.sliding(2).map {
      case List(p, n) => DenseMatrix.zeros[Double](n, p)
    }.toSeq

  val biases: Seq[DenseVector[Double]] =
    sizes.drop(1) map { s => DenseVector.zeros[Double](s) }

  def randomize(): Unit = {
    import breeze.stats.distributions.Gaussian
    val g = new Gaussian(0, 1)
    weights foreach { w => for (r <- 0 until w.rows; c <- 0 until w.cols) w(r,c) = g.sample() }
    biases foreach { b => for (j <- 0 until b.length) b(j) = g.sample() }
  }

  def feedForward(inputs: DenseVector[Double]): DenseVector[Double] = {
    (weights zip biases).foldLeft(inputs) { case (a, (w, b)) => sigmoid((w * a) + b) }
  }

  def backPropagate(
    x: DenseVector[Double],
    y: DenseVector[Double]
  ): (Seq[DenseVector[Double]], Seq[DenseMatrix[Double]]) = {
    lazy val activations: Stream[DenseVector[Double]] = x #:: zs.map(z => sigmoid(z))
    lazy val zs: Stream[DenseVector[Double]] = (activations zip (weights zip biases)).map { case (a, (w, b)) => (w * a) + b }
    assert(zs.size == weights.size)
    assert(activations.size == weights.size + 1)
    val outputError = costDerivative(activations.last, y) :* sigmoidDerivative(zs.last)
    val errors = (weights.reverse zip zs.reverse.drop(1)).scanLeft(outputError) {
      case (err, (w, z)) =>
        (w.t * err) :* sigmoidDerivative(z)
    }.reverse
    val dWeights = (errors zip activations) map { case (err, a) => err * a.t }
    (errors, dWeights)
  }

  def updateMiniBatch(batch: Seq[(DenseVector[Double], DenseVector[Double])], learningRate: Double): Unit = {
    val updateScale = learningRate / batch.size
    val totalDB = biases.map { b => DenseVector.zeros[Double](b.length) }
    val totalDW = weights.map { w => DenseMatrix.zeros[Double](w.rows, w.cols) }
    for ((input, expectedOutput) <- batch) {
      val (dBiases, dWeights) = backPropagate(input, expectedOutput)
      for ((b, db) <- totalDB zip dBiases) { b :+= db }
      for ((w, dw) <- totalDW zip dWeights) { w :+= dw }
    }
    for ((b, db) <- biases zip totalDB) { b :-= updateScale * db }
    for ((w, dw) <- weights zip totalDW) { w :-= updateScale * dw }
  }

  def stochasticGradientDescent(
    trainingData: Seq[(DenseVector[Double], DenseVector[Double])],
    epochs: Int,
    miniBatchSize: Int,
    learningRate: Double,
    testData: Seq[(DenseVector[Double], DenseVector[Double])] = Seq.empty
  ): Unit = {
    val r = new scala.util.Random()
    for (epoch <- 1 to epochs) {
      val shuffledTrainingData = r.shuffle(trainingData)
      val miniBatches = shuffledTrainingData.grouped(miniBatchSize)
      for (batch <- miniBatches) updateMiniBatch(batch, learningRate)
      println(s"Epoch $epoch: ${evaluate(testData)}/${testData.size} correct")
    }
  }

  def evaluate(testData: Seq[(DenseVector[Double], DenseVector[Double])]): Int = {
    testData.count {
      case (input, expectedOutput) =>
        argmax(feedForward(input)) == argmax(expectedOutput)
    }
  }

  def cost(a: DenseVector[Double], y: DenseVector[Double]): DenseVector[Double] = 0.5d * ((a - y) :^ 2.0d)
  def costDerivative(a: DenseVector[Double], y: DenseVector[Double]): DenseVector[Double] = a - y
  def sigmoidDerivative(t: DenseVector[Double]): DenseVector[Double] = sigmoid(t) :* (1.0d - sigmoid(t))
}
