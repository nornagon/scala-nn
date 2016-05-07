package scalann

import breeze.linalg.DenseVector
import java.io.DataInputStream
import java.util.zip.GZIPInputStream


object MNISTHandwriting {
  def gzipSource(f: String) = new DataInputStream(new GZIPInputStream(getClass.getResourceAsStream(f)))

  def main(args: Array[String]): Unit = {
    val trainingImages = readImages("train-images-idx3-ubyte.gz")
    val trainingLabels = readLabels("train-labels-idx1-ubyte.gz")
    val testImages = readImages("t10k-images-idx3-ubyte.gz")
    val testLabels = readLabels("t10k-labels-idx1-ubyte.gz")

    val network = new Network(Seq(trainingImages.head.size, 100, 10))
    network.randomize()
    network.stochasticGradientDescent(trainingImages zip trainingLabels, 30, 10, 3.0, testImages zip testLabels)
  }

  def readImages(fileName: String): Seq[DenseVector[Double]] = {
    val inp = gzipSource(fileName)
    val magic = inp.readInt()
    assert(magic == 0x00000803)
    val numImages = inp.readInt()
    val numRows = inp.readInt()
    val numCols = inp.readInt()
    val a = new Array[Byte](numRows * numCols)
    for (i <- 1 to numImages) yield {
      inp.readFully(a)
      DenseVector(a.map(b => (b & 0xff) / 255.0d))
    }
  }

  def readLabels(fileName: String): Seq[DenseVector[Double]] = {
    val inp = gzipSource(fileName)
    val magic = inp.readInt()
    assert(magic == 0x00000801)
    val numLabels = inp.readInt()
    for (i <- 1 to numLabels) yield {
      val label = inp.readByte()
      val v = DenseVector.zeros[Double](10)
      v(label) = 1.0d
      v
    }
  }

}
