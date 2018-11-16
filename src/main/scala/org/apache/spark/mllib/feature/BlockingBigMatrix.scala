package org.apache.spark.mllib.feature

import java.util.concurrent.{Semaphore, TimeUnit}

import breeze.linalg.Vector
import glint.models.client.BigMatrix
import org.apache.spark.internal.Logging

import scala.concurrent.{ExecutionContext, Future, blocking}
import scala.reflect.ClassTag


class BlockingBigMatrix[V: ClassTag](underlying: BigMatrix[V], maxSimultaneousRequests: Int)
                                    (implicit ec: ExecutionContext) extends BigMatrix[V] with Logging {

  require(maxSimultaneousRequests > 0, "Max simultaneous requests must be non-zero")

  val rows: Long = underlying.rows
  val cols: Int = underlying.cols

  val semaphore = new Semaphore(maxSimultaneousRequests)

  private def blockingRequest[W](request: () => Future[W])(implicit ec: ExecutionContext): Future[W] = {
    blocking({
      while(!semaphore.tryAcquire(1, TimeUnit.MINUTES)) { logInfo("matrix blocking for more than a minute") }
      request().transform(success => {
        semaphore.release()
        success
      }, err => {
        semaphore.release()
        err
      })
    })
  }

  override def pull(rows: Array[Long])(implicit ec: ExecutionContext): Future[Array[Vector[V]]] = {
    blockingRequest(() => underlying.pull(rows))
  }

  override def destroy()(implicit ec: ExecutionContext): Future[Boolean] = {
    waitNoRequests()
    underlying.destroy()
  }

  override def push(rows: Array[Long],
                    cols: Array[Int],
                    values: Array[V])(implicit ec: ExecutionContext): Future[Boolean] = {
    blockingRequest(() => underlying.push(rows, cols, values))
  }

  override def pull(rows: Array[Long],
                    cols: Array[Int])(implicit ec: ExecutionContext): Future[Array[V]] = {
    blockingRequest(() => underlying.pull(rows, cols))
  }

  def waitNoRequests(): Unit = {
    blocking({
      while(!semaphore.tryAcquire(maxSimultaneousRequests, 1, TimeUnit.MINUTES)) { "matrix blocking for more than a minute waiting until no requests" }
      semaphore.release(maxSimultaneousRequests)
    })
  }
}
