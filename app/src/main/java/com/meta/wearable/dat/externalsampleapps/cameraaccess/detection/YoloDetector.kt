/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max
import kotlin.math.min

/** Runs the bundled COCO YOLO26n LiteRT model on camera frames. */
class YoloDetector(context: Context) : AutoCloseable {
  companion object {
    private const val TAG = "CameraAccess:YoloDetector"
    private const val MODEL_ASSET = "yolo26n_w8a32.tflite"
    private const val LABELS_ASSET = "coco_labels.txt"
    private const val INPUT_SIZE = 640
    private const val CHANNEL_COUNT = 84
    private const val BOX_CHANNEL_COUNT = 4
    private const val PREDICTION_COUNT = 8400
    private const val CONFIDENCE_THRESHOLD = 0.35f
    private const val IOU_THRESHOLD = 0.45f
    private const val MAX_DETECTIONS = 30
    private const val PADDING_COLOR = 114
  }

  private data class PreparedModel(
      val compiledModel: CompiledModel,
      val inputBuffers: List<TensorBuffer>,
      val outputBuffers: List<TensorBuffer>,
      val accelerator: String,
  )

  private data class LetterboxTransform(
      val scale: Float,
      val offsetX: Float,
      val offsetY: Float,
      val sourceWidth: Int,
      val sourceHeight: Int,
  )

  private val labels =
      context.assets.open(LABELS_ASSET).bufferedReader().useLines { lines ->
        lines.filter { it.isNotBlank() }.toList()
      }
  private val modelFile = copyModelToCodeCache(context)
  private val preparedModel = prepareModel(modelFile)
  private val model = preparedModel.compiledModel
  private val inputBuffers = preparedModel.inputBuffers
  private val outputBuffers = preparedModel.outputBuffers
  val accelerator: String = preparedModel.accelerator

  private val inputBitmap =
      Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
  private val inputPixels = IntArray(INPUT_SIZE * INPUT_SIZE)
  private val inputTensor = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)
  private val resizePaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)

  init {
    require(labels.size == CHANNEL_COUNT - BOX_CHANNEL_COUNT) {
      "Expected ${CHANNEL_COUNT - BOX_CHANNEL_COUNT} labels but found ${labels.size}"
    }
  }

  fun detect(bitmap: Bitmap): YoloResult {
    val startedAt = SystemClock.elapsedRealtimeNanos()
    val transform = preprocess(bitmap)
    inputBuffers[0].writeFloat(inputTensor)
    model.run(inputBuffers, outputBuffers)
    val output = outputBuffers[0].readFloat()
    check(output.size == CHANNEL_COUNT * PREDICTION_COUNT) {
      "Unexpected YOLO output size ${output.size}; expected ${CHANNEL_COUNT * PREDICTION_COUNT}"
    }
    val detections = postprocess(output, transform)
    val elapsedMs = (SystemClock.elapsedRealtimeNanos() - startedAt) / 1_000_000L
    return YoloResult(detections, elapsedMs, accelerator)
  }

  private fun preprocess(source: Bitmap): LetterboxTransform {
    val scale = min(INPUT_SIZE.toFloat() / source.width, INPUT_SIZE.toFloat() / source.height)
    val scaledWidth = source.width * scale
    val scaledHeight = source.height * scale
    val offsetX = (INPUT_SIZE - scaledWidth) / 2f
    val offsetY = (INPUT_SIZE - scaledHeight) / 2f

    val canvas = Canvas(inputBitmap)
    canvas.drawColor(Color.rgb(PADDING_COLOR, PADDING_COLOR, PADDING_COLOR))
    canvas.drawBitmap(
        source,
        null,
        RectF(offsetX, offsetY, offsetX + scaledWidth, offsetY + scaledHeight),
        resizePaint,
    )
    inputBitmap.getPixels(inputPixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

    val planeSize = INPUT_SIZE * INPUT_SIZE
    for (index in inputPixels.indices) {
      val color = inputPixels[index]
      inputTensor[index] = Color.red(color) / 255f
      inputTensor[planeSize + index] = Color.green(color) / 255f
      inputTensor[2 * planeSize + index] = Color.blue(color) / 255f
    }

    return LetterboxTransform(scale, offsetX, offsetY, source.width, source.height)
  }

  private fun postprocess(
      output: FloatArray,
      transform: LetterboxTransform,
  ): List<YoloDetection> {
    val candidates = ArrayList<YoloDetection>()
    for (prediction in 0 until PREDICTION_COUNT) {
      var bestClass = -1
      var bestConfidence = CONFIDENCE_THRESHOLD
      for (classIndex in labels.indices) {
        val confidence = output[(BOX_CHANNEL_COUNT + classIndex) * PREDICTION_COUNT + prediction]
        if (confidence > bestConfidence) {
          bestConfidence = confidence
          bestClass = classIndex
        }
      }
      if (bestClass < 0) continue

      val centerX = output[prediction]
      val centerY = output[PREDICTION_COUNT + prediction]
      val width = output[2 * PREDICTION_COUNT + prediction]
      val height = output[3 * PREDICTION_COUNT + prediction]
      // YOLO26 LiteRT outputs normalized xywh coordinates. Convert them back to the
      // 640 x 640 model-input coordinate space before undoing the letterbox transform.
      val centerXPixels = centerX * INPUT_SIZE
      val centerYPixels = centerY * INPUT_SIZE
      val widthPixels = width * INPUT_SIZE
      val heightPixels = height * INPUT_SIZE
      val left = ((centerXPixels - widthPixels / 2f - transform.offsetX) / transform.scale)
          .coerceIn(0f, transform.sourceWidth.toFloat())
      val top = ((centerYPixels - heightPixels / 2f - transform.offsetY) / transform.scale)
          .coerceIn(0f, transform.sourceHeight.toFloat())
      val right = ((centerXPixels + widthPixels / 2f - transform.offsetX) / transform.scale)
          .coerceIn(0f, transform.sourceWidth.toFloat())
      val bottom = ((centerYPixels + heightPixels / 2f - transform.offsetY) / transform.scale)
          .coerceIn(0f, transform.sourceHeight.toFloat())
      if (right <= left || bottom <= top) continue

      candidates +=
          YoloDetection(
              left = left / transform.sourceWidth,
              top = top / transform.sourceHeight,
              right = right / transform.sourceWidth,
              bottom = bottom / transform.sourceHeight,
              label = labels[bestClass],
              confidence = bestConfidence,
              classIndex = bestClass,
          )
    }

    return nonMaximumSuppression(candidates)
  }

  private fun nonMaximumSuppression(candidates: List<YoloDetection>): List<YoloDetection> {
    val sorted = candidates.sortedByDescending { it.confidence }
    val selected = ArrayList<YoloDetection>(min(MAX_DETECTIONS, sorted.size))
    for (candidate in sorted) {
      val overlaps =
          selected.any { chosen ->
            chosen.classIndex == candidate.classIndex && intersectionOverUnion(chosen, candidate) > IOU_THRESHOLD
          }
      if (!overlaps) {
        selected += candidate
        if (selected.size == MAX_DETECTIONS) break
      }
    }
    return selected
  }

  private fun intersectionOverUnion(first: YoloDetection, second: YoloDetection): Float {
    val intersectionLeft = max(first.left, second.left)
    val intersectionTop = max(first.top, second.top)
    val intersectionRight = min(first.right, second.right)
    val intersectionBottom = min(first.bottom, second.bottom)
    val intersectionArea =
        max(0f, intersectionRight - intersectionLeft) * max(0f, intersectionBottom - intersectionTop)
    val firstArea = (first.right - first.left) * (first.bottom - first.top)
    val secondArea = (second.right - second.left) * (second.bottom - second.top)
    val unionArea = firstArea + secondArea - intersectionArea
    return if (unionArea > 0f) intersectionArea / unionArea else 0f
  }

  private fun prepareModel(modelFile: File): PreparedModel {
    return try {
      compile(modelFile, Accelerator.GPU, "GPU")
    } catch (gpuError: Throwable) {
      Log.w(TAG, "GPU initialization failed; falling back to CPU", gpuError)
      compile(modelFile, Accelerator.CPU, "CPU")
    }
  }

  private fun compile(modelFile: File, target: Accelerator, name: String): PreparedModel {
    val options = CompiledModel.Options(target)
    if (target == Accelerator.GPU) {
      options.gpuOptions =
          CompiledModel.GpuOptions(
              serializationDir = modelFile.parentFile?.absolutePath,
              modelCacheKey = "${modelFile.name}_${modelFile.length()}",
              serializeProgramCache = true,
          )
    }
    val compiledModel = CompiledModel.create(modelFile.absolutePath, options)
    val inputs = compiledModel.createInputBuffers()
    val outputs = compiledModel.createOutputBuffers()
    return try {
      val inputShape =
          compiledModel.getInputTensorType(inputName = "args_0").layout?.dimensions?.toIntArray()
      val outputShape =
          compiledModel.getOutputTensorType(outputName = "output_0").layout?.dimensions?.toIntArray()
      check(inputShape?.contentEquals(intArrayOf(1, 3, INPUT_SIZE, INPUT_SIZE)) == true) {
        "Unsupported YOLO input shape: ${inputShape?.toList()}"
      }
      check(outputShape?.contentEquals(intArrayOf(1, CHANNEL_COUNT, PREDICTION_COUNT)) == true) {
        "Unsupported YOLO output shape: ${outputShape?.toList()}"
      }

      inputs[0].writeFloat(FloatArray(3 * INPUT_SIZE * INPUT_SIZE))
      compiledModel.run(inputs, outputs)
      check(outputs[0].readFloat().size == CHANNEL_COUNT * PREDICTION_COUNT)
      Log.i(TAG, "YOLO model initialized with $name acceleration")
      PreparedModel(compiledModel, inputs, outputs, name)
    } catch (error: Throwable) {
      inputs.forEach { runCatching { it.close() } }
      outputs.forEach { runCatching { it.close() } }
      runCatching { compiledModel.close() }
      throw error
    }
  }

  private fun copyModelToCodeCache(context: Context): File {
    val modelDirectory = File(context.codeCacheDir, "yolo").apply { mkdirs() }
    val destination = File(modelDirectory, MODEL_ASSET)
    val assetLength = context.assets.openFd(MODEL_ASSET).use { it.length }
    if (!destination.exists() || destination.length() != assetLength) {
      context.assets.open(MODEL_ASSET).use { input ->
        FileOutputStream(destination).use { output -> input.copyTo(output) }
      }
    }
    return destination
  }

  override fun close() {
    inputBitmap.recycle()
    inputBuffers.forEach { runCatching { it.close() } }
    outputBuffers.forEach { runCatching { it.close() } }
    runCatching { model.close() }
  }
}
