/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.ui

import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.unit.dp
import com.meta.wearable.dat.externalsampleapps.cameraaccess.detection.YoloDetection
import kotlin.math.max
import kotlin.math.roundToInt

/** Draws normalized detection bounds over an image displayed with ContentScale.Crop. */
@Composable
fun DetectionOverlay(
    detections: List<YoloDetection>,
    frameWidth: Int,
    frameHeight: Int,
    modifier: Modifier = Modifier,
) {
  Canvas(modifier = modifier) {
    if (frameWidth <= 0 || frameHeight <= 0) return@Canvas

    val imageScale = max(size.width / frameWidth, size.height / frameHeight)
    val displayedWidth = frameWidth * imageScale
    val displayedHeight = frameHeight * imageScale
    val offsetX = (size.width - displayedWidth) / 2f
    val offsetY = (size.height - displayedHeight) / 2f
    val strokeWidth = 3.dp.toPx()
    val labelPadding = 4.dp.toPx()
    val textPaint =
        Paint(Paint.ANTI_ALIAS_FLAG).apply {
          color = android.graphics.Color.WHITE
          textSize = 13.dp.toPx()
          typeface = android.graphics.Typeface.DEFAULT_BOLD
        }
    val backgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG)

    detections.forEach { detection ->
      val color = detectionColor(detection.classIndex)
      val left = offsetX + detection.left * frameWidth * imageScale
      val top = offsetY + detection.top * frameHeight * imageScale
      val right = offsetX + detection.right * frameWidth * imageScale
      val bottom = offsetY + detection.bottom * frameHeight * imageScale

      drawRect(
          color = color,
          topLeft = Offset(left, top),
          size = Size(right - left, bottom - top),
          style = Stroke(width = strokeWidth),
      )

      val text = "${detection.label} ${(detection.confidence * 100).roundToInt()}%"
      val textWidth = textPaint.measureText(text)
      val textHeight = textPaint.fontMetrics.run { bottom - top }
      val labelLeft = left.coerceIn(0f, max(0f, size.width - textWidth - 2 * labelPadding))
      val labelTop = max(0f, top - textHeight - 2 * labelPadding)
      backgroundPaint.color = color.copy(alpha = 0.88f).toArgb()
      drawContext.canvas.nativeCanvas.drawRoundRect(
          labelLeft,
          labelTop,
          labelLeft + textWidth + 2 * labelPadding,
          labelTop + textHeight + 2 * labelPadding,
          labelPadding,
          labelPadding,
          backgroundPaint,
      )
      drawContext.canvas.nativeCanvas.drawText(
          text,
          labelLeft + labelPadding,
          labelTop + labelPadding - textPaint.fontMetrics.top,
          textPaint,
      )
    }
  }
}

private fun detectionColor(classIndex: Int): Color {
  val hue = ((classIndex * 47) % 360).toFloat()
  return Color.hsv(hue = hue, saturation = 0.78f, value = 1f)
}
