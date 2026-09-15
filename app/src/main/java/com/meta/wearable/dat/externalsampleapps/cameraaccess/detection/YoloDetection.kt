/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.detection

/** A detected object whose bounds are normalized to the original camera frame. */
data class YoloDetection(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val label: String,
    val confidence: Float,
    val classIndex: Int,
)

data class YoloResult(
    val detections: List<YoloDetection>,
    val inferenceTimeMs: Long,
    val accelerator: String,
)
