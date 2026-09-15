# Camera Access App

A sample Android application demonstrating integration with Meta Wearables Device Access Toolkit. This app showcases streaming video from Meta AI glasses, capturing photos, and managing connection states.

## Features

- Connect to Meta AI glasses
- Stream camera feed from the device
- Run real-time YOLO object detection over the wearable camera stream
- Capture photos from glasses
- Share captured photos
- Open firmware and DAT glasses app update flows when required

## Prerequisites

- Android Studio Narwhal (2025.1.1) or newer
- JDK 17 or newer
- Android SDK 36 or newer
- Meta Wearables Device Access Toolkit (included as a dependency)
- A Meta AI glasses device for testing (optional for development)

## Building the app

### Using Android Studio

1. Clone this repository
1. Open the project in Android Studio
1. Add your personal access token (classic) to the `local.properties` file (see [SDK for Android setup](https://wearables.developer.meta.com/docs/develop/dat/build-integration-android#step-2-add-the-sdk-to-gradle))
1. Click **File** > **Sync Project with Gradle Files**
1. Click **Run** > **Run...** > **app**

## Running the app

1. Turn 'Developer Mode' on in the Meta AI app.
1. Launch the app.
1. Press the "Connect" button to complete app registration.
1. Once connected, the camera stream from the device will be displayed
1. Use the on-screen controls to:
   - View YOLO bounding boxes, labels, inference latency, and the active accelerator
   - Capture photos
   - View and save captured photos
   - Disconnect from the device
1. If a firmware update is required, tap "Update firmware" from the connection screen.
1. If session start reports that the app on the glasses is outdated, tap "Update app on glasses" from the connection screen.

## Real-time object detection

The app bundles a 640 x 640 YOLO26n LiteRT model trained on the 80 COCO classes. Inference runs on a dedicated background thread and is throttled independently from the 24 FPS wearable stream so rendering remains responsive. The runtime attempts GPU acceleration first and falls back to CPU when the phone does not support the model on its GPU.

The model and labels live in `app/src/main/assets`. To use a custom detection model, replace the model and labels and update the tensor constants in `YoloDetector.kt`. The bundled model expects a float NCHW `[1, 3, 640, 640]` input and returns `[1, 84, 8400]` predictions.

See `THIRD_PARTY_NOTICES.md` before redistributing the bundled model.

## Troubleshooting

For issues related to the Meta Wearables Device Access Toolkit, please refer to the [developer documentation](https://wearables.developer.meta.com/docs/develop/dat/) or visit our [discussions forum](https://github.com/facebook/meta-wearables-dat-android/discussions)

## License

This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.
