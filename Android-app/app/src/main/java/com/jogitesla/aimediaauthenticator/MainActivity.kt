package com.jogitesla.aimediaauthenticator

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.jogitesla.aimediaauthenticator.ui.theme.AIMediaAuthenticatorTheme
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import java.nio.FloatBuffer
import ai.onnxruntime.OnnxTensor
import java.io.File
import java.io.FileOutputStream
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.ui.Alignment
import androidx.activity.result.contract.ActivityResultContracts
import android.net.Uri
import android.provider.MediaStore
import android.graphics.Bitmap
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.math.max
import android.graphics.Color
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.unit.dp
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue

class MainActivity : ComponentActivity() {
    private lateinit var ortEnv: OrtEnvironment
    private lateinit var ortSession: OrtSession

    private var resultTextState: ((String) -> Unit)? = null


    private val imagePicker =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                processSelectedImage(it)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize ONNX Runtime
        try {
            ortEnv = OrtEnvironment.getEnvironment()

            // Copy model files to internal storage
            val modelFile = copyAssetToFile("fusion_model.onnx")
            // Ensure the data file is also copied if your model relies on external data
            copyAssetToFile("fusion_model.onnx.data")

            // Load model using file path
            ortSession = ortEnv.createSession(modelFile.absolutePath)

            Log.d("ONNX", "Model loaded successfully")

            // Run dummy inference to warm up the model
//            runDummyInference()

        } catch (e: Exception) {
            Log.e("ONNX", "Failed to initialize ONNX", e)
        }

        // UI
        enableEdgeToEdge()
        setContent {
            AIMediaAuthenticatorTheme {
                var resultText by remember {
                    mutableStateOf("No image selected")
                }

                // connect non-UI code → UI
                resultTextState = { newText ->
                    resultText = newText
                }
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Column(
                        modifier = Modifier
                            .padding(innerPadding)
                            .fillMaxSize(),
                        verticalArrangement = Arrangement.Center,
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Button(onClick = { openGallery() }) {
                            Text("Select Image from Gallery")
                        }
                        Spacer(modifier = Modifier.height(16.dp))

                        Text(text = resultText)
                    }
                }
            }
        }
    }

    private fun openGallery() {
        imagePicker.launch("image/*")
    }

    private fun processSelectedImage(uri: Uri) {
        try {
            val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)

            val imageData = bitmapToFloatArray(bitmap)
            val fftData = extractFFTFeatures(bitmap)

            //DEBUG
            val minFft = fftData.minOrNull()
            val maxFft = fftData.maxOrNull()
            val meanFft = fftData.average()

            Log.d(
                "FFT_ANDROID",
                "FFT stats → min=$minFft max=$maxFft mean=$meanFft"
            )


            val imageTensor = OnnxTensor.createTensor(
                ortEnv,
                FloatBuffer.wrap(imageData),
                longArrayOf(1, 3, 224, 224)
            )

            val fftTensor = OnnxTensor.createTensor(
                ortEnv,
                FloatBuffer.wrap(fftData),
                longArrayOf(1, 1024)
            )

            val inputs = mapOf(
                "images" to imageTensor,
                "fft" to fftTensor
            )

            val output = ortSession.run(inputs)
            val logits = output[0].value as Array<FloatArray>

            val prediction =
                if (logits[0][0] > logits[0][1]) "REAL IMAGE"
                else "AI GENERATED"

            resultTextState?.invoke(prediction)

            Log.d("RESULT", "Prediction: $prediction")

            val exp0 = kotlin.math.exp(logits[0][0].toDouble())
            val exp1 = kotlin.math.exp(logits[0][1].toDouble())
            val sum = exp0 + exp1

            val aiConfidence = (exp1 / sum * 100).toInt()
            val realConfidence = (exp0 / sum * 100).toInt()

            Log.d("RESULT", "AI Confidence: $aiConfidence%")
            Log.d("RESULT", "REAL Confidence: $realConfidence%")

        } catch (e: Exception) {
            Log.e("IMAGE", "Inference failed", e)
        }
    }

    private fun extractFFTFeatures(bitmap: Bitmap): FloatArray {
        val gray = bitmapToGrayscaleMatrix(bitmap)

        // FFT (complex magnitude)
        // Ensure you have the FFTUtils class defined in the package below
        val fftMag = com.jogitesla.aimediaauthenticator.fft.FFTUtils.fft2D(gray)

        val size = 224
        val shifted = Array(size) { DoubleArray(size) }

        // FFT shift (center)
        val half = size / 2
        for (y in 0 until size) {
            for (x in 0 until size) {
                val newY = (y + half) % size
                val newX = (x + half) % size
                shifted[newY][newX] = fftMag[y][x]
            }
        }

        // Log magnitude + max normalization
        var maxVal = 0.0
        for (y in 0 until size) {
            for (x in 0 until size) {
                shifted[y][x] = ln(shifted[y][x] + 1.0)
                maxVal = max(maxVal, shifted[y][x])
            }
        }

        for (y in 0 until size) {
            for (x in 0 until size) {
                shifted[y][x] /= maxVal
            }
        }

        // Downsample to 32x32 -> 1024 features
        val features = DoubleArray(1024)
        var idx = 0
        val step = size / 32

        for (y in 0 until size step step) {
            for (x in 0 until size step step) {
                if (idx < 1024) { // Safety check
                    features[idx++] = shifted[y][x]
                }
            }
        }

        // Mean–Std normalization
        val mean = features.average()
        var variance = 0.0
        for (v in features) variance += (v - mean) * (v - mean)
        val std = sqrt(variance / features.size + 1e-8)

        return FloatArray(1024) { i ->
            ((features[i] - mean) / std).toFloat()
        }
    }

    private fun bitmapToFloatArray(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        val floatArray = FloatArray(3 * 224 * 224)
        var idx = 0

        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = resized.getPixel(x, y)

                val r = ((pixel shr 16) and 0xFF) / 255f
                val g = ((pixel shr 8) and 0xFF) / 255f
                val b = (pixel and 0xFF) / 255f

                floatArray[idx++] = r
                floatArray[idx++] = g
                floatArray[idx++] = b
            }
        }
        return floatArray
    }

    private fun bitmapToGrayscaleMatrix(bitmap: Bitmap): Array<Array<Double>> {
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val gray = Array(224) { Array(224) { 0.0 } }

        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = resized.getPixel(x, y)

                val r = Color.red(pixel)
                val g = Color.green(pixel)
                val b = Color.blue(pixel)

                // Same luminance logic OpenCV uses internally
                gray[y][x] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            }
        }
        return gray
    }

    // Dummy inference to warm up the model
//    private fun runDummyInference() {
//        try {
//            // 1. Prepare dummy image input (Batch, Channel, Height, Width)
//            val imageData = FloatArray(1 * 3 * 224 * 224) { 0.5f }
//            val imageBuffer = FloatBuffer.wrap(imageData)
//
//            val imageTensor = OnnxTensor.createTensor(
//                ortEnv,
//                imageBuffer,
//                longArrayOf(1, 3, 224, 224)
//            )
//
//            // 2. Prepare dummy FFT input (Batch, Features)
//            // FIX: Use a manually created FloatArray instead of calling extractFFTFeatures(bitmap)
//            val fftData = FloatArray(1024) { 0.0f }
//            val fftBuffer = FloatBuffer.wrap(fftData)
//
//            val fftTensor = OnnxTensor.createTensor(
//                ortEnv,
//                fftBuffer,
//                longArrayOf(1, 1024)
//            )
//
//            // 3. Explicitly typed input map
//            val inputs: Map<String, OnnxTensor> = mapOf(
//                "images" to imageTensor,
//                "fft" to fftTensor
//            )
//
//            // 4. Run inference
//            val output = ortSession.run(inputs)
//
//            val logits = output[0].value as Array<FloatArray>
//            Log.d("ONNX", "Dummy Inference Logits: ${logits[0][0]}, ${logits[0][1]}")
//
//        } catch (e: Exception) {
//            Log.e("ONNX", "Dummy Inference failed", e)
//        }
//    }

    private fun copyAssetToFile(assetName: String): File {
        val outFile = File(filesDir, assetName)

        if (!outFile.exists()) {
            assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
        }
        return outFile
    }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    AIMediaAuthenticatorTheme {
        Greeting("Android")
    }
}