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

class MainActivity : ComponentActivity() {
    private lateinit var ortEnv: OrtEnvironment
    private lateinit var ortSession: OrtSession
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Initialize ONNX Runtime
        try {
            ortEnv = OrtEnvironment.getEnvironment()

            // Copy model files to internal storage
            val modelFile = copyAssetToFile("fusion_model.onnx")
            copyAssetToFile("fusion_model.onnx.data")

            // Load model using file path (REQUIRED for external data)
            ortEnv = OrtEnvironment.getEnvironment()
            ortSession = ortEnv.createSession(modelFile.absolutePath)

            Log.d("ONNX", "Model loaded successfully from file")

            Log.d("ONNX", "Model Loaded successfully")

            // Run dummy inference
            runDummyInference()
        } catch (e: Exception) {
            Log.e("ONNX", "Failed too initialize ONNX", e)
        }

        // UI
        enableEdgeToEdge()
        setContent {
            AIMediaAuthenticatorTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Greeting(
                        name = "Android",
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }

    // Add dummy inference
    private fun runDummyInference() {
        try {
            // 1️⃣ Prepare dummy image input
            val imageData = FloatArray(1 * 3 * 224 * 224) { 0.5f }
            val imageBuffer = FloatBuffer.wrap(imageData)

            val imageTensor = OnnxTensor.createTensor(
                ortEnv,
                imageBuffer,
                longArrayOf(1, 3, 224, 224)
            )

            // 2️⃣ Prepare dummy FFT input
            val fftData = FloatArray(1024) { 0.0f }
            val fftBuffer = FloatBuffer.wrap(fftData)

            val fftTensor = OnnxTensor.createTensor(
                ortEnv,
                fftBuffer,
                longArrayOf(1, 1024)
            )

            // 3️⃣ Explicitly typed input map (fixes inference errors)
            val inputs: Map<String, OnnxTensor> = mapOf(
                "images" to imageTensor,
                "fft" to fftTensor
            )

            // 4️⃣ Run inference
            val output = ortSession.run(inputs)

            val logits = output[0].value as Array<FloatArray>
            Log.d("ONNX", "Logits: ${logits[0][0]}, ${logits[0][1]}")

        } catch (e: Exception) {
            Log.e("ONNX", "Inference failed", e)
        }
    }

    // ADD FILE COPY HELPER FUNCTION
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