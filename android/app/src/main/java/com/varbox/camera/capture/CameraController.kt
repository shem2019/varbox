package com.varbox.camera.capture

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Handler
import android.os.Looper
import android.util.Range
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.varbox.camera.clock.SharedClock
import com.varbox.camera.rig.RigSettings
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.coroutines.resume

/**
 * Drives one phone's camera through CameraX, with Camera2 interop for the fixed frame rate and the
 * locked exposure, focus and white balance. An image-analysis stream records each frame's capture
 * time on the shared clock for the sidecar; recording begins the moment the start command arrives
 * and the frames carry absolute shared timestamps, so the message delay and any pre-roll wash out.
 */
@OptIn(ExperimentalCamera2Interop::class)
class CameraController(private val context: Context) {

    private var provider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null

    private val frameTimesMs = CopyOnWriteArrayList<Long>()
    @Volatile private var collecting = false
    @Volatile private var recordedDurationMs = 0L
    private var realtimeTimestamps = false

    private lateinit var settings: RigSettings
    private var appliedSettings: Map<String, Any?> = emptyMap()

    // Set when recording starts.
    private var recordingId = ""
    private var round = ""
    private var letter = "A"
    private var startAtMs = 0L
    private var outputFile: File? = null
    private var finalized: CompletableDeferred<File>? = null

    private val mainHandler = Handler(Looper.getMainLooper())

    suspend fun bind(
        lifecycleOwner: LifecycleOwner,
        previewView: PreviewView,
        settings: RigSettings,
        capabilities: Capabilities,
    ): Map<String, Any?> {
        this.settings = settings
        readTimestampSource()

        val provider = awaitProvider()
        this.provider = provider

        val preview = Preview.Builder().build()
        preview.setSurfaceProvider(previewView.surfaceProvider)
        val recorder = Recorder.Builder()
            .setQualitySelector(QualitySelector.from(Quality.FHD))
            .build()
        val videoCapture = VideoCapture.withOutput(recorder)
        this.videoCapture = videoCapture

        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
        analysis.setAnalyzer(ContextCompat.getMainExecutor(context)) { image ->
            if (collecting) {
                frameTimesMs.add(
                    SharedClock.sharedFromSensorNs(image.imageInfo.timestamp, realtimeTimestamps)
                )
            }
            image.close()
        }

        provider.unbindAll()
        camera = provider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            videoCapture,
            analysis,
        )
        appliedSettings = applySettings(settings, capabilities)
        return appliedSettings
    }

    private suspend fun awaitProvider(): ProcessCameraProvider =
        suspendCancellableCoroutine { cont ->
            val future = ProcessCameraProvider.getInstance(context)
            future.addListener({ cont.resume(future.get()) }, ContextCompat.getMainExecutor(context))
        }

    private fun readTimestampSource() {
        runCatching {
            val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val backId = manager.cameraIdList.firstOrNull { id ->
                manager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING) ==
                    CameraCharacteristics.LENS_FACING_BACK
            } ?: manager.cameraIdList.first()
            val source = manager.getCameraCharacteristics(backId)
                .get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
            realtimeTimestamps =
                source == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
        }
    }

    /** Push the locked settings through Camera2 and report what held and what the phone refused. */
    private fun applySettings(s: RigSettings, caps: Capabilities): Map<String, Any?> {
        val control = camera?.let { Camera2CameraControl.from(it.cameraControl) }
            ?: return mapOf("error" to "no camera control")
        val applied = linkedMapOf<String, Any?>()
        val couldNot = mutableListOf<String>()
        val options = CaptureRequestOptions.Builder()

        options.setCaptureRequestOption(
            CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(s.fps, s.fps)
        )
        applied["fps"] = s.fps

        if (caps.canDisableStabilization) {
            options.setCaptureRequestOption(
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF,
            )
            options.setCaptureRequestOption(
                CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF,
            )
            applied["stabilization"] = false
        } else {
            couldNot.add("stabilization")
        }

        options.setCaptureRequestOption(CaptureRequest.CONTROL_AWB_MODE, awbModeFor(s.colorTempK))
        applied["awbMode"] = awbModeFor(s.colorTempK)

        if (s.manual && caps.manualSensor) {
            options.setCaptureRequestOption(
                CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF
            )
            options.setCaptureRequestOption(CaptureRequest.SENSOR_EXPOSURE_TIME, s.shutterNs)
            options.setCaptureRequestOption(CaptureRequest.SENSOR_SENSITIVITY, s.iso)
            options.setCaptureRequestOption(
                CaptureRequest.SENSOR_FRAME_DURATION, 1_000_000_000L / s.fps
            )
            applied["manualExposure"] = true
            applied["shutterNs"] = s.shutterNs
            applied["iso"] = s.iso
        } else {
            options.setCaptureRequestOption(CaptureRequest.CONTROL_AE_LOCK, true)
            applied["manualExposure"] = false
            applied["aeLock"] = true
            if (s.manual) couldNot.add("manualShutterAndIso")
        }

        // Fixed focus near infinity so every phone holds the ring at the same plane.
        options.setCaptureRequestOption(
            CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF
        )
        options.setCaptureRequestOption(CaptureRequest.LENS_FOCUS_DISTANCE, 0f)
        applied["focusLocked"] = true

        control.setCaptureRequestOptions(options.build())
        applied["couldNotApply"] = couldNot
        return applied
    }

    private fun awbModeFor(kelvin: Int): Int = when {
        kelvin < 3500 -> CaptureRequest.CONTROL_AWB_MODE_INCANDESCENT
        kelvin < 4800 -> CaptureRequest.CONTROL_AWB_MODE_WARM_FLUORESCENT
        kelvin < 5600 -> CaptureRequest.CONTROL_AWB_MODE_DAYLIGHT
        kelvin < 6800 -> CaptureRequest.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT
        else -> CaptureRequest.CONTROL_AWB_MODE_SHADE
    }

    /** Begin recording at once; the beep fires at the shared start instant for the audio check. */
    @SuppressLint("MissingPermission")
    fun beginRecording(recordingId: String, round: String, letter: String, startAtMs: Long) {
        val videoCapture = this.videoCapture ?: return
        this.recordingId = recordingId
        this.round = round
        this.letter = letter
        this.startAtMs = startAtMs
        frameTimesMs.clear()
        recordedDurationMs = 0L

        val dir = File(context.getExternalFilesDir(null), "recordings").apply { mkdirs() }
        val safeRound = round.ifBlank { "r1" }.replace(Regex("[^A-Za-z0-9]"), "")
        val file = File(dir, "${safeRound}_$letter.mp4")
        outputFile = file
        val done = CompletableDeferred<File>()
        finalized = done

        val options = FileOutputOptions.Builder(file).build()
        recording = videoCapture.output
            .prepareRecording(context, options)
            .withAudioEnabled()
            .start(ContextCompat.getMainExecutor(context)) { event ->
                when (event) {
                    // Collect frame times only from the moment recording actually starts, so the
                    // encoder-startup window (which differs by phone model) stays out of the
                    // sidecar: frames[0] is the first frame in the file, not an earlier preview
                    // frame captured before the muxer began writing.
                    is VideoRecordEvent.Start -> collecting = true
                    is VideoRecordEvent.Finalize -> {
                        collecting = false
                        recordedDurationMs = event.recordingStats.recordedDurationNanos / 1_000_000L
                        done.complete(file)
                    }
                }
            }
        scheduleBeep(startAtMs)
    }

    private fun scheduleBeep(startAtMs: Long) {
        val delay = (startAtMs - SharedClock.nowShared()).coerceAtLeast(0L)
        mainHandler.postDelayed({
            runCatching {
                ToneGenerator(AudioManager.STREAM_MUSIC, 90).apply {
                    startTone(ToneGenerator.TONE_PROP_BEEP, 150)
                    mainHandler.postDelayed({ release() }, 300)
                }
            }
        }, delay)
    }

    /** Stop and finalise the recording, then build its sidecar from the collected frame times. */
    suspend fun stopRecording(): Pair<File, Sidecar>? {
        val rec = recording ?: return null
        rec.stop()
        recording = null
        val file = finalized?.let { withTimeoutOrNull(10_000) { it.await() } } ?: outputFile
        ?: return null
        val frames = frameTimesMs.sorted()
        val sidecar = Sidecar(
            recordingId = recordingId,
            round = round,
            letter = letter,
            device = "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}".trim(),
            settings = settings,
            appliedSettings = appliedSettings,
            startAtMs = startAtMs,
            clock = SharedClock.offset,
            frameTimesMs = frames,
            // The first time collected is the first frame in the file, since collection begins at
            // the Start event. The pipeline aligns the cameras on this rather than on a preview
            // frame.
            videoFirstFrameMs = frames.firstOrNull() ?: startAtMs,
            frameTimesSource = "image-analysis-from-record-start",
            recordedDurationMs = recordedDurationMs,
        )
        return file to sidecar
    }

    fun release() {
        collecting = false
        runCatching { recording?.stop() }
        recording = null
        runCatching { provider?.unbindAll() }
        camera = null
    }
}
