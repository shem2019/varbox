package com.varbox.camera.capture

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.os.Build
import android.util.Size
import com.varbox.camera.rig.RigSettings

/** What one phone's back camera can do, so the director only offers settings every phone supports. */
data class Capabilities(
    val model: String,
    val fpsOptions: List<Int>, // fixed frame rates the camera can lock (e.g. 30, 60)
    val has1080p: Boolean,
    val manualSensor: Boolean,
    val isoMin: Int,
    val isoMax: Int,
    val exposureMinNs: Long,
    val exposureMaxNs: Long,
    val canDisableStabilization: Boolean,
) {
    fun toMap(): Map<String, Any?> = mapOf(
        "model" to model,
        "fpsOptions" to fpsOptions,
        "has1080p" to has1080p,
        "manualSensor" to manualSensor,
        "isoMin" to isoMin,
        "isoMax" to isoMax,
        "exposureMinNs" to exposureMinNs,
        "exposureMaxNs" to exposureMaxNs,
        "canDisableStabilization" to canDisableStabilization,
    )

    companion object {
        fun fromMap(m: Map<String, Any?>): Capabilities {
            @Suppress("UNCHECKED_CAST")
            val fps = (m["fpsOptions"] as? List<Number>)?.map { it.toInt() } ?: emptyList()
            fun i(k: String) = (m[k] as? Number)?.toInt() ?: 0
            fun l(k: String) = (m[k] as? Number)?.toLong() ?: 0L
            fun b(k: String) = m[k] as? Boolean ?: false
            return Capabilities(
                model = m["model"] as? String ?: "phone",
                fpsOptions = fps,
                has1080p = b("has1080p"),
                manualSensor = b("manualSensor"),
                isoMin = i("isoMin"),
                isoMax = i("isoMax"),
                exposureMinNs = l("exposureMinNs"),
                exposureMaxNs = l("exposureMaxNs"),
                canDisableStabilization = b("canDisableStabilization"),
            )
        }
    }
}

object CameraCapabilities {
    fun read(context: Context): Capabilities {
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val cameraId = backCameraId(manager)
        val chars = manager.getCameraCharacteristics(cameraId)

        val fpsRanges = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
        val fixed = (fpsRanges ?: emptyArray())
            .filter { it.lower == it.upper && it.lower in intArrayOf(30, 60) }
            .map { it.lower }
            .distinct()
            .sorted()

        val streamMap = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val sizes = streamMap?.getOutputSizes(ImageFormat.YUV_420_888) ?: emptyArray()
        val has1080p = sizes.any { it == Size(1920, 1080) }

        val caps = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: IntArray(0)
        val manual = caps.contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR)
        val iso = chars.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        val exposure = chars.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)

        val stabModes = chars.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES)
            ?: IntArray(0)
        val canDisableStab = stabModes.contains(
            CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF
        )

        return Capabilities(
            model = "${Build.MANUFACTURER} ${Build.MODEL}".trim(),
            fpsOptions = fixed.ifEmpty { listOf(30) },
            has1080p = has1080p,
            manualSensor = manual,
            isoMin = iso?.lower ?: 100,
            isoMax = iso?.upper ?: 800,
            exposureMinNs = exposure?.lower ?: 125_000L,
            exposureMaxNs = exposure?.upper ?: 100_000_000L,
            canDisableStabilization = canDisableStab,
        )
    }

    private fun backCameraId(manager: CameraManager): String {
        for (id in manager.cameraIdList) {
            val facing = manager.getCameraCharacteristics(id)
                .get(CameraCharacteristics.LENS_FACING)
            if (facing == CameraCharacteristics.LENS_FACING_BACK) return id
        }
        return manager.cameraIdList.firstOrNull() ?: "0"
    }
}

/** Reduce every member's capabilities to the settings they all support, matched to the job fps. */
object SettingsPlanner {
    fun common(all: List<Capabilities>, preferredFps: Int): RigSettings {
        if (all.isEmpty()) return RigSettings(fps = preferredFps)
        val fpsEveryone = all.map { it.fpsOptions.toSet() }
            .reduce { a, b -> a intersect b }
        val fps = when {
            preferredFps in fpsEveryone -> preferredFps
            30 in fpsEveryone -> 30
            else -> fpsEveryone.minOrNull() ?: 30
        }
        val manual = all.all { it.manualSensor }
        // An ISO and a 1/500 s shutter every sensor can reach.
        val isoFloor = all.maxOf { it.isoMin }
        val isoCeil = all.minOf { it.isoMax }.coerceAtLeast(isoFloor)
        val iso = 400.coerceIn(isoFloor, isoCeil)
        val shutterFloor = all.maxOf { it.exposureMinNs }
        val shutterCeil = all.minOf { it.exposureMaxNs }.coerceAtLeast(shutterFloor)
        val shutter = 2_000_000L.coerceIn(shutterFloor, shutterCeil)
        return RigSettings(
            fps = fps,
            width = 1920,
            height = 1080,
            stabilization = false,
            manual = manual,
            shutterNs = shutter,
            iso = iso,
            colorTempK = 5200,
        )
    }
}
