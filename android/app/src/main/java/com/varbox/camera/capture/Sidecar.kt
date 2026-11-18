package com.varbox.camera.capture

import com.varbox.camera.clock.ClockOffset
import com.varbox.camera.rig.RigSettings
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * The JSON written beside each phone video. It records the device, the settings actually applied
 * (and any that could not be), the clock offset to the shared reference, and every frame's capture
 * time on that shared clock. The GPU lines the cameras up from `frames`.
 */
data class Sidecar(
    val recordingId: String,
    val round: String,
    val letter: String,
    val device: String,
    val settings: RigSettings,
    val appliedSettings: Map<String, Any?>,
    val startAtMs: Long,
    val clock: ClockOffset?,
    val frameTimesMs: List<Long>,
) {
    fun toJson(): String {
        val o = JSONObject()
        o.put("recording_id", recordingId)
        o.put("round", round)
        o.put("camera", letter)
        o.put("device", device)
        o.put("fps", settings.fps)
        o.put("width", settings.width)
        o.put("height", settings.height)
        o.put("start_at_ms", startAtMs)
        o.put("clock_offset_ms", clock?.offsetMs ?: 0.0)
        o.put("clock_uncertainty_ms", clock?.uncertaintyMs ?: 0.0)
        o.put("settings", JSONObject(settings.toMap()))
        o.put("applied", JSONObject(appliedSettings))
        o.put("frames", JSONArray(frameTimesMs))
        return o.toString()
    }

    fun writeTo(file: File) = file.writeText(toJson())
}
