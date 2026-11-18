package com.varbox.camera.clock

import android.os.SystemClock
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** The device's offset to the shared reference clock, with how uncertain it is (half the round trip). */
data class ClockOffset(val offsetMs: Double, val uncertaintyMs: Double)

/**
 * Keeps the device aligned to a shared reference (SNTP against time.google.com) and converts the
 * various Android clocks to that shared time, so every phone's frame timestamps live on one axis.
 */
object SharedClock {
    @Volatile
    var offset: ClockOffset? = null
        private set

    /** Take several samples and keep the one with the lowest round trip. */
    suspend fun measure(host: String = "time.google.com", samples: Int = 6): ClockOffset? =
        withContext(Dispatchers.IO) {
            var best: ClockSample? = null
            repeat(samples) {
                val sample = runCatching { SntpClient.query(host) }.getOrNull()
                if (sample != null && (best == null || sample.roundTripMs < best!!.roundTripMs)) {
                    best = sample
                }
            }
            best?.let {
                val o = ClockOffset(it.offsetMs, it.roundTripMs / 2.0)
                offset = o
                o
            }
        }

    private fun offsetMs(): Double = offset?.offsetMs ?: 0.0

    fun sharedFromEpoch(epochMs: Long): Long = (epochMs + offsetMs()).toLong()

    fun nowShared(): Long = sharedFromEpoch(System.currentTimeMillis())

    /**
     * Shared time for a Camera2 SENSOR_TIMESTAMP (nanoseconds). The timestamp's base is the
     * boot clock when the source is REALTIME, otherwise the uptime clock.
     */
    fun sharedFromSensorNs(timestampNs: Long, realtime: Boolean): Long {
        val tsMs = timestampNs / 1_000_000L
        val base = if (realtime) {
            System.currentTimeMillis() - SystemClock.elapsedRealtime()
        } else {
            System.currentTimeMillis() - SystemClock.uptimeMillis()
        }
        return sharedFromEpoch(tsMs + base)
    }
}
