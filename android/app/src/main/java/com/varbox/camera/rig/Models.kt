package com.varbox.camera.rig

/** The capture settings every phone in a rig applies. 1/500 s shutter freezes punches. */
data class RigSettings(
    val fps: Int = 30,
    val width: Int = 1920,
    val height: Int = 1080,
    val stabilization: Boolean = false,
    val manual: Boolean = true,
    val shutterNs: Long = 2_000_000L, // 1/500 s
    val iso: Int = 400,
    val colorTempK: Int = 5200,
) {
    fun toMap(): Map<String, Any?> = mapOf(
        "fps" to fps,
        "width" to width,
        "height" to height,
        "stabilization" to stabilization,
        "manual" to manual,
        "shutterNs" to shutterNs,
        "iso" to iso,
        "colorTempK" to colorTempK,
    )

    companion object {
        fun fromMap(m: Map<String, Any?>?): RigSettings {
            if (m == null) return RigSettings()
            fun i(k: String, d: Int) = (m[k] as? Number)?.toInt() ?: d
            fun l(k: String, d: Long) = (m[k] as? Number)?.toLong() ?: d
            fun b(k: String, d: Boolean) = m[k] as? Boolean ?: d
            return RigSettings(
                fps = i("fps", 30),
                width = i("width", 1920),
                height = i("height", 1080),
                stabilization = b("stabilization", false),
                manual = b("manual", true),
                shutterNs = l("shutterNs", 2_000_000L),
                iso = i("iso", 400),
                colorTempK = i("colorTempK", 5200),
            )
        }
    }
}

/** The director's record/stop command. A phone acts when the sequence number grows. */
data class RigCommand(
    val action: String = "idle", // idle | record | stop
    val startAtMs: Long = 0L,
    val recordingId: String = "",
    val round: String = "",
    val seq: Long = 0L,
) {
    fun toMap(): Map<String, Any?> = mapOf(
        "action" to action,
        "startAtMs" to startAtMs,
        "recordingId" to recordingId,
        "round" to round,
        "seq" to seq,
    )

    companion object {
        fun fromMap(m: Map<String, Any?>?): RigCommand {
            if (m == null) return RigCommand()
            return RigCommand(
                action = m["action"] as? String ?: "idle",
                startAtMs = (m["startAtMs"] as? Number)?.toLong() ?: 0L,
                recordingId = m["recordingId"] as? String ?: "",
                round = m["round"] as? String ?: "",
                seq = (m["seq"] as? Number)?.toLong() ?: 0L,
            )
        }
    }
}

data class Rig(
    val id: String = "",
    val name: String = "",
    val directorId: String = "",
    val memberIds: List<String> = emptyList(),
    val settings: RigSettings = RigSettings(),
    val command: RigCommand = RigCommand(),
)

/** A phone's clock offset to the shared reference (shared = device epoch + offsetMs). */
data class MemberClock(
    val offsetMs: Double = 0.0,
    val uncertaintyMs: Double = 0.0,
    val atMs: Long = 0L,
) {
    fun toMap(): Map<String, Any?> =
        mapOf("offsetMs" to offsetMs, "uncertaintyMs" to uncertaintyMs, "atMs" to atMs)

    companion object {
        fun fromMap(m: Map<String, Any?>?): MemberClock {
            if (m == null) return MemberClock()
            fun d(k: String) = (m[k] as? Number)?.toDouble() ?: 0.0
            return MemberClock(d("offsetMs"), d("uncertaintyMs"), (m["atMs"] as? Number)?.toLong() ?: 0L)
        }
    }
}

data class Member(
    val uid: String = "",
    val name: String = "",
    val letter: String = "A",
    val model: String = "",
    val capabilities: Map<String, Any?> = emptyMap(),
    val applied: Map<String, Any?> = emptyMap(),
    val clock: MemberClock = MemberClock(),
)
