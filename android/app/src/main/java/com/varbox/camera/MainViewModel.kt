package com.varbox.camera

import android.app.Application
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewModelScope
import androidx.work.workDataOf
import com.varbox.camera.capture.CameraCapabilities
import com.varbox.camera.capture.CameraController
import com.varbox.camera.capture.Capabilities
import com.varbox.camera.capture.SettingsPlanner
import com.varbox.camera.capture.Sidecar
import com.varbox.camera.clock.ClockOffset
import com.varbox.camera.clock.SharedClock
import com.varbox.camera.rig.Member
import com.varbox.camera.rig.MemberClock
import com.varbox.camera.rig.Rig
import com.varbox.camera.rig.RigCommand
import com.varbox.camera.rig.RigRepository
import com.varbox.camera.rig.RigSettings
import com.varbox.camera.site.SiteStore
import com.varbox.camera.upload.UploadWorker
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.io.File

/** Holds the app's state and wires the rig, the clock, the camera and the uploads together. */
@OptIn(ExperimentalCoroutinesApi::class)
class MainViewModel(app: Application) : AndroidViewModel(app) {
    private val store = SiteStore(app)
    private val rigs = RigRepository()
    private val controller = CameraController(app)

    val site: StateFlow<SiteStore.Site?> =
        store.site.stateIn(viewModelScope, SharingStarted.Eagerly, null)

    val rigId: StateFlow<String?> =
        store.rig.stateIn(viewModelScope, SharingStarted.Eagerly, null)

    val rig: StateFlow<Rig?> = rigId
        .flatMapLatest { id -> if (id == null) flowOf(null) else rigs.observeRig(id) }
        .stateIn(viewModelScope, SharingStarted.Eagerly, null)

    val members: StateFlow<List<Member>> = rigId
        .flatMapLatest { id -> if (id == null) flowOf(emptyList()) else rigs.observeMembers(id) }
        .stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())

    private val _uid = MutableStateFlow<String?>(null)
    val uid: StateFlow<String?> = _uid.asStateFlow()

    private val _clock = MutableStateFlow<ClockOffset?>(null)
    val clock: StateFlow<ClockOffset?> = _clock.asStateFlow()

    private val _applied = MutableStateFlow<Map<String, Any?>>(emptyMap())
    val applied: StateFlow<Map<String, Any?>> = _applied.asStateFlow()

    private val _status = MutableStateFlow("")
    val status: StateFlow<String> = _status.asStateFlow()

    private val _recording = MutableStateFlow(false)
    val recording: StateFlow<Boolean> = _recording.asStateFlow()

    val wifiOnly: StateFlow<Boolean> =
        store.wifiOnlyFlow.stateIn(viewModelScope, SharingStarted.Eagerly, false)

    private var lastSeq = -1L
    private var roundNumber = 1
    private var cameraBound = false

    init {
        viewModelScope.launch {
            runCatching { _uid.value = rigs.signIn() }
                .onFailure { _status.value = "Sign-in failed: ${it.message}" }
        }
        observeCommands()
    }

    fun isDirector(): Boolean = rig.value?.directorId != null && rig.value?.directorId == uid.value

    fun myLetter(): String = members.value.firstOrNull { it.uid == uid.value }?.letter ?: "A"

    // ----------------------------------------------------------------- pairing
    fun handleScan(text: String, onSite: () -> Unit, onRig: () -> Unit) {
        Wire.parseSitePairing(text)?.let { pairing ->
            viewModelScope.launch {
                store.saveSite(pairing)
                _status.value = "Signed in to ${pairing.url}"
                onSite()
            }
            return
        }
        Wire.parseRigJoin(text)?.let { id ->
            viewModelScope.launch { join(id); onRig() }
            return
        }
        _status.value = "That QR is not a VAR Box code."
    }

    fun createRig(name: String) = viewModelScope.launch {
        val id = uid.value ?: rigs.signIn().also { _uid.value = it }
        val rigId = rigs.createRig(name, id, deviceName(), model())
        store.saveRig(rigId)
        measureClock()
    }

    private suspend fun join(rigId: String) {
        val id = uid.value ?: rigs.signIn().also { _uid.value = it }
        runCatching { rigs.joinRig(rigId, id, deviceName(), model()) }
            .onSuccess {
                store.saveRig(rigId)
                measureClock()
            }
            .onFailure { _status.value = "Could not join: ${it.message}" }
    }

    fun setWifiOnly(value: Boolean) = viewModelScope.launch { store.setWifiOnly(value) }

    // ----------------------------------------------------------------- clock
    fun measureClock() = viewModelScope.launch {
        _status.value = "Measuring the shared clock…"
        val offset = SharedClock.measure()
        _clock.value = offset
        val id = uid.value
        val rid = rigId.value
        if (offset != null && id != null && rid != null) {
            rigs.publishClock(rid, id, MemberClock(offset.offsetMs, offset.uncertaintyMs, System.currentTimeMillis()))
            _status.value = "Clock set (±%.0f ms)".format(offset.uncertaintyMs)
        } else if (offset == null) {
            _status.value = "Could not reach the time server; try again on a better network."
        }
    }

    // ----------------------------------------------------------------- settings
    /** Director only: offer the settings every member supports and push them to the rig. */
    fun pushCommonSettings(preferredFps: Int) = viewModelScope.launch {
        val rid = rigId.value ?: return@launch
        val caps = members.value.mapNotNull { m ->
            if (m.capabilities.isEmpty()) null else Capabilities.fromMap(m.capabilities)
        }
        val settings = SettingsPlanner.common(caps, preferredFps)
        rigs.setSettings(rid, settings)
        _status.value = "Settings pushed to ${members.value.size} phone(s)"
    }

    // ----------------------------------------------------------------- camera
    fun bindCamera(previewView: PreviewView, owner: LifecycleOwner, settings: RigSettings) {
        viewModelScope.launch {
            val caps = runCatching { CameraCapabilities.read(getApplication()) }.getOrNull()
            if (caps != null) {
                val applied = runCatching { controller.bind(owner, previewView, settings, caps) }
                    .getOrElse { mapOf("error" to (it.message ?: "bind failed")) }
                _applied.value = applied
                cameraBound = true
                val id = uid.value
                val rid = rigId.value
                if (id != null && rid != null) {
                    rigs.publishCapabilities(rid, id, caps.toMap(), caps.model)
                    rigs.publishApplied(rid, id, applied)
                }
            } else {
                _status.value = "This device's camera could not be read."
            }
        }
    }

    // ----------------------------------------------------------------- recording
    /** Director: schedule the take 3 s ahead so every phone starts on the same shared instant. */
    fun startTake() = viewModelScope.launch {
        val rid = rigId.value ?: return@launch
        if (_clock.value == null) measureClock().join()
        val round = "r$roundNumber"
        val recordingId = "$rid-$round-${System.currentTimeMillis()}"
        val startAt = SharedClock.nowShared() + 3000
        rigs.sendCommand(
            rid, RigCommand("record", startAt, recordingId, round, System.currentTimeMillis())
        )
    }

    fun stopTake() = viewModelScope.launch {
        val rid = rigId.value ?: return@launch
        rigs.sendCommand(rid, RigCommand("stop", 0, "", "r$roundNumber", System.currentTimeMillis()))
    }

    private fun observeCommands() {
        viewModelScope.launch {
            rig.collect { r ->
                val cmd = r?.command ?: return@collect
                if (cmd.seq <= lastSeq) return@collect
                lastSeq = cmd.seq
                when (cmd.action) {
                    "record" -> onRecordCommand(cmd)
                    "stop" -> onStopCommand()
                }
            }
        }
    }

    private fun onRecordCommand(cmd: RigCommand) {
        if (!cameraBound) {
            _status.value = "Open the Record screen before the take starts."
            return
        }
        controller.beginRecording(cmd.recordingId, cmd.round, myLetter(), cmd.startAtMs)
        _recording.value = true
        _status.value = "Recording ${cmd.round} (camera ${myLetter()})"
    }

    private fun onStopCommand() {
        if (!_recording.value) return
        viewModelScope.launch {
            val result = runCatching { controller.stopRecording() }.getOrNull()
            _recording.value = false
            if (result != null) {
                enqueueUpload(result.first, result.second)
                roundNumber += 1
                _status.value = "Saved ${result.first.name}; uploading…"
            }
        }
    }

    private fun enqueueUpload(video: File, sidecar: Sidecar) {
        val sidecarFile = File(video.parentFile, video.nameWithoutExtension + ".json")
        sidecar.writeTo(sidecarFile)
        val r = rig.value
        val data = workDataOf(
            UploadWorker.KEY_VIDEO to video.absolutePath,
            UploadWorker.KEY_SIDECAR to sidecarFile.absolutePath,
            UploadWorker.KEY_REC_UID to sidecar.recordingId,
            UploadWorker.KEY_LETTER to sidecar.letter,
            UploadWorker.KEY_FPS to sidecar.settings.fps,
            UploadWorker.KEY_EXPECTED to (r?.memberIds?.size ?: members.value.size).coerceAtLeast(1),
            UploadWorker.KEY_ROUND to sidecar.round,
            UploadWorker.KEY_TITLE to ((r?.name ?: "Recording") + ", " + sidecar.round),
            UploadWorker.KEY_RIG to (r?.id ?: ""),
            UploadWorker.KEY_DEVICE to sidecar.device,
        )
        UploadWorker.enqueue(
            getApplication(), data, wifiOnly.value, "upload-${sidecar.recordingId}-${sidecar.letter}"
        )
    }

    private fun deviceName(): String = model()
    private fun model(): String = "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}".trim()

    override fun onCleared() {
        controller.release()
        super.onCleared()
    }
}
