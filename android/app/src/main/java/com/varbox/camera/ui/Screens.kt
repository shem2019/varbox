package com.varbox.camera.ui

import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.FilterChip
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.work.WorkInfo
import androidx.work.WorkManager
import com.varbox.camera.MainViewModel
import com.varbox.camera.Wire
import com.varbox.camera.pair.QrCodes
import com.varbox.camera.rig.RigSettings

@Composable
fun SiteScanScreen(vm: MainViewModel) {
    Column(Modifier.fillMaxSize()) {
        Text(
            "Sign this phone in to VAR Box",
            Modifier.padding(16.dp),
            style = androidx.compose.material3.MaterialTheme.typography.titleLarge,
        )
        Text(
            "On the dashboard, open Team → Add a phone and scan the QR it shows.",
            Modifier.padding(horizontal = 16.dp),
        )
        Box(Modifier.weight(1f).padding(16.dp)) {
            QrScanner(onCode = { vm.handleScan(it, onSite = {}, onRig = {}) })
        }
        StatusLine(vm)
    }
}

@Composable
fun PairScreen(vm: MainViewModel) {
    var mode by remember { mutableStateOf("choose") }
    var rigName by remember { mutableStateOf("Ringside") }
    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Set up the camera rig", style = androidx.compose.material3.MaterialTheme.typography.titleLarge)
        when (mode) {
            "choose" -> {
                Text("One phone creates the rig and shows a QR; the others scan it to join.")
                Button(onClick = { mode = "create" }) { Text("Create a rig (this is the director)") }
                OutlinedButton(onClick = { mode = "join" }) { Text("Join a rig by scanning") }
            }
            "create" -> {
                OutlinedTextField(rigName, { rigName = it }, label = { Text("Rig name") })
                Button(onClick = { vm.createRig(rigName) }) { Text("Create") }
                OutlinedButton(onClick = { mode = "choose" }) { Text("Back") }
            }
            "join" -> {
                Text("Point at the director's join QR.")
                Box(Modifier.weight(1f)) {
                    QrScanner(onCode = { vm.handleScan(it, onSite = {}, onRig = {}) })
                }
                OutlinedButton(onClick = { mode = "choose" }) { Text("Back") }
            }
        }
        StatusLine(vm)
    }
}

@Composable
fun RecordScreen(vm: MainViewModel) {
    val context = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val rig by vm.rig.collectAsStateWithLifecycle()
    val recording by vm.recording.collectAsStateWithLifecycle()
    val applied by vm.applied.collectAsStateWithLifecycle()
    val previewView = remember { PreviewView(context) }
    val settings = rig?.settings ?: RigSettings()

    androidx.compose.runtime.LaunchedEffect(settings) {
        vm.bindCamera(previewView, owner, settings)
    }

    Column(Modifier.fillMaxSize()) {
        Box(Modifier.weight(1f)) {
            AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
            if (recording) {
                Text(
                    "● REC ${vm.myLetter()}",
                    Modifier.align(Alignment.TopStart).padding(16.dp),
                    color = androidx.compose.material3.MaterialTheme.colorScheme.primary,
                )
            }
        }
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Camera ${vm.myLetter()} · ${settings.fps} fps · ${appliedSummary(applied)}")
            if (vm.isDirector()) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = { vm.startTake() }, enabled = !recording) { Text("Start take") }
                    OutlinedButton(onClick = { vm.stopTake() }, enabled = recording) { Text("Stop") }
                }
                Text("The take starts three seconds after you press, on every phone at once.")
            } else {
                Text("The director starts and stops the take. Keep this screen open.")
            }
            StatusLine(vm)
        }
    }
}

@Composable
fun RigScreen(vm: MainViewModel) {
    val rig by vm.rig.collectAsStateWithLifecycle()
    val members by vm.members.collectAsStateWithLifecycle()
    val clock by vm.clock.collectAsStateWithLifecycle()
    val wifiOnly by vm.wifiOnly.collectAsStateWithLifecycle()
    var fps by remember { mutableStateOf(30) }
    val scroll = rememberScrollState()

    Column(Modifier.fillMaxSize().verticalScroll(scroll).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(rig?.name ?: "Rig", style = androidx.compose.material3.MaterialTheme.typography.titleLarge)

        if (vm.isDirector() && rig != null) {
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Add a phone", style = androidx.compose.material3.MaterialTheme.typography.titleMedium)
                    val bitmap = remember(rig!!.id) {
                        QrCodes.generate(Wire.rigJoinPayload(rig!!.id, rig!!.name))
                    }
                    Image(bitmap.asImageBitmap(), null, Modifier.size(220.dp))
                    Text("Scan this from VAR Box Camera on the other phones.")
                }
            }
        }

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("Phones", style = androidx.compose.material3.MaterialTheme.typography.titleMedium)
                members.forEach { m ->
                    val applied = (m.applied["couldNotApply"] as? List<*>)?.joinToString() ?: ""
                    Text("Camera ${m.letter} · ${m.model.ifBlank { m.name }}" +
                        (if (m.clock.uncertaintyMs > 0) " · clock ±%.0f ms".format(m.clock.uncertaintyMs) else " · no clock yet") +
                        (if (applied.isNotEmpty()) " · could not set: $applied" else ""))
                }
            }
        }

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Clock", style = androidx.compose.material3.MaterialTheme.typography.titleMedium)
                Text(clock?.let { "This phone: ±%.0f ms".format(it.uncertaintyMs) } ?: "Not measured yet")
                OutlinedButton(onClick = { vm.measureClock() }) { Text("Measure clock") }
            }
        }

        if (vm.isDirector()) {
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Settings", style = androidx.compose.material3.MaterialTheme.typography.titleMedium)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        listOf(30, 60).forEach { option ->
                            FilterChip(fps == option, { fps = option }, { Text("$option fps") })
                        }
                    }
                    Button(onClick = { vm.pushCommonSettings(fps) }) {
                        Text("Apply to every phone")
                    }
                    Text("Only the settings every phone supports are offered.")
                }
            }
        }

        Card(Modifier.fillMaxWidth()) {
            Row(
                Modifier.padding(16.dp).fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("Upload on Wi-Fi only")
                Switch(wifiOnly, { vm.setWifiOnly(it) })
            }
        }
        StatusLine(vm)
    }
}

@Composable
fun UploadsScreen(vm: MainViewModel) {
    val context = LocalContext.current
    val infos by WorkManager.getInstance(context)
        .getWorkInfosByTagLiveData("upload")
        .observeAsState(emptyList())
    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Uploads", style = androidx.compose.material3.MaterialTheme.typography.titleLarge)
        if (infos.isEmpty()) {
            Text("Recordings upload here after each take.")
        }
        infos.sortedByDescending { it.id.toString() }.forEach { info ->
            val pct = info.progress.getInt(com.varbox.camera.upload.UploadWorker.KEY_PROGRESS, -1)
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp)) {
                    Text(stateLabel(info))
                    if (info.state == WorkInfo.State.RUNNING && pct >= 0) {
                        Text("$pct%")
                    }
                }
            }
        }
        StatusLine(vm)
    }
}

@Composable
private fun StatusLine(vm: MainViewModel) {
    val status by vm.status.collectAsStateWithLifecycle()
    if (status.isNotEmpty()) {
        Text(status, Modifier.padding(top = 8.dp), fontFamily = FontFamily.Monospace)
    }
    Spacer(Modifier.height(4.dp))
}

private fun appliedSummary(applied: Map<String, Any?>): String {
    if (applied.isEmpty()) return "settings pending"
    val manual = applied["manualExposure"] == true
    val stab = applied["stabilization"] == false
    return buildString {
        append(if (manual) "manual exposure" else "locked auto-exposure")
        if (stab) append(", stabilisation off")
    }
}

private fun stateLabel(info: WorkInfo): String = when (info.state) {
    WorkInfo.State.SUCCEEDED -> "Uploaded"
    WorkInfo.State.RUNNING -> "Uploading"
    WorkInfo.State.ENQUEUED -> "Waiting for the network"
    WorkInfo.State.FAILED -> "Upload failed"
    WorkInfo.State.BLOCKED -> "Queued"
    WorkInfo.State.CANCELLED -> "Cancelled"
}
