package com.varbox.camera

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.List
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.varbox.camera.ui.PairScreen
import com.varbox.camera.ui.RecordScreen
import com.varbox.camera.ui.RigScreen
import com.varbox.camera.ui.SiteScanScreen
import com.varbox.camera.ui.UploadsScreen
import com.varbox.camera.ui.VarBoxTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            VarBoxTheme {
                Surface(Modifier.fillMaxSize()) {
                    Root()
                }
            }
        }
    }
}

private val PERMISSIONS = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)

@Composable
private fun Root() {
    val vm: MainViewModel = viewModel()
    val context = androidx.compose.ui.platform.LocalContext.current
    var granted by remember {
        mutableStateOf(PERMISSIONS.all {
            context.checkSelfPermission(it) == PackageManager.PERMISSION_GRANTED
        })
    }
    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { result -> granted = result.values.all { it } }

    androidx.compose.runtime.LaunchedEffect(Unit) {
        if (!granted) launcher.launch(PERMISSIONS)
    }

    val site by vm.site.collectAsStateWithLifecycle()
    val rigId by vm.rigId.collectAsStateWithLifecycle()

    when {
        !granted -> PermissionPrompt { launcher.launch(PERMISSIONS) }
        site == null -> SiteScanScreen(vm)
        rigId == null -> PairScreen(vm)
        else -> MainScaffold(vm)
    }
}

@Composable
private fun PermissionPrompt(onAsk: () -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Center,
    ) {
        Text("VAR Box Camera needs the camera and microphone to record a rig.")
        Button(onClick = onAsk, modifier = Modifier.padding(top = 12.dp)) { Text("Grant access") }
    }
}

@Composable
private fun MainScaffold(vm: MainViewModel) {
    var tab by remember { mutableIntStateOf(0) }
    val tabs = listOf("Record", "Rig", "Uploads")
    Scaffold(
        bottomBar = {
            NavigationBar {
                tabs.forEachIndexed { index, label ->
                    NavigationBarItem(
                        selected = tab == index,
                        onClick = { tab = index },
                        icon = {
                            Icon(
                                when (index) {
                                    0 -> Icons.Filled.PlayArrow
                                    1 -> Icons.Filled.Settings
                                    else -> Icons.Filled.List
                                },
                                contentDescription = label,
                            )
                        },
                        label = { Text(label) },
                    )
                }
            }
        },
    ) { padding ->
        Column(Modifier.fillMaxSize().padding(padding)) {
            when (tab) {
                0 -> RecordScreen(vm)
                1 -> RigScreen(vm)
                else -> UploadsScreen(vm)
            }
        }
    }
}
