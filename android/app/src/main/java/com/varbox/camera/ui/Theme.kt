package com.varbox.camera.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val Accent = Color(0xFFE23A3A)

private val DarkColors = darkColorScheme(
    primary = Accent,
    background = Color(0xFF0B0B0D),
    surface = Color(0xFF15161B),
)

private val LightColors = lightColorScheme(primary = Accent)

@Composable
fun VarBoxTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = if (isSystemInDarkTheme()) DarkColors else LightColors,
        content = content,
    )
}
