import java.io.File

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.google.gms.google-services")
}

// Firebase needs android/app/google-services.json. It is not in the repo (it belongs to a Firebase
// project Shem creates). Stop the build early with a clear message rather than the plugin's terse
// one, but only when a build is actually run, so tasks like `wrapper` still work without it.
val firebaseConfig = File(projectDir, "google-services.json")
val checkFirebaseConfig = tasks.register("checkFirebaseConfig") {
    doFirst {
        if (!firebaseConfig.exists()) {
            throw GradleException(
                "\n\nandroid/app/google-services.json is missing.\n" +
                    "Create a Firebase project, add an Android app with package com.varbox.camera, " +
                    "enable Firestore and Anonymous Auth, then download google-services.json into " +
                    "android/app/. See docs/phone_rig.md.\n"
            )
        }
    }
}
tasks.matching { it.name == "preBuild" }.configureEach { dependsOn(checkFirebaseConfig) }

android {
    namespace = "com.varbox.camera"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.varbox.camera"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
    }
    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.10.01")
    implementation(composeBom)

    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")

    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.navigation:navigation-compose:2.8.3")
    implementation("androidx.compose.runtime:runtime-livedata")

    // CameraX: preview, video capture, and an image-analysis stream for per-frame timestamps.
    val camerax = "1.4.0"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-video:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    // Firebase: anonymous auth and Firestore hold the rigs.
    implementation(platform("com.google.firebase:firebase-bom:33.5.1"))
    implementation("com.google.firebase:firebase-auth-ktx")
    implementation("com.google.firebase:firebase-firestore-ktx")

    // QR scanning (pairing) and generation (rig join code).
    implementation("com.google.mlkit:barcode-scanning:17.3.0")
    implementation("com.google.zxing:core:3.5.3")

    // Background uploads and stored settings.
    implementation("androidx.work:work-runtime-ktx:2.9.1")
    implementation("androidx.datastore:datastore-preferences:1.1.1")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-play-services:1.9.0")
    // CameraX and WorkManager return Guava ListenableFutures; make the type visible to Kotlin.
    implementation("com.google.guava:guava:33.3.1-android")

    debugImplementation("androidx.compose.ui:ui-tooling")
    implementation("androidx.compose.ui:ui-tooling-preview")
}
