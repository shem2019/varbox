package com.varbox.camera.upload

import android.content.Context
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.varbox.camera.site.SiteStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** Uploads one camera in the background, resuming across retries. Progress is reported as `pct`. */
class UploadWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val site = SiteStore(applicationContext).current()
            ?: return@withContext Result.failure()
        val data = inputData
        val job = UploadJob(
            siteUrl = site.url,
            token = site.token,
            videoPath = data.getString(KEY_VIDEO) ?: return@withContext Result.failure(),
            sidecarPath = data.getString(KEY_SIDECAR) ?: "",
            recUid = data.getString(KEY_REC_UID) ?: return@withContext Result.failure(),
            letter = data.getString(KEY_LETTER) ?: "A",
            fps = data.getInt(KEY_FPS, 30),
            expectedCameras = data.getInt(KEY_EXPECTED, 2),
            round = data.getString(KEY_ROUND) ?: "r1",
            title = data.getString(KEY_TITLE) ?: "Recording",
            rig = data.getString(KEY_RIG) ?: "",
            device = data.getString(KEY_DEVICE) ?: "phone",
        )
        try {
            UploadClient().run(job) { pct -> setProgressAsync(workDataOf(KEY_PROGRESS to pct)) }
            Result.success()
        } catch (e: Exception) {
            if (runAttemptCount < 8) Result.retry() else Result.failure()
        }
    }

    companion object {
        const val KEY_VIDEO = "video"
        const val KEY_SIDECAR = "sidecar"
        const val KEY_REC_UID = "rec_uid"
        const val KEY_LETTER = "letter"
        const val KEY_FPS = "fps"
        const val KEY_EXPECTED = "expected"
        const val KEY_ROUND = "round"
        const val KEY_TITLE = "title"
        const val KEY_RIG = "rig"
        const val KEY_DEVICE = "device"
        const val KEY_PROGRESS = "pct"
        const val TAG = "upload"

        fun enqueue(context: Context, data: Data, wifiOnly: Boolean, uniqueName: String) {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(if (wifiOnly) NetworkType.UNMETERED else NetworkType.CONNECTED)
                .build()
            val request = OneTimeWorkRequestBuilder<UploadWorker>()
                .setInputData(data)
                .setConstraints(constraints)
                .addTag(TAG)
                .build()
            WorkManager.getInstance(context)
                .enqueueUniqueWork(uniqueName, ExistingWorkPolicy.KEEP, request)
        }
    }
}
