package com.varbox.camera.upload

import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/** What one camera uploads: the video, its sidecar, and the recording it belongs to. */
data class UploadJob(
    val siteUrl: String,
    val token: String,
    val videoPath: String,
    val sidecarPath: String,
    val recUid: String,
    val letter: String,
    val fps: Int,
    val expectedCameras: Int,
    val round: String,
    val title: String,
    val rig: String,
    val device: String,
)

/**
 * Uploads one camera to the site's token-authenticated phone API. Chunks are resumable: progress is
 * kept in a file beside the video so a retry carries on from where it stopped, and the server's 409
 * reply corrects the offset if they disagree.
 */
class UploadClient {
    companion object {
        const val CHUNK = 8 * 1024 * 1024
    }

    fun run(job: UploadJob, onProgress: (Int) -> Unit) {
        val video = File(job.videoPath)
        val size = video.length()
        val progressFile = File(job.videoPath + ".progress")

        var uploadId = 0L
        var offset = 0L
        if (progressFile.exists()) {
            runCatching {
                val p = JSONObject(progressFile.readText())
                uploadId = p.optLong("uploadId")
                offset = p.optLong("offset")
            }
        }
        if (uploadId == 0L) {
            uploadId = createUpload(job, video.name, size)
            offset = 0
            saveProgress(progressFile, uploadId, offset)
        }

        video.inputStream().use { input ->
            if (offset > 0) input.skip(offset)
            val buffer = ByteArray(CHUNK)
            while (offset < size) {
                val toRead = minOf(CHUNK.toLong(), size - offset).toInt()
                var read = 0
                while (read < toRead) {
                    val n = input.read(buffer, read, toRead - read)
                    if (n < 0) break
                    read += n
                }
                val received = putChunk(job, uploadId, offset, buffer, read)
                if (received != offset + read) {
                    // The server resumed elsewhere; follow it (re-open the stream at its offset).
                    offset = received
                    input.close()
                    return run(job, onProgress)
                }
                offset = received
                saveProgress(progressFile, uploadId, offset)
                onProgress(((offset * 100) / maxOf(1, size)).toInt())
            }
        }
        putSidecar(job, uploadId, File(job.sidecarPath).readText())
        progressFile.delete()
        onProgress(100)
    }

    private fun createUpload(job: UploadJob, name: String, size: Long): Long {
        val body = JSONObject()
            .put("rec_uid", job.recUid)
            .put("letter", job.letter)
            .put("name", name)
            .put("size", size)
            .put("fps", job.fps)
            .put("expected_cameras", job.expectedCameras)
            .put("round", job.round)
            .put("title", job.title)
            .put("rig", job.rig)
            .put("device", job.device)
            .toString()
        val conn = open(job, "/api/phone/uploads", "POST")
        conn.setRequestProperty("Content-Type", "application/json")
        conn.doOutput = true
        conn.outputStream.use { it.write(body.toByteArray()) }
        val json = readJson(conn)
        return json.getJSONObject("upload").getLong("id")
    }

    private fun putChunk(job: UploadJob, uploadId: Long, offset: Long, data: ByteArray, len: Int): Long {
        val conn = open(job, "/api/phone/uploads/$uploadId?offset=$offset", "PUT")
        conn.setRequestProperty("Content-Type", "application/octet-stream")
        conn.doOutput = true
        conn.setFixedLengthStreamingMode(len)
        conn.outputStream.use { it.write(data, 0, len) }
        val json = readJson(conn, allow = intArrayOf(200, 409))
        return json.getLong("received")
    }

    private fun putSidecar(job: UploadJob, uploadId: Long, sidecarJson: String) {
        val conn = open(job, "/api/phone/uploads/$uploadId/sidecar", "PUT")
        conn.setRequestProperty("Content-Type", "application/json")
        conn.doOutput = true
        conn.outputStream.use { it.write(sidecarJson.toByteArray()) }
        readJson(conn)
    }

    private fun open(job: UploadJob, path: String, method: String): HttpURLConnection {
        val conn = URL(job.siteUrl.trimEnd('/') + path).openConnection() as HttpURLConnection
        conn.requestMethod = method
        conn.setRequestProperty("Authorization", "Bearer ${job.token}")
        conn.connectTimeout = 15_000
        conn.readTimeout = 60_000
        return conn
    }

    private fun readJson(conn: HttpURLConnection, allow: IntArray = intArrayOf(200)): JSONObject {
        val code = conn.responseCode
        val stream = if (code in 200..299 || code == 409) conn.inputStream else conn.errorStream
        val text = stream?.bufferedReader()?.use { it.readText() } ?: "{}"
        if (code !in allow.toList() && code !in 200..299) {
            throw RuntimeException("upload HTTP $code: $text")
        }
        return JSONObject(text)
    }

    private fun saveProgress(file: File, uploadId: Long, offset: Long) {
        file.writeText(JSONObject().put("uploadId", uploadId).put("offset", offset).toString())
    }
}
