package com.varbox.camera.clock

import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress

/** One SNTP measurement: the device-to-server clock offset and the round trip it came from. */
data class ClockSample(val offsetMs: Double, val roundTripMs: Double)

/**
 * Minimal SNTP client. `offsetMs` is (server time − device time): add it to
 * System.currentTimeMillis() to read the shared reference clock. One UDP exchange per call.
 */
object SntpClient {
    private const val NTP_PORT = 123
    private const val PACKET_SIZE = 48
    private const val MODE_CLIENT = 3
    private const val VERSION = 3
    private const val SECONDS_1900_TO_1970 = 2208988800L
    private const val TWO_POW_32 = 4294967296.0

    fun query(host: String, timeoutMs: Int = 2000): ClockSample {
        DatagramSocket().use { socket ->
            socket.soTimeout = timeoutMs
            val address = InetAddress.getByName(host)
            val buf = ByteArray(PACKET_SIZE)
            buf[0] = ((VERSION shl 3) or MODE_CLIENT).toByte()
            val t0 = System.currentTimeMillis()
            writeTimestamp(buf, 40, t0)
            socket.send(DatagramPacket(buf, buf.size, address, NTP_PORT))
            socket.receive(DatagramPacket(buf, buf.size))
            val t3 = System.currentTimeMillis()
            val serverReceive = readTimestamp(buf, 32) // t1
            val serverTransmit = readTimestamp(buf, 40) // t2
            val roundTrip = (t3 - t0) - (serverTransmit - serverReceive)
            val offset = ((serverReceive - t0) + (serverTransmit - t3)) / 2.0
            return ClockSample(offset, roundTrip.toDouble())
        }
    }

    private fun writeTimestamp(buffer: ByteArray, offset: Int, timeMs: Long) {
        val seconds = timeMs / 1000L + SECONDS_1900_TO_1970
        val millis = timeMs % 1000L
        buffer[offset] = (seconds shr 24).toByte()
        buffer[offset + 1] = (seconds shr 16).toByte()
        buffer[offset + 2] = (seconds shr 8).toByte()
        buffer[offset + 3] = seconds.toByte()
        val fraction = (millis * TWO_POW_32 / 1000.0).toLong()
        buffer[offset + 4] = (fraction shr 24).toByte()
        buffer[offset + 5] = (fraction shr 16).toByte()
        buffer[offset + 6] = (fraction shr 8).toByte()
        buffer[offset + 7] = (Math.random() * 255.0).toInt().toByte()
    }

    private fun readTimestamp(buffer: ByteArray, offset: Int): Long {
        val seconds = read32(buffer, offset)
        val fraction = read32(buffer, offset + 4)
        return (seconds - SECONDS_1900_TO_1970) * 1000L + (fraction * 1000.0 / TWO_POW_32).toLong()
    }

    private fun read32(buffer: ByteArray, offset: Int): Long {
        val b0 = (buffer[offset].toInt() and 0xFF).toLong()
        val b1 = (buffer[offset + 1].toInt() and 0xFF).toLong()
        val b2 = (buffer[offset + 2].toInt() and 0xFF).toLong()
        val b3 = (buffer[offset + 3].toInt() and 0xFF).toLong()
        return (b0 shl 24) or (b1 shl 16) or (b2 shl 8) or b3
    }
}
