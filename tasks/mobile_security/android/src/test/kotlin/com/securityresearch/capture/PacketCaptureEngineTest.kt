package com.securityresearch.capture

import java.io.ByteArrayInputStream
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class PacketCaptureEngineTest {
    private lateinit var tempDir: File

    @BeforeTest
    fun setUp() {
        tempDir = createTempDir(prefix = "capture-test-")
    }

    @AfterTest
    fun tearDown() {
        tempDir.deleteRecursively()
    }

    @Test
    fun `capture writes frames to pcap file`() {
        val output = File(tempDir, "session.pcap")
        val config = PacketCaptureConfig(outputPath = output.absolutePath, autoStopWhenEmpty = true)
        val engine = PacketCaptureEngine(config, AtomicBoolean(false))
        val payload = ByteArray(128) { it.toByte() }
        val stream = ByteArrayInputStream(payload)

        engine.capture(stream)

        assertTrue(output.exists(), "pcap output should exist")
        val bytes = output.readBytes()
        // File must contain header (24 bytes) + at least one frame header (16 bytes) + payload
        assertTrue(bytes.size >= 24 + 16 + payload.size, "pcap output should include payload")
    }

    @Test
    fun `capture stops when stop signal is set`() {
        val output = File(tempDir, "session-stop.pcap")
        val config = PacketCaptureConfig(outputPath = output.absolutePath, bufferSize = 16)
        val stopSignal = AtomicBoolean(false)
        val engine = PacketCaptureEngine(config, stopSignal)
        val payload = ByteArray(1024) { 0x1 }
        val stream = object : ByteArrayInputStream(payload) {
            override fun read(buffer: ByteArray, off: Int, len: Int): Int {
                stopSignal.set(true)
                return super.read(buffer, off, len)
            }
        }

        engine.capture(stream)

        assertTrue(output.exists(), "pcap output should be created even when stopped early")
        assertEquals(true, stopSignal.get())
    }
}
