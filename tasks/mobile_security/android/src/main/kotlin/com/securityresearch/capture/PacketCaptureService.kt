package com.securityresearch.capture

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.net.VpnService
import android.os.Build
import android.os.ParcelFileDescriptor
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import java.io.BufferedOutputStream
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.util.concurrent.atomic.AtomicBoolean

/**
 * High-fidelity packet capture service backed by [VpnService].
 *
 * The service establishes an in-process virtual network interface and streams
 * raw IP packets to a pcap file suitable for offline analysis in Wireshark. It
 * is designed for legitimate security research in controlled environments and
 * requires the `android.permission.INTERNET` permission plus the VPN service
 * user consent flow managed by the system.
 */
class PacketCaptureService : VpnService() {
    private val loggerTag = "PacketCaptureService"
    private val stopSignal = AtomicBoolean(false)
    private var captureThread: Thread? = null
    private var captureEngine: PacketCaptureEngine? = null
    private var isForeground = false

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP_CAPTURE) {
            stopSignal.set(true)
            captureThread?.interrupt()
            stopSelf()
            return START_NOT_STICKY
        }

        val config = PacketCaptureConfig.fromIntent(intent)
        Log.i(loggerTag, "Starting packet capture with config: $config")
        ensureForeground()
        val vpnInterface = establishInterface(config)
        stopSignal.set(false)
        captureEngine = PacketCaptureEngine(config, stopSignal)
        captureThread = Thread({ runCapture(config, vpnInterface) }, "vpn-capture").apply { start() }
        return START_STICKY
    }

    override fun onDestroy() {
        stopSignal.set(true)
        captureThread?.interrupt()
        try {
            captureThread?.join(2_000)
        } catch (interrupted: InterruptedException) {
            Thread.currentThread().interrupt()
            Log.w(loggerTag, "Interrupted while stopping capture thread", interrupted)
        }
        captureThread = null
        captureEngine = null
        if (isForeground) {
            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
            isForeground = false
        }
        Log.i(loggerTag, "Packet capture service destroyed")
        super.onDestroy()
    }

    private fun ensureForeground() {
        if (isForeground) {
            return
        }
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
            manager.getNotificationChannel(FOREGROUND_CHANNEL_ID) == null
        ) {
            val channel = NotificationChannel(
                FOREGROUND_CHANNEL_ID,
                "Packet Capture",
                NotificationManager.IMPORTANCE_LOW,
            )
            channel.description = "Notifications for active packet capture sessions"
            manager.createNotificationChannel(channel)
        }
        val notification = NotificationCompat.Builder(this, FOREGROUND_CHANNEL_ID)
            .setContentTitle("Packet capture running")
            .setContentText("Capturing VPN traffic for analysis")
            .setSmallIcon(android.R.drawable.stat_sys_data_connect)
            .setOngoing(true)
            .build()
        startForeground(FOREGROUND_NOTIFICATION_ID, notification)
        isForeground = true
    }

    private fun establishInterface(config: PacketCaptureConfig): ParcelFileDescriptor {
        val builder = Builder()
            .setSession(config.sessionName)
            .setMtu(config.mtu)
        config.addresses.forEach { builder.addAddress(it.address, it.prefixLength) }
        config.routes.forEach { builder.addRoute(it.address, it.prefixLength) }
        config.dnsServers.forEach { builder.addDnsServer(it) }
        config.searchDomains.forEach { builder.addSearchDomain(it) }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU && config.metered != null) {
            builder.setMetered(config.metered)
        }
        config.configurePendingIntent?.let { builder.setConfigureIntent(it(this)) }

        val pfd = builder.establish()
        requireNotNull(pfd) { "Unable to establish VPN interface" }
        return pfd
    }

    private fun runCapture(config: PacketCaptureConfig, interfaceFd: ParcelFileDescriptor) {
        val engine = requireNotNull(captureEngine) { "Capture engine unavailable" }
        try {
            ParcelFileDescriptor.AutoCloseInputStream(interfaceFd).use { input ->
                engine.capture(input)
            }
        } catch (ex: IOException) {
            Log.e(loggerTag, "Packet capture terminated unexpectedly", ex)
        }
        if (config.autoStopWhenEmpty) {
            stopSelf()
        }
    }

    companion object {
        const val ACTION_STOP_CAPTURE = "com.securityresearch.capture.STOP"

        /**
         * Request that the capture service start recording traffic using the provided configuration.
         */
        fun start(context: Context, config: PacketCaptureConfig) {
            val intent = Intent(context, PacketCaptureService::class.java)
            config.writeToIntent(intent)
            ContextCompat.startForegroundService(context, intent)
        }

        /**
         * Stop the currently running capture session, if any.
         */
        fun stop(context: Context) {
            val intent = Intent(context, PacketCaptureService::class.java)
            intent.action = ACTION_STOP_CAPTURE
            ContextCompat.startForegroundService(context, intent)
        }

        private const val FOREGROUND_CHANNEL_ID = "capture"
        private const val FOREGROUND_NOTIFICATION_ID = 1
    }
}

/**
 * Immutable configuration describing how a capture session should be created.
 */
data class PacketCaptureConfig(
    val sessionName: String = "SecurityResearchCapture",
    val mtu: Int = 1500,
    val addresses: List<CaptureAddress> = listOf(CaptureAddress("10.0.0.2", 32)),
    val routes: List<CaptureRoute> = listOf(CaptureRoute("0.0.0.0", 0)),
    val dnsServers: List<String> = emptyList(),
    val searchDomains: List<String> = emptyList(),
    val outputPath: String = File.createTempFile("capture", ".pcap", null).absolutePath,
    val bufferSize: Int = 4096,
    val autoStopWhenEmpty: Boolean = false,
    val metered: Boolean? = null,
    val configurePendingIntent: ((Context) -> PendingIntent)? = null
) {
    fun writeToIntent(intent: Intent) {
        intent.putExtra(KEY_CONFIG, toBundle())
    }

    private fun toBundle() = android.os.Bundle().apply {
        putString(KEY_SESSION_NAME, sessionName)
        putInt(KEY_MTU, mtu)
        putStringArrayList(KEY_ADDRESSES, ArrayList(addresses.map { it.serialise() }))
        putStringArrayList(KEY_ROUTES, ArrayList(routes.map { it.serialise() }))
        putStringArrayList(KEY_DNS, ArrayList(dnsServers))
        putStringArrayList(KEY_SEARCH_DOMAINS, ArrayList(searchDomains))
        putString(KEY_OUTPUT_PATH, outputPath)
        putInt(KEY_BUFFER_SIZE, bufferSize)
        putBoolean(KEY_AUTO_STOP, autoStopWhenEmpty)
        metered?.let { putBoolean(KEY_METERED, it) }
    }

    companion object {
        private const val KEY_CONFIG = "capture_config"
        private const val KEY_SESSION_NAME = "session_name"
        private const val KEY_MTU = "mtu"
        private const val KEY_ADDRESSES = "addresses"
        private const val KEY_ROUTES = "routes"
        private const val KEY_DNS = "dns"
        private const val KEY_SEARCH_DOMAINS = "search_domains"
        private const val KEY_OUTPUT_PATH = "output"
        private const val KEY_BUFFER_SIZE = "buffer_size"
        private const val KEY_AUTO_STOP = "auto_stop"
        private const val KEY_METERED = "metered"

        fun fromIntent(intent: Intent?): PacketCaptureConfig {
            val extras = intent?.getBundleExtra(KEY_CONFIG) ?: return PacketCaptureConfig()
            return PacketCaptureConfig(
                sessionName = extras.getString(KEY_SESSION_NAME, "SecurityResearchCapture"),
                mtu = extras.getInt(KEY_MTU, 1500),
                addresses = extras.getStringArrayList(KEY_ADDRESSES)?.map { CaptureAddress.deserialize(it) }
                    ?: listOf(CaptureAddress("10.0.0.2", 32)),
                routes = extras.getStringArrayList(KEY_ROUTES)?.map { CaptureRoute.deserialize(it) }
                    ?: listOf(CaptureRoute("0.0.0.0", 0)),
                dnsServers = extras.getStringArrayList(KEY_DNS) ?: emptyList(),
                searchDomains = extras.getStringArrayList(KEY_SEARCH_DOMAINS) ?: emptyList(),
                outputPath = extras.getString(KEY_OUTPUT_PATH)
                    ?: File.createTempFile("capture", ".pcap").absolutePath,
                bufferSize = extras.getInt(KEY_BUFFER_SIZE, 4096),
                autoStopWhenEmpty = extras.getBoolean(KEY_AUTO_STOP, false),
                metered = extras.takeIf { it.containsKey(KEY_METERED) }?.getBoolean(KEY_METERED)
            )
        }
    }
}

data class CaptureAddress(val address: String, val prefixLength: Int) {
    fun serialise(): String = "$address/$prefixLength"

    companion object {
        fun deserialize(raw: String): CaptureAddress {
            val parts = raw.split('/')
            require(parts.size == 2) { "Invalid address serialisation: $raw" }
            return CaptureAddress(parts[0], parts[1].toInt())
        }
    }
}

data class CaptureRoute(val address: String, val prefixLength: Int) {
    fun serialise(): String = "$address/$prefixLength"

    companion object {
        fun deserialize(raw: String): CaptureRoute {
            val parts = raw.split('/')
            require(parts.size == 2) { "Invalid route serialisation: $raw" }
            return CaptureRoute(parts[0], parts[1].toInt())
        }
    }
}

/**
 * PacketCaptureEngine reads packets from the VPN interface and writes PCAP output to disk.
 */
class PacketCaptureEngine(
    private val config: PacketCaptureConfig,
    private val stopSignal: AtomicBoolean
) {
    fun capture(inputStream: InputStream) {
        val outputFile = File(config.outputPath)
        outputFile.parentFile?.mkdirs()
        PcapWriter(outputFile).use { writer ->
            val buffer = ByteArray(config.bufferSize)
            while (!stopSignal.get()) {
                val bytesRead = try {
                    inputStream.read(buffer)
                } catch (interrupted: IOException) {
                    if (stopSignal.get()) {
                        break
                    }
                    throw interrupted
                }
                if (bytesRead <= 0) {
                    if (config.autoStopWhenEmpty) {
                        break
                    }
                    continue
                }
                writer.writeFrame(buffer, bytesRead)
            }
        }
    }
}

/**
 * Minimal PCAP writer that produces headers compatible with Wireshark.
 */
class PcapWriter(private val file: File) : AutoCloseable {
    private val output = BufferedOutputStream(file.outputStream(), 64 * 1024)

    init {
        writeHeader()
    }

    fun writeFrame(data: ByteArray, length: Int) {
        val timestamp = System.currentTimeMillis()
        val seconds = (timestamp / 1000).toInt()
        val micros = ((timestamp % 1000) * 1000).toInt()
        val header = ByteArray(16)
        header.writeInt32LE(0, seconds)
        header.writeInt32LE(4, micros)
        header.writeInt32LE(8, length)
        header.writeInt32LE(12, length)
        output.write(header)
        output.write(data, 0, length)
    }

    private fun writeHeader() {
        val header = ByteArray(24)
        header.writeInt32LE(0, 0xD4C3B2A1.toInt())
        header.writeInt16LE(4, 2)
        header.writeInt16LE(6, 4)
        header.writeInt32LE(8, 0)
        header.writeInt32LE(12, 0)
        header.writeInt32LE(16, 65535)
        header.writeInt32LE(20, 101) // LINKTYPE_RAW
        output.write(header)
    }

    override fun close() {
        try {
            output.flush()
            output.close()
        } catch (ex: IOException) {
            Log.w("PcapWriter", "Failed to close pcap output", ex)
        }
    }
}

private fun ByteArray.writeInt16LE(offset: Int, value: Int) {
    this[offset] = (value and 0xFF).toByte()
    this[offset + 1] = ((value shr 8) and 0xFF).toByte()
}

private fun ByteArray.writeInt32LE(offset: Int, value: Int) {
    this[offset] = (value and 0xFF).toByte()
    this[offset + 1] = ((value shr 8) and 0xFF).toByte()
    this[offset + 2] = ((value shr 16) and 0xFF).toByte()
    this[offset + 3] = ((value shr 24) and 0xFF).toByte()
}
