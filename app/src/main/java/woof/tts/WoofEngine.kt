package woof.tts

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.Process
import woof.tts.text.Normalizer
import woof.tts.text.Respeller
import woof.tts.text.Tokenizer
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.ln

/** Drives the three Woof ONNX graphs (encoder -> autoregressive step loop -> decoder), mirroring
 *  export_onnx.WoofONNX. Produces 24 kHz mono float audio in [-1, 1]. */
class WoofEngine(context: Context, threads: Int = defaultThreads()) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val enc: OrtSession
    private val step: OrtSession
    private val dec: OrtSession
    private val splitKv: Boolean
    private val numLayers: Int
    private val synthLock = Any()
    val sampleRate = 24000

    init {
        val dir = File(context.filesDir, "woof_models")
        ensureModels(context, dir)
        val encThreads = threads.coerceIn(1, 2)
        // Step is M=1 GEMV; single-thread execution avoids per-op thread-pool barrier overhead on mobile.
        val stepThreads = 1
        val decThreads = threads.coerceIn(1, 4)

        val sessions = withAudioPriority {
            Triple(
                sessionOpts(encThreads).use { env.createSession(File(dir, "woof_encoder.onnx").absolutePath, it) },
                sessionOpts(stepThreads).use { env.createSession(File(dir, "woof_step.onnx").absolutePath, it) },
                sessionOpts(decThreads).use { env.createSession(File(dir, "woof_decoder.onnx").absolutePath, it) }
            )
        }
        enc = sessions.first
        step = sessions.second
        dec = sessions.third

        val stepIns = step.inputNames
        splitKv = stepIns.contains("k_in0")
        numLayers = if (splitKv) stepIns.count { it.startsWith("k_in") } else 0
    }

    private fun sessionOpts(intraThreads: Int): OrtSession.SessionOptions =
        OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(intraThreads)
            setInterOpNumThreads(1)
        }

    fun synthesize(text: String, voice: String = "juno", seed: Long = 1L, temp: Float = 0.5f, eos: Float = 0.9f): FloatArray {
        val chunks = ArrayList<FloatArray>()
        var total = 0
        synthesizeStreaming(
            text = text,
            voice = voice,
            seed = seed,
            temp = temp,
            eos = eos,
            firstChunkFrames = 0,
            chunkStrideFrames = 0
        ) { chunk ->
            chunks.add(chunk)
            total += chunk.size
        }
        if (chunks.size == 1) return chunks[0]
        val out = FloatArray(total)
        var off = 0
        for (c in chunks) {
            System.arraycopy(c, 0, out, off, c.size)
            off += c.size
        }
        return out
    }

    /** Synthesizes [text] and emits 24 kHz mono float chunks via [onChunk]. When [firstChunkFrames] > 0,
     *  long clauses decode causally in chunks starting at [firstChunkFrames] (using overlap + crossfade)
     *  to cut Time-To-First-Audio while keeping playback gapless. */
    fun synthesizeStreaming(
        text: String,
        voice: String = "juno",
        seed: Long = 1L,
        temp: Float = 0.5f,
        eos: Float = 0.9f,
        firstChunkFrames: Int = 44,
        chunkStrideFrames: Int = CHUNK_STRIDE_FRAMES,
        isCancelled: () -> Boolean = { false },
        onChunk: (FloatArray) -> Unit
    ) {
        val ids = Tokenizer.encodeText(text)
        if (ids.size <= 1 || isCancelled()) return
        synchronized(synthLock) {
            if (isCancelled()) return
            withAudioPriority {
                synthesizeStreamingImpl(ids, voice, seed, temp, eos, firstChunkFrames, chunkStrideFrames, isCancelled, onChunk)
            }
        }
    }

    private fun synthesizeStreamingImpl(
        ids: IntArray,
        voice: String,
        seed: Long,
        temp: Float,
        eos: Float,
        firstChunkFrames: Int,
        chunkStrideFrames: Int,
        isCancelled: () -> Boolean,
        onChunk: (FloatArray) -> Unit
    ) {
        val voiceId = if (voice.trim().contains("mira", ignoreCase = true)) 1L else 0L
        val idLongs = LongArray(ids.size) { ids[it].toLong() }

        var stateRes: OrtSession.Result? = null
        try {
            OnnxTensor.createTensor(env, LongBuffer.wrap(idLongs), longArrayOf(ids.size.toLong())).use { tt ->
                OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(voiceId)), longArrayOf()).use { vt ->
                    stateRes = enc.run(mapOf("tokens" to tt, "voice_id" to vt))
                }
            }
            val p0 = 16 + ids.size + 1
            // Bound worst-case runaway loops while giving >2.5x headroom for slow expressive speech.
            val maxFrames = maxOf(1, minOf(750, maxOf(120, ids.size * 4 + 100), 1100 - p0))
            val eosT = ln((eos / (1 - eos)).toDouble()).toFloat()
            val rng = java.util.Random(seed)
            val lat = ArrayList<FloatArray>()
            val noiseArr = FloatArray(LATENT_DIM)
            val posArr = LongArray(1)

            var emittedSamples = 0
            var prevTail: FloatArray? = null
            var nextDecodeAt = if (firstChunkFrames > 0) firstChunkFrames else Int.MAX_VALUE

            for (f in 0 until maxFrames) {
                if (isCancelled()) return
                val curRes = stateRes!!
                val eosVal = (curRes.get("eos").get() as OnnxTensor).floatBuffer.get(0)
                if (f > 3 && eosVal > eosT) break

                for (i in 0 until LATENT_DIM) {
                    noiseArr[i] = (rng.nextGaussian() * temp).toFloat()
                }
                posArr[0] = (p0 + f).toLong()

                val nT = OnnxTensor.createTensor(env, FloatBuffer.wrap(noiseArr), longArrayOf(LATENT_DIM.toLong()))
                val posT = OnnxTensor.createTensor(env, LongBuffer.wrap(posArr), longArrayOf())
                val nextRes: OrtSession.Result
                try {
                    val feeds = HashMap<String, OnnxTensor>(if (splitKv) 3 + numLayers * 2 else 5)
                    feeds["s_in"] = curRes.get("s").get() as OnnxTensor
                    feeds["noise"] = nT
                    feeds["pos"] = posT
                    if (splitKv) {
                        for (l in 0 until numLayers) {
                            feeds["k_in$l"] = curRes.get("k_cache$l").get() as OnnxTensor
                            feeds["v_in$l"] = curRes.get("v_cache$l").get() as OnnxTensor
                        }
                    } else {
                        feeds["k_in"] = curRes.get("k_cache").get() as OnnxTensor
                        feeds["v_in"] = curRes.get("v_cache").get() as OnnxTensor
                    }
                    nextRes = step.run(feeds)
                } finally {
                    nT.close()
                    posT.close()
                }
                // Close prior step Result only after next step consumes its KV/s tensors zero-copy.
                stateRes = nextRes
                curRes.close()

                lat.add(floats(nextRes.get("z").get() as OnnxTensor))

                if (firstChunkFrames > 0 && lat.size == nextDecodeAt && !isCancelled()) {
                    val stepResult = decodeChunk(lat, lat.size, isLast = false, emittedSamples, prevTail)
                    emittedSamples = stepResult.emittedSamples
                    prevTail = stepResult.tail
                    if (stepResult.audio.isNotEmpty()) onChunk(stepResult.audio)
                    nextDecodeAt = if (chunkStrideFrames > 0) nextDecodeAt + chunkStrideFrames else Int.MAX_VALUE
                }
            }

            if (lat.isNotEmpty() && !isCancelled()) {
                val finalResult = decodeChunk(lat, lat.size, isLast = true, emittedSamples, prevTail)
                if (finalResult.audio.isNotEmpty()) onChunk(finalResult.audio)
            }
        } finally {
            stateRes?.close()
        }
    }

    private class ChunkResult(val audio: FloatArray, val emittedSamples: Int, val tail: FloatArray?)

    private fun decodeChunk(
        lat: List<FloatArray>,
        totalFrames: Int,
        isLast: Boolean,
        emittedSamples: Int,
        prevTail: FloatArray?
    ): ChunkResult {
        val curFrame = emittedSamples / SAMPLES_PER_FRAME
        val startFrame = maxOf(0, curFrame - DECODER_OVERLAP_FRAMES)
        val count = totalFrames - startFrame
        val flat = FloatArray(count * LATENT_DIM)
        for (i in 0 until count) {
            System.arraycopy(lat[startFrame + i], 0, flat, i * LATENT_DIM, LATENT_DIM)
        }
        val decoded = OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), longArrayOf(count.toLong(), LATENT_DIM.toLong())).use { lt ->
            dec.run(mapOf("latents" to lt)).use { r -> floats(r.get("audio").get() as OnnxTensor) }
        }
        val emitEndFrame = if (isLast) totalFrames else maxOf(startFrame + 1, totalFrames - 1)
        val sliceStart = emittedSamples - startFrame * SAMPLES_PER_FRAME
        val sliceEnd = minOf(decoded.size, (emitEndFrame - startFrame) * SAMPLES_PER_FRAME)
        if (sliceEnd <= sliceStart) return ChunkResult(FloatArray(0), emittedSamples, prevTail)

        val segLen = sliceEnd - sliceStart
        val seg = decoded.copyOfRange(sliceStart, sliceEnd)
        if (prevTail != null) {
            val cf = minOf(prevTail.size, seg.size)
            for (i in 0 until cf) {
                val a = i.toFloat() / cf
                seg[i] = prevTail[i] * (1f - a) + seg[i] * a
            }
        }
        if (!isLast && segLen > CROSSFADE_SAMPLES) {
            val tail = seg.copyOfRange(segLen - CROSSFADE_SAMPLES, segLen)
            val emit = seg.copyOfRange(0, segLen - CROSSFADE_SAMPLES)
            return ChunkResult(emit, emittedSamples + emit.size, tail)
        }
        return ChunkResult(seg, emittedSamples + seg.size, null)
    }

    fun close() {
        enc.close(); step.close(); dec.close()
    }

    private fun floats(t: OnnxTensor): FloatArray {
        val b = t.floatBuffer
        val a = FloatArray(b.remaining())
        b.get(a)
        return a
    }

    private fun ensureModels(context: Context, dir: File) {
        dir.mkdirs()
        val apkStamp = File(context.applicationInfo.sourceDir).let { "${it.length()}:${it.lastModified()}" }
        val stampFile = File(dir, ".stamp")
        val stampValid = stampFile.exists() && runCatching { stampFile.readText().trim() == apkStamp }.getOrDefault(false)
        var updated = false

        for (name in listOf("woof_encoder.onnx", "woof_step.onnx", "woof_decoder.onnx")) {
            val out = File(dir, name)
            val expectedLen = try {
                context.assets.openFd("models/$name").use { it.length }
            } catch (_: Exception) {
                -1L
            }
            val valid = stampValid && if (expectedLen > 0) {
                out.exists() && out.length() == expectedLen
            } else {
                out.exists() && out.length() > 1024 * 1024
            }
            if (valid) continue
            updated = true
            val tmp = File(dir, "$name.tmp")
            context.assets.open("models/$name").use { input ->
                tmp.outputStream().use { output -> input.copyTo(output) }
            }
            if (out.exists()) out.delete()
            tmp.renameTo(out)
        }
        if (updated || !stampValid) {
            runCatching { stampFile.writeText(apkStamp) }
        }
    }

    companion object {
        private const val LATENT_DIM = 32
        private const val SAMPLES_PER_FRAME = 960
        private const val DECODER_OVERLAP_FRAMES = 24
        private const val CROSSFADE_SAMPLES = 240
        const val CHUNK_STRIDE_FRAMES = 36

        /** Target frames before first audio emit per clause: larger first chunk prevents Chunk 2 underruns, smaller follow-up keeps inter-sentence latency near zero. */
        fun firstChunkFramesForClause(clauseIndex: Int): Int = if (clauseIndex == 0) 44 else 28

        fun defaultThreads(): Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 2)

        private var sharedEngine: WoofEngine? = null
        private var sharedNormalizer: Normalizer? = null
        private var sharedRefs = 0
        private val sharedLock = Any()

        /** Returns process-wide shared engine and text normalizer so Activity and TtsService do not duplicate ~330 MB+ ORT sessions. */
        fun acquireShared(context: Context): Pair<WoofEngine, Normalizer> = synchronized(sharedLock) {
            val app = context.applicationContext
            val eng = sharedEngine ?: WoofEngine(app).also { sharedEngine = it }
            val norm = sharedNormalizer ?: Normalizer(
                Respeller().apply {
                    load(app.assets.open("respell.tsv").bufferedReader().use { it.readText() })
                }
            ).also { sharedNormalizer = it }
            sharedRefs++
            eng to norm
        }

        fun releaseShared() {
            val toClose = synchronized(sharedLock) {
                if (sharedRefs > 0) sharedRefs--
                if (sharedRefs == 0) {
                    sharedEngine.also {
                        sharedEngine = null
                        sharedNormalizer = null
                    }
                } else null
            }
            toClose?.close()
        }

        private inline fun <T> withAudioPriority(block: () -> T): T {
            val tid = Process.myTid()
            val prev = Process.getThreadPriority(tid)
            var raised = false
            for (prio in intArrayOf(
                Process.THREAD_PRIORITY_AUDIO,
                Process.THREAD_PRIORITY_URGENT_DISPLAY,
                Process.THREAD_PRIORITY_DISPLAY
            )) {
                if (runCatching { Process.setThreadPriority(tid, prio) }.isSuccess) {
                    raised = true
                    break
                }
            }
            return try {
                block()
            } finally {
                if (raised) runCatching { Process.setThreadPriority(tid, prev) }
            }
        }
    }
}