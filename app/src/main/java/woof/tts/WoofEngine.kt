package woof.tts

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
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
    val sampleRate = 24000

    init {
        val dir = File(context.filesDir, "woof_models")
        ensureModels(context, dir)
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setInterOpNumThreads(1)
        }
        enc = env.createSession(File(dir, "woof_encoder.onnx").absolutePath, opts)
        step = env.createSession(File(dir, "woof_step.onnx").absolutePath, opts)
        dec = env.createSession(File(dir, "woof_decoder.onnx").absolutePath, opts)
    }

    fun synthesize(text: String, voice: String = "juno", seed: Long = 1L, temp: Float = 0.5f, eos: Float = 0.9f): FloatArray {
        val ids = Tokenizer.encodeText(text)
        if (ids.size <= 1) return FloatArray(0)
        val voiceId = if (voice == "mira") 1L else 0L
        var s = FloatArray(0)
        var eosVal = 0f
        var kc = FloatArray(0)
        var vc = FloatArray(0)
        var kShape = longArrayOf()
        var vShape = longArrayOf()
        val idLongs = LongArray(ids.size) { ids[it].toLong() }
        OnnxTensor.createTensor(env, LongBuffer.wrap(idLongs), longArrayOf(ids.size.toLong())).use { tt ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(voiceId)), longArrayOf()).use { vt ->
                enc.run(mapOf("tokens" to tt, "voice_id" to vt)).use { r ->
                    s = floats(r.get("s").get() as OnnxTensor)
                    eosVal = (r.get("eos").get() as OnnxTensor).floatBuffer.get()
                    val kt = r.get("k_cache").get() as OnnxTensor; kc = floats(kt); kShape = kt.info.shape
                    val vct = r.get("v_cache").get() as OnnxTensor; vc = floats(vct); vShape = vct.info.shape
                }
            }
        }
        val p0 = 16 + ids.size + 1
        val maxFrames = maxOf(1, minOf(750, 1100 - p0))
        val eosT = ln((eos / (1 - eos)).toDouble()).toFloat()
        val rng = java.util.Random(seed)
        val lat = ArrayList<FloatArray>()
        for (f in 0 until maxFrames) {
            if (f > 3 && eosVal > eosT) break
            val noise = FloatArray(32) { (rng.nextGaussian() * temp).toFloat() }
            val sT = OnnxTensor.createTensor(env, FloatBuffer.wrap(s), longArrayOf(s.size.toLong()))
            val nT = OnnxTensor.createTensor(env, FloatBuffer.wrap(noise), longArrayOf(32L))
            val posT = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf((p0 + f).toLong())), longArrayOf())
            val kT = OnnxTensor.createTensor(env, FloatBuffer.wrap(kc), kShape)
            val vT = OnnxTensor.createTensor(env, FloatBuffer.wrap(vc), vShape)
            try {
                step.run(mapOf("s_in" to sT, "noise" to nT, "pos" to posT, "k_in" to kT, "v_in" to vT)).use { r ->
                    lat.add(floats(r.get("z").get() as OnnxTensor))
                    s = floats(r.get("s").get() as OnnxTensor)
                    eosVal = (r.get("eos").get() as OnnxTensor).floatBuffer.get()
                    val kt = r.get("k_cache").get() as OnnxTensor; kc = floats(kt); kShape = kt.info.shape
                    val vct = r.get("v_cache").get() as OnnxTensor; vc = floats(vct); vShape = vct.info.shape
                }
            } finally {
                listOf(sT, nT, posT, kT, vT).forEach { it.close() }
            }
        }
        val t = lat.size
        if (t == 0) return FloatArray(0)
        val flat = FloatArray(t * 32)
        for (i in 0 until t) System.arraycopy(lat[i], 0, flat, i * 32, 32)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), longArrayOf(t.toLong(), 32L)).use { lt ->
            return dec.run(mapOf("latents" to lt)).use { r -> floats(r.get("audio").get() as OnnxTensor) }
        }
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
        for (name in listOf("woof_encoder.onnx", "woof_step.onnx", "woof_decoder.onnx")) {
            val out = File(dir, name)
            if (out.exists() && out.length() > 1024 * 1024) continue
            val tmp = File(dir, "$name.tmp")
            context.assets.open("models/$name").use { input ->
                tmp.outputStream().use { output -> input.copyTo(output) }
            }
            tmp.renameTo(out)
        }
    }

    companion object {
        fun defaultThreads(): Int = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
    }
}