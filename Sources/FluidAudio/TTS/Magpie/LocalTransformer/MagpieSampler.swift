import Foundation

/// Samples the 8 codebook tokens from one decoder hidden state by driving the
/// Swift Local Transformer auto-regressively.
///
/// Mirrors `local_transformer_sample` in
/// `mobius/models/tts/magpie/coreml/generate_coreml.py` (lines 172–242).
public struct MagpieLocalSampler: Sendable {

    private let lt: MagpieLocalTransformer
    private let audioEmbeddings: [[Float]]

    /// - Parameter audioEmbeddings: per-codebook `[numCodesPerCodebook × dModel]` fp32.
    public init(
        localTransformer: MagpieLocalTransformer,
        audioEmbeddings: [[Float]]
    ) {
        self.lt = localTransformer
        self.audioEmbeddings = audioEmbeddings
    }

    /// Sample one frame of `numCodebooks` codes.
    ///
    /// - Parameters:
    ///   - decoderHidden: conditional decoder hidden state, `[dModel]`.
    ///   - uncondDecoderHidden: unconditional path for CFG; `nil` disables CFG.
    ///   - forbidEos: mask `audioEosId` (set `true` while `t < minFrames`).
    ///   - options: temperature / topK / cfgScale.
    ///   - rng: caller-owned RNG so the whole generation can be seeded.
    public func sample(
        decoderHidden: [Float],
        uncondDecoderHidden: [Float]? = nil,
        forbidEos: Bool,
        options: MagpieSynthesisOptions,
        rng: inout any RandomNumberGenerator
    ) -> [Int32] {
        let numCodebooks = lt.weights.numCodebooks
        let D = lt.weights.localDim
        let useCfg = uncondDecoderHidden != nil && options.cfgScale != 1.0

        // Project decoder hidden through in_proj → first LT token.
        let condFirst = lt.projectInput(hidden: decoderHidden)
        var condSeq = condFirst  // growing buffer, flat row-major
        var condLen = 1

        var uncondSeq: [Float] = []
        var uncondLen = 0
        if let uncondHidden = uncondDecoderHidden {
            uncondSeq = lt.projectInput(hidden: uncondHidden)
            uncondLen = 1
        }

        var codes = Swift.Array<Int32>(repeating: 0, count: numCodebooks)
        let forbidden = forbiddenTokens(eosMasked: forbidEos)

        for cb in 0..<numCodebooks {
            let condOut = lt.forward(sequence: condSeq, length: condLen)
            let lastOffset = (condLen - 1) * D
            let lastHidden = Swift.Array(condOut[lastOffset..<(lastOffset + D)])
            var logits = lt.codebookLogits(lastHidden: lastHidden, codebook: cb)

            if useCfg, let _ = uncondDecoderHidden {
                let uncondOut = lt.forward(sequence: uncondSeq, length: uncondLen)
                let uncondLast = Swift.Array(
                    uncondOut[((uncondLen - 1) * D)..<((uncondLen - 1) * D + D)])
                let uncondLogits = lt.codebookLogits(lastHidden: uncondLast, codebook: cb)
                let scale = options.cfgScale
                for i in 0..<logits.count {
                    logits[i] = scale * logits[i] + (1.0 - scale) * uncondLogits[i]
                }
            }

            // Mask forbidden tokens.
            for tok in forbidden where Int(tok) < logits.count {
                logits[Int(tok)] = -.infinity
            }

            let sampled = sampleTopK(
                logits: logits, topK: options.topK, temperature: options.temperature,
                rng: &rng)
            codes[cb] = Int32(sampled)

            // Embed sampled token → next LT input (both cond and uncond paths).
            let tokenEmb = audioEmbeddings[cb]
            let row = Int(sampled)
            let start = row * lt.weights.dModel
            let hiddenSlice = Swift.Array(tokenEmb[start..<(start + lt.weights.dModel)])
            let nextInput = lt.projectInput(hidden: hiddenSlice)

            condSeq.append(contentsOf: nextInput)
            condLen += 1
            if useCfg {
                uncondSeq.append(contentsOf: nextInput)
                uncondLen += 1
            }
        }

        return codes
    }

    // MARK: - Sampling utils

    private func forbiddenTokens(eosMasked: Bool) -> [Int32] {
        if eosMasked {
            // Block EOS + CTX_BOS + reserved.
            return [MagpieConstants.audioEosId] + MagpieConstants.forbiddenAudioIds
        } else {
            return MagpieConstants.forbiddenAudioIds
        }
    }

    /// Categorical sampling with optional top-k truncation + temperature.
    ///
    /// Matches the Python reference: select top-k logits (others → -inf), then
    /// softmax with temperature, then multinomial draw.
    private func sampleTopK(
        logits: [Float], topK: Int, temperature: Float,
        rng: inout any RandomNumberGenerator
    ) -> Int {
        var truncated = logits
        if topK > 0 && topK < truncated.count {
            // Find kth-largest threshold via partial sort.
            var indexed = truncated.enumerated().map { ($0.offset, $0.element) }
            indexed.sort { $0.1 > $1.1 }
            let threshold = indexed[topK - 1].1
            for i in 0..<truncated.count {
                if truncated[i] < threshold {
                    truncated[i] = -.infinity
                }
            }
        }
        let t = max(temperature, 1e-8)
        for i in 0..<truncated.count {
            truncated[i] /= t
        }
        let maxVal = truncated.max() ?? 0
        var sum: Float = 0
        for i in 0..<truncated.count {
            let e = expf(truncated[i] - maxVal)
            truncated[i] = e
            sum += e
        }
        if sum <= 0 || !sum.isFinite {
            // Degenerate — fall back to argmax over original logits.
            return logits.indices.max(by: { logits[$0] < logits[$1] }) ?? 0
        }
        let u = Float.random(in: 0..<1, using: &rng) * sum
        var cumulative: Float = 0
        for i in 0..<truncated.count {
            cumulative += truncated[i]
            if cumulative >= u {
                return i
            }
        }
        return truncated.count - 1
    }
}
