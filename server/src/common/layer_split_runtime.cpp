#include "layer_split_runtime.h"

namespace dflash::common {

bool run_layer_split_ar_decode(
        int last_tok,
        int committed,
        int n_gen,
        int vocab,
        const std::vector<float> & prefill_last_logits,
        const SamplerCfg & sampler,
        std::mt19937_64 & rng,
        const LayerSplitForwardStep & forward_one,
        const std::function<bool(int)> & is_eos,
        std::vector<int32_t> & out_tokens,
        const DaemonIO & io,
        bool seed_already_streamed) {
    if (n_gen <= 0) return true;

    std::vector<float> logits_buf;

    if (!seed_already_streamed) {
        // Fresh decode after prefill: first token comes from prefill logits
        // (when sampling) or the seed argmax, then is streamed.
        if (sampler.needs_logit_processing()) {
            if ((int)prefill_last_logits.size() != vocab) return false;
            last_tok = sample_logits(prefill_last_logits.data(), vocab, sampler,
                                     out_tokens, rng);
        }

        out_tokens.push_back(last_tok);
        io.emit(last_tok);
        if (io.cancelled) {
            io.emit(-1);
            return true;
        }
        if (is_eos(last_tok)) {
            io.emit(-1);
            return true;
        }
        ++committed;
        // Remaining budget after the seed emission.
        --n_gen;
    }

    for (int i = 0; i < n_gen; ++i) {
        std::vector<int32_t> one(1, last_tok);
        int next_tok = -1;
        logits_buf.clear();
        if (!forward_one(one, committed, next_tok,
                         sampler.needs_logit_processing() ? &logits_buf : nullptr)) {
            return false;
        }
        if (sampler.needs_logit_processing()) {
            if ((int)logits_buf.size() != vocab) return false;
            next_tok = sample_logits(logits_buf.data(), vocab, sampler,
                                     out_tokens, rng);
        }

        last_tok = next_tok;
        out_tokens.push_back(last_tok);
        io.emit(last_tok);
        ++committed;
        if (io.cancelled) break;
        if (is_eos(last_tok)) break;
    }

    // Quantum budget exhaustion is NOT stream termination. Multi-request SCHED
    // emits CONTINUE/DONE from finalize_live_quantum / RESTORE_CHAIN admit.
    if (io.cancelled || is_eos(last_tok)) {
        io.emit(-1);
    }
    return true;
}

}  // namespace dflash::common
