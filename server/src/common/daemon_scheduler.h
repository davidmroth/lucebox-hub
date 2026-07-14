// Transport-agnostic multi-request slot / quantum types (Phase 1).
//
// The pipe (stdin) is one client of this model. Phase 5 may attach sockets;
// do not bake "there is only one pipe" into these types.
//
// Distinct from PrefixSnapshot slots (LIST_SLOTS / SNAPSHOT) — those are cold
// KV copies. These are live target-cache slots for concurrent generates.

#pragma once

#include <cstdint>
#include <deque>
#include <vector>

namespace dflash::common {

struct LiveRequestState {
    bool active = false;
    int  request_id = 0;
    int  slot_id = 0;
    int  remaining = 0;
    int  quantum = 0;
    int  emitted = 0;
    int  epoch = 0;
};

struct PendingQuantum {
    int request_id = 0;
    int slot_id = 0;
    int epoch = 0;
    int n_gen = 0;
};

// Fair round-robin quantum enqueue across live requests.
inline bool enqueue_next_quantum(
    std::vector<LiveRequestState> & requests,
    std::deque<PendingQuantum> & pending,
    size_t & cursor) {
    if (requests.empty()) return false;
    const size_t n = requests.size();
    for (size_t i = 0; i < n; ++i) {
        const size_t idx = (cursor + i) % n;
        LiveRequestState & req = requests[idx];
        if (!req.active || req.remaining <= 0) continue;
        const int q = req.quantum > 0
            ? (req.remaining < req.quantum ? req.remaining : req.quantum)
            : (req.remaining < 1 ? req.remaining : 1);
        if (q <= 0) continue;
        pending.push_back(PendingQuantum{
            req.request_id, req.slot_id, req.epoch, q});
        cursor = (idx + 1) % n;
        return true;
    }
    return false;
}

// Stream-tagged frame markers (opt-in --stream-tagged).
constexpr int32_t kStreamTagMarker     = -2;
constexpr int32_t kStreamContinueSentinel = -4;
constexpr int32_t kStreamDoneSentinel  = -1;

}  // namespace dflash::common
