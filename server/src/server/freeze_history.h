// freeze_history — pure partition logic for FlowKV freeze-history feature.
//
// Partitions a token stream into three regions by turn boundary:
//   VERBATIM PREFIX : turns[0] (system + tool-defs) — never compressed.
//   FROZEN region   : aged conversational/tool turns after the system prefix,
//                     up to the hot window — compressed once and cached.
//   HOT TAIL        : the last hot_window_turns turns — kept verbatim.
//
// Pure functions: no IO, no globals, no CUDA deps. Tested standalone.

#pragma once

#include "server/prefix_cache.h"  // PrefixHash

#include <cstdint>
#include <vector>

namespace dflash::common {

// ─── Pure functions ───────────────────────────────────────────────────────

// Compute a stable content-hash of a token slice [begin, end).
// Reuses hash_prefix from prefix_cache so no SHA-1 is re-implemented here.
//
// Returns a zeroed PrefixHash when the slice is empty (begin >= end).
PrefixHash frozen_block_key(const int32_t * ids, int begin, int end);

}  // namespace dflash::common
