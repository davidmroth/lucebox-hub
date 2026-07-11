// Sharded prefix snapshots for layer-split (thick, thin, RESTORE_CHAIN).

#include "qwen35_layer_split_adapter.h"

#include "internal.h"

#include <cstdio>
#include <vector>

namespace dflash::common {

bool Qwen35LayerSplitAdapter::snapshot_is_thin(int slot) const {
    if (!snapshot_slot_valid(slot)) return false;
    const auto & snaps = prefix_snapshots_[(size_t)slot];
    if (snaps.size() != shards_.size()) return false;
    for (const auto & snap : snaps) {
        if (!snap.ctx || !snap.is_thin) return false;
    }
    return true;
}

bool Qwen35LayerSplitAdapter::snapshot_save_thin(int slot,
                                                 int kv_start,
                                                 int kv_end) {
    if (!snapshot_slot_valid(slot)) return false;
    if (snapshot_backends_.size() != shards_.size()) return false;
    if (kv_end <= kv_start || kv_start < 0) return false;
    const int cur_pos = shards_.empty() ? 0 : shards_.front().cache.cur_pos;
    if (kv_end > cur_pos) {
        set_last_error("snapshot_save_thin: kv_end exceeds cur_pos");
        return false;
    }
    if (kvflash_active() &&
        (cur_pos > kvflash_tokens_ || !kvflash_pager_.is_identity())) {
        if (!kvflash_sync_identity(kv_end)) {
            set_last_error("snapshot_save_thin: kvflash identity sync failed");
            return false;
        }
    }
    snapshot_free(slot);
    auto & snaps = prefix_snapshots_[(size_t)slot];
    if (snaps.size() != shards_.size()) snaps.resize(shards_.size());
    for (size_t i = 0; i < shards_.size(); ++i) {
        if (!snapshot_target_cache_thin(shards_[i].weights, shards_[i].cache,
                                        snapshot_backends_[i], kv_start, kv_end,
                                        snaps[i])) {
            for (size_t j = 0; j <= i; ++j) {
                free_prefix_snapshot(snaps[j]);
            }
            return false;
        }
    }
    return true;
}

bool Qwen35LayerSplitAdapter::apply_restore_chain(
        int thick_slot, const std::vector<int> & thin_slots) {
    if (shards_.empty()) return false;

    if (thick_slot != -1) {
        if (!snapshot_used(thick_slot) || snapshot_is_thin(thick_slot)) {
            return false;
        }
    }
    for (int id : thin_slots) {
        if (!snapshot_used(id) || !snapshot_is_thin(id)) {
            return false;
        }
    }

    for (auto & shard : shards_) {
        reset_recurrent_state(shard.cache);
    }

    for (size_t si = 0; si < shards_.size(); ++si) {
        const PrefixSnapshot * thick_ptr = nullptr;
        if (thick_slot >= 0) {
            thick_ptr = &prefix_snapshots_[(size_t)thick_slot][si];
        }
        std::vector<const PrefixSnapshot *> thin_ptrs;
        thin_ptrs.reserve(thin_slots.size());
        for (int id : thin_slots) {
            thin_ptrs.push_back(&prefix_snapshots_[(size_t)id][si]);
        }
        if (!restore_target_cache_chain(
                thick_ptr,
                thin_ptrs.empty() ? nullptr : thin_ptrs.data(),
                (int)thin_ptrs.size(),
                shards_[si].cache)) {
            std::fprintf(stderr, "[target-split] RESTORE_CHAIN failed shard=%zu: %s\n",
                         si, dflash27b_last_error());
            return false;
        }
    }

    if (thick_slot >= 0) {
        if (snapshot_prefill_logits_.size() == (size_t)PREFIX_SLOTS) {
            prefill_last_logits_ = snapshot_prefill_logits_[(size_t)thick_slot];
        }
        if (kvflash_active()) {
            const int cur_pos = shards_.front().cache.cur_pos;
            if (!kvflash_sync_identity(cur_pos)) return false;
            layer_split_kvflash_restore_history(
                kvflash_history_, kvflash_history_snapshots_, thick_slot, cur_pos);
        }
        if (!restore_draft_features(thick_slot)) return false;
    }

    return true;
}

}  // namespace dflash::common
