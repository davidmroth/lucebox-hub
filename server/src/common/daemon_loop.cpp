// Generic daemon command loop implementation.
//
// Handles stdin command parsing and protocol plumbing. All model-specific
// operations are dispatched through the ModelBackend interface.

#include "daemon_loop.h"
#include "daemon_scheduler.h"

#include "sampler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#include <fcntl.h>
#define ssize_t long
#endif

namespace dflash::common {

// ── DaemonIO ────────────────────────────────────────────────────────────

void DaemonIO::emit(int32_t v) const {
    // Call the token callback for non-sentinel tokens.
    if (on_token && v >= 0) {
        if (!on_token(v)) {
            cancelled = true;
            return;
        }
    }

    if (stream_fd < 0) return;

    auto write_i32 = [this](int32_t x) {
#ifndef _WIN32
        ssize_t n = ::write(stream_fd, &x, sizeof(x));
        (void)n;
#else
        _write(stream_fd, &x, sizeof(x));
#endif
    };

    // Tagged framing: [-2, request_id, token_or_sentinel]
    if (stream_tagged && (v >= 0 || v == -1 || v == -4)) {
        write_i32(-2);
        write_i32(request_id);
        write_i32(v);
        return;
    }

    write_i32(v);
}

void DaemonIO::emit_done() const { emit(-1); }

void DaemonIO::emit_continue() const { emit(-4); }

DaemonIO DaemonIO::with_token_callback(const TokenCallback & cb) const {
    DaemonIO out = *this;
    if (!cb) return out;
    TokenCallback existing = out.on_token;
    out.on_token = [existing, cb](int32_t tok) -> bool {
        if (existing && !existing(tok)) return false;
        return cb(tok);
    };
    return out;
}

// Default typed compress: delegates to handle_compress via temp file + DaemonIO collector.
ModelBackend::CompressResult ModelBackend::compress(const CompressRequest & req) {
    CompressResult result;

    if (req.input_ids.empty()) return result;

    // Write input IDs to temp file (handle_compress reads from file)
    const size_t to_write = req.input_ids.size() * sizeof(int32_t);
    std::string tmp_path;
#if defined(_WIN32)
    {
        static std::atomic<unsigned long long> ctr{0};
        const auto uniq =
            std::to_string((unsigned long long)
                std::chrono::steady_clock::now().time_since_epoch().count()) +
            "_" + std::to_string(ctr++);
        std::filesystem::path p =
            std::filesystem::temp_directory_path() / ("pflash_" + uniq + ".bin");
        tmp_path = p.string();
        FILE * f = std::fopen(tmp_path.c_str(), "wb");
        if (!f) return result;
        const size_t w = std::fwrite(req.input_ids.data(), 1, to_write, f);
        std::fclose(f);
        if (w != to_write) { std::remove(tmp_path.c_str()); return result; }
    }
#else
    {
        char tmpl[] = "/tmp/pflash_XXXXXX.bin";
        int tmp_fd = mkstemps(tmpl, 4);
        if (tmp_fd < 0) return result;
        tmp_path = tmpl;
        const char *src = reinterpret_cast<const char *>(req.input_ids.data());
        size_t remaining = to_write;
        while (remaining > 0) {
            ssize_t n = ::write(tmp_fd, src, remaining);
            if (n <= 0) {
                ::close(tmp_fd);
                ::unlink(tmp_path.c_str());
                return result;
            }
            src += n;
            remaining -= (size_t)n;
        }
        ::close(tmp_fd);
    }
#endif

    // Build collecting DaemonIO
    DaemonIO io;
    io.stream_fd = -1;
    io.on_token = [&](int32_t tok) -> bool {
        result.compressed_ids.push_back(tok);
        return true;
    };

    // Build command string for legacy handle_compress
    int keep_x1000 = (int)(req.keep_ratio * 1000.0f);
    std::string cmd = std::string("compress ") + tmp_path + " "
        + std::to_string(keep_x1000) + " " + req.drafter_path;
    if (req.skip_park) cmd += " nopark";

    result.ok = handle_compress(cmd, io) && !result.compressed_ids.empty();
    std::remove(tmp_path.c_str());
    return result;
}

// ── Helpers ─────────────────────────────────────────────────────────────

static bool starts_with(const std::string & s, const std::string & p) {
    return s.size() >= p.size() && std::memcmp(s.data(), p.data(), p.size()) == 0;
}

static bool looks_like_path(const std::string & s) {
    if (s.empty()) return false;
    if (s[0] == '/' || s[0] == '.') return true;
    return s.find('/') != std::string::npos;
}

// Parse "0,1,2" or "-" into thin snapshot slot IDs.
static bool parse_thin_slot_list(const char * thin_str,
                                 std::vector<int> & out_ids,
                                 ModelBackend & backend) {
    out_ids.clear();
    if (!thin_str || thin_str[0] == '\0' || std::strcmp(thin_str, "-") == 0) {
        return true;
    }
    const char * p = thin_str;
    while (*p) {
        char * end = nullptr;
        const long id_l = std::strtol(p, &end, 10);
        if (end == p) {
            std::fprintf(stderr,
                "[snap] RESTORE_CHAIN malformed thin list near '%s'\n", p);
            return false;
        }
        const int id = (int)id_l;
        if (id < 0 || id >= ModelBackend::kMaxSlots ||
            !backend.snapshot_used(id) || !backend.snapshot_is_thin(id)) {
            std::fprintf(stderr, "[snap] RESTORE_CHAIN bad thin slot=%d\n", id);
            return false;
        }
        out_ids.push_back(id);
        if (*end == '\0') break;
        if (*end != ',') {
            std::fprintf(stderr,
                "[snap] RESTORE_CHAIN expected ',' after slot %d, got '%c'\n",
                id, *end);
            return false;
        }
        p = end + 1;
        if (*p == '\0' || *p == ',') {
            std::fprintf(stderr, "[snap] RESTORE_CHAIN empty thin slot entry\n");
            return false;
        }
    }
    return true;
}

// Read a prompt file: raw int32 stream (file size implies token count).
static std::vector<int32_t> read_uncounted_i32(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const auto sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<int32_t> ids(sz / sizeof(int32_t));
    if (!ids.empty()) {
        f.read(reinterpret_cast<char *>(ids.data()),
               (std::streamsize)ids.size() * sizeof(int32_t));
        if (!f) return {};
    }
    return ids;
}

// Read a prompt file with uint32 length prefix + N int32 token IDs.
static std::vector<int32_t> read_counted_i32(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    uint32_t n = 0;
    f.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (!f) return {};
    std::vector<int32_t> ids((size_t)n);
    if (n > 0) {
        f.read(reinterpret_cast<char *>(ids.data()),
               (std::streamsize)ids.size() * sizeof(int32_t));
        if (!f) return {};
    }
    return ids;
}

static bool write_counted_i32(const std::string & path,
                               const std::vector<int32_t> & ids) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t n = (uint32_t)ids.size();
    f.write(reinterpret_cast<const char *>(&n), sizeof(n));
    if (n > 0)
        f.write(reinterpret_cast<const char *>(ids.data()),
                (std::streamsize)ids.size() * sizeof(int32_t));
    return (bool)f;
}

// Parse optional inline-snap suffix: ` snap=<pos>:<slot>`.
static void parse_inline_snap(const std::string & line,
                               int & snap_pos, int & snap_slot) {
    snap_pos  = -1;
    snap_slot = -1;
    size_t p = line.find(" snap=");
    if (p != std::string::npos) {
        int L = -1, S = -1;
        if (std::sscanf(line.c_str() + p + 6, "%d:%d", &L, &S) == 2 &&
            L > 0 && S >= 0 && S < ModelBackend::kMaxSlots) {
            snap_pos  = L;
            snap_slot = S;
        }
    }
}

// ── Main loop ───────────────────────────────────────────────────────────

int run_daemon(ModelBackend & backend, const DaemonLoopArgs & args) {

    // Pipe-backed stdio (server_tools / smoke) is fully buffered by default;
    // force line buffering so ready banners and ack lines arrive promptly.
    setvbuf(stdout, nullptr, _IOLBF, 0);
    setvbuf(stderr, nullptr, _IOLBF, 0);

    DaemonIO io;
    io.stream_fd = args.stream_fd;
    io.stream_tagged = args.stream_tagged;

    const int n_slots = std::max(1, backend.target_cache_slot_count());
    std::vector<LiveRequestState> requests((size_t)n_slots);
    std::deque<PendingQuantum> pending_quanta;
    size_t scheduler_cursor = 0;
    int current_req_id = 0;
    int current_slot = backend.active_target_cache_slot();
    // True when the current stdin line carried an explicit "SLOT k" prefix.
    // Multi-slot RESTORE_CHAIN / SNAPSHOT_THIN require this so restore cannot
    // silently hit whichever live cache SCHED_DRAIN left active.
    bool saw_slot_prefix = false;

    auto sync_io_ids = [&]() {
        io.request_id = current_req_id;
    };
    sync_io_ids();

    auto require_stream_tagged = [&](const char * label) -> bool {
        if (args.stream_tagged) return true;
        std::fprintf(stderr, "[scheduler] %s requires --stream-tagged\n", label);
        std::printf("err stream_tagged_required\n");
        std::fflush(stdout);
        return false;
    };

    auto require_slot_prefix = [&](const char * label) -> bool {
        if (n_slots <= 1 || saw_slot_prefix) return true;
        std::fprintf(stderr,
            "[scheduler] %s requires SLOT <id> when target_cache_slots=%d "
            "(active=%d)\n",
            label, n_slots, backend.active_target_cache_slot());
        std::printf("err slot_required\n");
        std::fflush(stdout);
        return false;
    };

    auto parse_req_slot_prefix = [&](std::string & cmd) {
        // Optional leading "REQ <id>" / "REQUEST <id>" and/or "SLOT <id>".
        for (;;) {
            while (!cmd.empty() && std::isspace((unsigned char)cmd[0])) {
                cmd.erase(cmd.begin());
            }
            if (starts_with(cmd, "REQ ") || starts_with(cmd, "REQUEST ")) {
                const size_t skip = starts_with(cmd, "REQUEST ") ? 8 : 4;
                size_t p = skip;
                while (p < cmd.size() && std::isspace((unsigned char)cmd[p])) ++p;
                current_req_id = std::atoi(cmd.c_str() + (int)p);
                while (p < cmd.size() && !std::isspace((unsigned char)cmd[p])) ++p;
                cmd = cmd.substr(p);
                sync_io_ids();
                continue;
            }
            if (starts_with(cmd, "SLOT ")) {
                size_t p = 5;
                while (p < cmd.size() && std::isspace((unsigned char)cmd[p])) ++p;
                const int sid = std::atoi(cmd.c_str() + (int)p);
                while (p < cmd.size() && !std::isspace((unsigned char)cmd[p])) ++p;
                cmd = cmd.substr(p);
                if (!backend.activate_target_cache_slot(sid)) {
                    std::fprintf(stderr, "[scheduler] bad SLOT %d (slots=%d)\n",
                                 sid, n_slots);
                    std::printf("err bad_slot\n");
                    std::fflush(stdout);
                    cmd.clear();
                    return false;
                }
                current_slot = sid;
                saw_slot_prefix = true;
                continue;
            }
            break;
        }
        return true;
    };

    auto finalize_live_quantum = [&](LiveRequestState & req, int produced,
                                      int requested, bool ok,
                                      int last_emitted_tok) {
        // `requested` is the n_gen asked of this quantum. Fewer tokens than
        // requested, or an EOS at/ending the quantum, means stop — clear
        // remaining so SCHED_DRAIN cannot re-seed on an EOS last-token and
        // burn the leftover max_tokens budget.
        req.emitted += produced;
        const bool hit_eos =
            last_emitted_tok >= 0 && backend.token_is_eos(last_emitted_tok);
        const bool early_stop =
            !ok || produced <= 0 || produced < requested || hit_eos;
        if (early_stop) {
            req.remaining = 0;
        } else {
            req.remaining = std::max(0, req.remaining - produced);
        }
        if (req.remaining <= 0 || io.cancelled) {
            req.active = false;
            req.remaining = 0;
            backend.set_target_cache_slot_busy(req.slot_id, false);
            io.emit_done();
        } else {
            io.emit_continue();
        }
    };

    auto run_one_quantum = [&](PendingQuantum q) -> bool {
        if (q.slot_id < 0 || q.slot_id >= n_slots) return false;
        LiveRequestState & req = requests[(size_t)q.slot_id];
        if (!req.active || req.request_id != q.request_id || req.epoch != q.epoch) {
            return false;
        }
        if (!backend.activate_target_cache_slot(q.slot_id)) return false;
        current_slot = q.slot_id;
        current_req_id = q.request_id;
        sync_io_ids();

        GenerateResult r = backend.continue_generate(q.n_gen, io);
        const int produced = (int)r.tokens.size();
        const int last_tok =
            produced > 0 ? (int)r.tokens.back() : -1;
        finalize_live_quantum(req, produced, q.n_gen, r.ok, last_tok);
        return true;
    };

    backend.print_ready_banner();
    if (n_slots > 1) {
        std::printf("[daemon] target_cache_slots=%d (shared weights, serialized protocol)%s\n",
                    n_slots, args.stream_tagged ? " stream_tagged=1" : "");
    }
    // Always flush: when stdin/stdout are pipes (Python smoke / server_tools),
    // stdout is fully buffered and the ready banner otherwise never arrives.
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "quit" || line == "exit") break;

        saw_slot_prefix = false;
        if (!parse_req_slot_prefix(line)) continue;
        if (line.empty()) continue;

        // ── Multi-request scheduler commands ─────────────────────────

        if (line == "LIST_TARGET_CACHE_SLOTS") {
            std::printf("[daemon] target_cache_slots=%d active=%d\n",
                        n_slots, backend.active_target_cache_slot());
            std::fflush(stdout);
            continue;
        }
        if (line == "LIST_REQUESTS") {
            std::printf("[scheduler] requests=");
            bool first = true;
            for (const auto & req : requests) {
                if (!req.active) continue;
                std::printf("%s%d@slot%d(rem=%d,q=%d)",
                            first ? "" : ",",
                            req.request_id, req.slot_id, req.remaining, req.quantum);
                first = false;
            }
            std::printf("\n");
            std::fflush(stdout);
            continue;
        }
        if (starts_with(line, "CANCEL") ||
            (starts_with(line, "REQ ") && line.find(" CANCEL") != std::string::npos)) {
            int cancel_id = current_req_id;
            if (starts_with(line, "CANCEL ")) {
                cancel_id = std::atoi(line.c_str() + 7);
            }
            for (auto & req : requests) {
                if (req.active && req.request_id == cancel_id) {
                    req.active = false;
                    req.remaining = 0;
                    req.epoch += 1;
                    backend.set_target_cache_slot_busy(req.slot_id, false);
                    std::printf("[scheduler] cancelled req=%d slot=%d\n",
                                cancel_id, req.slot_id);
                    std::fflush(stdout);
                }
            }
            continue;
        }
        if (starts_with(line, "START ") || line == "CONTINUE" || starts_with(line, "CONT ")) {
            if (!require_stream_tagged(starts_with(line, "START") ? "START" : "CONTINUE")) {
                continue;
            }
            if (starts_with(line, "START ")) {
                // START <prompt_path> <total_gen> <quantum>
                char path[1024];
                int total_gen = 0, quantum = 0;
                if (std::sscanf(line.c_str() + 6, "%1023s %d %d",
                                path, &total_gen, &quantum) < 2 ||
                    total_gen <= 0) {
                    std::printf("err bad_args\n");
                    std::fflush(stdout);
                    continue;
                }
                if (quantum <= 0) quantum = total_gen;
                // Admit onto current_slot if free, else first free slot.
                int slot = current_slot;
                if (slot < 0 || slot >= n_slots || requests[(size_t)slot].active) {
                    slot = -1;
                    for (int s = 0; s < n_slots; ++s) {
                        if (!requests[(size_t)s].active) { slot = s; break; }
                    }
                }
                if (slot < 0) {
                    std::fprintf(stderr, "[scheduler] no free target cache slots\n");
                    std::printf("err no_free_slot\n");
                    std::fflush(stdout);
                    continue;
                }
                if (!backend.activate_target_cache_slot(slot)) {
                    std::printf("err bad_slot\n");
                    std::fflush(stdout);
                    continue;
                }
                current_slot = slot;
                // Prefill + first quantum via normal generate on empty prompt path.
                // Read prompt ids (length-prefixed or raw handled by generate file path).
                // Use bare-prompt generate through restore_and_generate isn't right —
                // use standard GenerateRequest via a small local helper: load file + generate.
                std::ifstream in(path, std::ios::binary);
                if (!in) {
                    std::printf("err bad_prompt\n");
                    std::fflush(stdout);
                    continue;
                }
                in.seekg(0, std::ios::end);
                const std::streamoff nbytes = in.tellg();
                in.seekg(0, std::ios::beg);
                if (nbytes <= 0 || (nbytes % 4) != 0) {
                    std::printf("err bad_prompt\n");
                    std::fflush(stdout);
                    continue;
                }
                std::vector<int32_t> ids((size_t)nbytes / 4);
                in.read(reinterpret_cast<char *>(ids.data()), nbytes);
                GenerateRequest greq;
                greq.prompt = std::move(ids);
                greq.n_gen = quantum;
                greq.stream = true;
                sync_io_ids();
                GenerateResult gr = backend.generate(greq, io);
                LiveRequestState & req = requests[(size_t)slot];
                req.active = true;
                req.request_id = current_req_id;
                req.slot_id = slot;
                req.quantum = quantum;
                req.emitted = (int)gr.tokens.size();
                req.remaining = std::max(0, total_gen - req.emitted);
                req.epoch += 1;
                backend.set_target_cache_slot_busy(slot, true);
                {
                    const int produced = (int)gr.tokens.size();
                    const int last_tok =
                        produced > 0 ? (int)gr.tokens.back() : -1;
                    const bool hit_eos =
                        last_tok >= 0 && backend.token_is_eos(last_tok);
                    const bool early_stop =
                        !gr.ok || produced <= 0 || produced < quantum || hit_eos;
                    if (early_stop) {
                        req.remaining = 0;
                    }
                    if (req.remaining <= 0 || io.cancelled) {
                        req.active = false;
                        req.remaining = 0;
                        backend.set_target_cache_slot_busy(slot, false);
                        io.emit_done();
                    } else {
                        io.emit_continue();
                    }
                }
                std::printf("ok START req=%d slot=%d emitted=%d remaining=%d\n",
                            current_req_id, slot, req.emitted, req.remaining);
                std::fflush(stdout);
                continue;
            }
            // CONTINUE / CONT — keep decoding on current slot's active request
            int slot = current_slot;
            if (slot < 0 || slot >= n_slots || !requests[(size_t)slot].active) {
                std::printf("err no_active_request\n");
                std::fflush(stdout);
                continue;
            }
            LiveRequestState & req = requests[(size_t)slot];
            const int q = std::max(1, std::min(req.quantum > 0 ? req.quantum : req.remaining,
                                               req.remaining));
            PendingQuantum pq{req.request_id, req.slot_id, req.epoch, q};
            run_one_quantum(pq);
            continue;
        }
        if (line == "SCHED_STEP") {
            if (!require_stream_tagged("SCHED_STEP")) continue;
            if (pending_quanta.empty()) {
                enqueue_next_quantum(requests, pending_quanta, scheduler_cursor);
            }
            if (pending_quanta.empty()) {
                std::printf("ok SCHED_STEP idle\n");
                std::fflush(stdout);
                continue;
            }
            PendingQuantum q = pending_quanta.front();
            pending_quanta.pop_front();
            run_one_quantum(q);
            enqueue_next_quantum(requests, pending_quanta, scheduler_cursor);
            std::printf("ok SCHED_STEP\n");
            std::fflush(stdout);
            continue;
        }
        if (line == "SCHED_DRAIN") {
            if (!require_stream_tagged("SCHED_DRAIN")) continue;
            int steps = 0;
            // Bound empty-continue spins: a zero-token CONTINUE must not
            // monopolize stdin (gen_len=0 / stalled continue_generate).
            int zero_token_streak = 0;
            while (steps < 65536) {
                if (pending_quanta.empty()) {
                    if (!enqueue_next_quantum(requests, pending_quanta, scheduler_cursor)) {
                        break;
                    }
                }
                PendingQuantum q = pending_quanta.front();
                pending_quanta.pop_front();
                LiveRequestState & before = requests[(size_t)std::max(0, q.slot_id)];
                const int rem_before = before.remaining;
                const bool ran = run_one_quantum(q);
                ++steps;
                if (ran) {
                    LiveRequestState & after = requests[(size_t)std::max(0, q.slot_id)];
                    if (after.active && after.remaining == rem_before) {
                        ++zero_token_streak;
                        if (zero_token_streak >= 8) {
                            std::fprintf(stderr,
                                "[scheduler] SCHED_DRAIN abort: zero-token streak "
                                "slot=%d remaining=%d\n",
                                q.slot_id, after.remaining);
                            after.active = false;
                            after.remaining = 0;
                            backend.set_target_cache_slot_busy(q.slot_id, false);
                            current_req_id = after.request_id;
                            sync_io_ids();
                            io.emit_done();
                            zero_token_streak = 0;
                        }
                    } else {
                        zero_token_streak = 0;
                    }
                } else {
                    zero_token_streak = 0;
                }
                enqueue_next_quantum(requests, pending_quanta, scheduler_cursor);
            }
            std::printf("ok SCHED_DRAIN steps=%d\n", steps);
            std::fflush(stdout);
            continue;
        }

        // ── Lifecycle commands (no sampler tail) ─────────────────────

        if (starts_with(line, "park")) {
            std::string what;
            if (line.size() > 5) what = line.substr(5);
            backend.park(what);
            io.emit(-1);
            continue;
        }
        if (starts_with(line, "unpark")) {
            std::string what;
            if (line.size() > 7) what = line.substr(7);
            backend.unpark(what);
            io.emit(-1);
            continue;
        }
        if (line == "free drafter" || line == "drafter free") {
            backend.free_drafter();
            io.emit(-1);
            continue;
        }
        if (starts_with(line, "compress ")) {
            backend.handle_compress(line, io);
            continue;
        }

        // ── Snapshot commands ────────────────────────────────────────

        if (line == "LIST_SLOTS") {
            std::printf("[snap] slots=");
            bool first = true;
            for (int i = 0; i < ModelBackend::kMaxSlots; ++i) {
                if (backend.snapshot_used(i)) {
                    std::printf("%s%d", first ? "" : ",", i);
                    first = false;
                }
            }
            std::printf("\n"); std::fflush(stdout);
            continue;
        }
        if (starts_with(line, "FREE_SNAPSHOT ")) {
            int slot = std::atoi(line.c_str() + 14);
            if (slot >= 0 && slot < ModelBackend::kMaxSlots) {
                backend.snapshot_free(slot);
            }
            std::printf("[snap] freed slot=%d\n", slot);
            std::fflush(stdout);
            continue;
        }

        // SNAPSHOT_THIN before SNAPSHOT; arch hook before generate commands.
        // Tool pins copy from the *active* live TargetCache — when N>1 the
        // caller must name which live slot via "SLOT k SNAPSHOT_THIN …".
        if (starts_with(line, "SNAPSHOT_THIN ") &&
            !require_slot_prefix("SNAPSHOT_THIN")) {
            io.emit(-1);
            continue;
        }
        if (backend.try_handle_command(line, io)) {
            continue;
        }
        if (starts_with(line, "SNAPSHOT ")) {
            if (!require_slot_prefix("SNAPSHOT")) {
                continue;
            }
            int slot = std::atoi(line.c_str() + 9);
            backend.snapshot_save(slot);
            std::printf("[snap] inline slot=%d cur_pos=%d live=%d\n",
                        slot,
                        backend.snapshot_used(slot) ? backend.snapshot_cur_pos(slot) : -1,
                        backend.active_target_cache_slot());
            std::fflush(stdout);
            continue;
        }

        // ── Generate commands (sampler tail honored) ─────────────────

        SamplerCfg sampler{};
        const bool have_sampler = parse_sampler_token(line, sampler);
        const bool do_sample    = have_sampler && sampler.needs_logit_processing();

        if (backend.is_target_parked()) {
            std::fprintf(stderr,
                "[daemon] target is parked; expected unpark before generate\n");
            std::printf("err target_parked\n"); std::fflush(stdout);
            io.emit(-1);
            continue;
        }

        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        // ── "generate <in> <n_gen> <out>" (file-based, legacy) ───────
        if (cmd == "generate") {
            std::string in_path, out_path;
            int n_gen = 0;
            iss >> in_path >> n_gen >> out_path;
            if (in_path.empty() || out_path.empty() || n_gen <= 0) {
                std::fprintf(stderr, "[daemon] bad: %s\n", line.c_str());
                std::printf("err bad_args\n"); std::fflush(stdout);
                continue;
            }
            auto prompt = read_counted_i32(in_path);
            if (prompt.empty()) {
                std::printf("err empty_prompt\n"); std::fflush(stdout);
                continue;
            }
            GenerateRequest req;
            req.prompt    = std::move(prompt);
            req.n_gen     = n_gen;
            req.sampler   = sampler;
            req.do_sample = do_sample;
            req.stream    = false;

            auto result = backend.generate(req, io);
            if (!result.ok) {
                std::printf("err %s\n", result.error.c_str());
                std::fflush(stdout);
                continue;
            }
            if (!write_counted_i32(out_path, result.tokens)) {
                std::printf("err write_out\n"); std::fflush(stdout);
                continue;
            }
            std::printf("ok N=%d gen=%zu prefill_s=%.3f decode_s=%.3f decode_tok_s=%.1f out=%s\n",
                        (int)req.prompt.size(), result.tokens.size(),
                        result.prefill_s, result.decode_s,
                        result.tokens.size() / std::max(1e-9, result.decode_s),
                        out_path.c_str());
            std::fflush(stdout);
            continue;
        }

        // ── RESTORE_CHAIN <thick> <thin_list> <prompt_path> <n_gen> [quantum] ─
        // Optional trailing <quantum>: when present with --stream-tagged, restore
        // + first quantum only, then register the live slot for CONTINUE /
        // SCHED_* (warm multi-request path). Omitting quantum keeps the legacy
        // blocking full generate.
        if (cmd == "RESTORE_CHAIN") {
            // Require SLOT when N>1 *before* busy — bare RESTORE_CHAIN must not
            // hit the wrong live cache. Busy check then uses that activated slot.
            if (!require_slot_prefix("RESTORE_CHAIN")) {
                io.emit(-1);
                continue;
            }
            const int restore_live_slot = backend.active_target_cache_slot();
            if (backend.target_cache_slot_busy(restore_live_slot)) {
                std::fprintf(stderr,
                    "[scheduler] RESTORE_CHAIN refused: slot %d busy\n",
                    restore_live_slot);
                std::printf("err slot_busy\n");
                std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            int thick_slot = -2;
            std::string thin_str;
            std::string in_path;
            int n_gen = 0;
            int quantum = 0;
            iss >> thick_slot >> thin_str >> in_path >> n_gen;
            if (!(iss >> quantum)) {
                quantum = 0;
            }
            if (thick_slot < -1 || thick_slot >= ModelBackend::kMaxSlots ||
                thin_str.empty() || in_path.empty() || n_gen <= 0 ||
                quantum < 0) {
                std::fprintf(stderr, "[snap] RESTORE_CHAIN bad args: %s\n",
                             line.c_str());
                std::printf("err bad_args\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            if (quantum > 0 && !args.stream_tagged) {
                std::printf("err stream_tagged_required\n");
                std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            if (thick_slot != -1 &&
                (!backend.snapshot_used(thick_slot) ||
                 backend.snapshot_is_thin(thick_slot))) {
                std::fprintf(stderr, "[snap] RESTORE_CHAIN bad thick slot=%d\n",
                             thick_slot);
                std::printf("err bad_slot\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            std::vector<int> thin_ids;
            if (!parse_thin_slot_list(thin_str.c_str(), thin_ids, backend)) {
                std::printf("err bad_thin_list\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            auto prompt = read_uncounted_i32(in_path);
            if (prompt.empty()) {
                std::printf("err empty_prompt\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }

            int snap_pos = -1, snap_slot = -1;
            parse_inline_snap(line, snap_pos, snap_slot);

            const bool admit = quantum > 0;
            const int first_gen = admit ? std::min(quantum, n_gen) : n_gen;

            GenerateRequest req;
            req.prompt    = std::move(prompt);
            req.n_gen     = first_gen;
            req.sampler   = sampler;
            req.do_sample = do_sample;
            req.stream    = true;
            req.snap_pos  = snap_pos;
            req.snap_slot = snap_slot;

            sync_io_ids();
            auto result = backend.restore_chain_and_generate(
                thick_slot, thin_ids, req, io);
            if (!result.ok) {
                std::printf("err %s\n", result.error.c_str());
                std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            // Success ack is printed inside LayerSplitBackend::restore_chain_and_generate_impl
            // immediately after the [prefill] line so the protocol cannot hang
            // behind post-restore teardown.

            if (admit) {
                if (restore_live_slot < 0 || restore_live_slot >= n_slots) {
                    std::printf("err bad_slot\n");
                    std::fflush(stdout);
                    io.emit(-1);
                    continue;
                }
                LiveRequestState & lreq = requests[(size_t)restore_live_slot];
                lreq.active = true;
                lreq.request_id = current_req_id;
                lreq.slot_id = restore_live_slot;
                lreq.quantum = quantum;
                lreq.emitted = 0;
                // Remaining budget after this quantum (may be cleared on early stop).
                lreq.remaining = std::max(0, n_gen - (int)result.tokens.size());
                lreq.epoch += 1;
                current_slot = restore_live_slot;
                backend.set_target_cache_slot_busy(restore_live_slot, true);
                // Pass remaining_delta semantics via a dedicated admit finalize:
                // emitted was 0; produced already subtracted from remaining above;
                // only force remaining=0 on early stop, then emit CONTINUE/DONE.
                {
                    const int produced = (int)result.tokens.size();
                    const int last_tok =
                        produced > 0 ? (int)result.tokens.back() : -1;
                    lreq.emitted = produced;
                    const bool hit_eos =
                        last_tok >= 0 && backend.token_is_eos(last_tok);
                    const bool early_stop =
                        !result.ok || produced <= 0 || produced < first_gen ||
                        hit_eos;
                    if (early_stop) {
                        lreq.remaining = 0;
                    }
                    if (lreq.remaining <= 0 || io.cancelled) {
                        lreq.active = false;
                        lreq.remaining = 0;
                        backend.set_target_cache_slot_busy(restore_live_slot, false);
                        io.emit_done();
                    } else {
                        io.emit_continue();
                    }
                }
                std::printf(
                    "ok RESTORE_CHAIN_ADMIT req=%d slot=%d emitted=%d remaining=%d "
                    "total_gen=%d quantum=%d\n",
                    current_req_id, restore_live_slot, lreq.emitted, lreq.remaining,
                    n_gen, quantum);
                std::fflush(stdout);
            }
            continue;
        }

        // ── RESTORE <slot> <prompt_path> <n_gen> ─────────────────────
        if (cmd == "RESTORE") {
            int slot = -1;
            std::string in_path;
            int n_gen = 0;
            iss >> slot >> in_path >> n_gen;
            if (slot < 0 || slot >= ModelBackend::kMaxSlots ||
                in_path.empty() || n_gen <= 0) {
                std::fprintf(stderr, "[snap] RESTORE bad args: %s\n", line.c_str());
                std::printf("err bad_args\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            if (!backend.snapshot_used(slot)) {
                std::fprintf(stderr, "[snap] RESTORE slot=%d not populated\n", slot);
                std::printf("err empty_slot\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            auto prompt = read_uncounted_i32(in_path);
            if (prompt.empty()) {
                std::printf("err empty_prompt\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }

            int snap_pos = -1, snap_slot = -1;
            parse_inline_snap(line, snap_pos, snap_slot);

            GenerateRequest req;
            req.prompt    = std::move(prompt);
            req.n_gen     = n_gen;
            req.sampler   = sampler;
            req.do_sample = do_sample;
            req.stream    = true;
            req.snap_pos  = snap_pos;
            req.snap_slot = snap_slot;

            auto result = backend.restore_and_generate(slot, req, io);
            if (!result.ok) {
                std::printf("err %s\n", result.error.c_str());
                std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            std::printf("ok N=%d gen=%zu prefix_len=%d (RESTORE slot=%d) stream_fd=%d\n",
                        (int)req.prompt.size(), result.tokens.size(),
                        backend.snapshot_cur_pos(slot), slot, io.stream_fd);
            std::fflush(stdout);
            continue;
        }

        // ── Bare prompt: "<path> <n_gen> [snap=L:S]" ─────────────────
        if (looks_like_path(cmd)) {
            const std::string & in_path = cmd;
            int n_gen = 0;
            iss >> n_gen;
            if (n_gen <= 0) {
                std::fprintf(stderr, "[daemon] bad: %s\n", line.c_str());
                std::printf("err bad_args\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            if (io.stream_fd < 0) {
                std::fprintf(stderr, "[daemon] bare-prompt requires --stream-fd\n");
                std::printf("err no_stream_fd\n"); std::fflush(stdout);
                continue;
            }
            auto prompt = read_uncounted_i32(in_path);
            if (prompt.empty()) {
                std::printf("err empty_prompt\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }

            int snap_pos = -1, snap_slot = -1;
            parse_inline_snap(line, snap_pos, snap_slot);

            GenerateRequest req;
            req.prompt    = std::move(prompt);
            req.n_gen     = n_gen;
            req.sampler   = sampler;
            req.do_sample = do_sample;
            req.stream    = true;
            req.snap_pos  = snap_pos;
            req.snap_slot = snap_slot;

            auto result = backend.generate(req, io);
            if (!result.ok) {
                io.emit(-1);
                std::printf("err %s\n", result.error.c_str());
                std::fflush(stdout);
                continue;
            }
            std::printf("ok N=%d gen=%zu prefill_s=%.3f decode_s=%.3f decode_tok_s=%.1f stream_fd=%d\n",
                        (int)req.prompt.size(), result.tokens.size(),
                        result.prefill_s, result.decode_s,
                        result.tokens.size() / std::max(1e-9, result.decode_s),
                        io.stream_fd);
            std::fflush(stdout);
            continue;
        }

        // ── Unknown command ──────────────────────────────────────────
        std::fprintf(stderr, "[daemon] unknown cmd: %s\n", line.c_str());
        std::printf("err unknown_command\n"); std::fflush(stdout);
        io.emit(-1);
    }

    backend.shutdown();
    return 0;
}

}  // namespace dflash::common
