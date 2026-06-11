#pragma once
// Pure admission-gate helper for the HTTP server context-length check.
//
// Extracted from http_server.cpp to make the admission decision unit-testable
// without HTTP plumbing.  No includes beyond <cstdlib> required.

/// Returns true iff the request should be rejected with HTTP 400 due to
/// context overflow.
///
/// Semantics:
///   - When compression is NOT enabled: reject if prompt_tokens + max_output
///     exceeds max_ctx (preserves the prior hard-gate behavior).
///   - When compression IS enabled: return false (let the request through).
///     The post-compress effective-size check downstream is the real gate;
///     rejecting at ingress before compression runs would prevent compression
///     from ever rescuing an over-long conversation.
///
/// @param prompt_tokens     Raw prompt token count (before compression).
/// @param max_output        max_tokens from the request.
/// @param max_ctx           Server context window size.
/// @param compression_enabled  True when pFlash/FlowKV compression will run
///                             for this request (i.e. mode != OFF AND drafter
///                             is loaded AND (mode==ALWAYS OR tokens>=threshold)).
inline bool should_reject_oversized(int prompt_tokens, int max_output,
                                    int max_ctx, bool compression_enabled)
{
    if (prompt_tokens + max_output <= max_ctx) {
        return false;  // fits — accept regardless of compression
    }
    // Oversized: only reject if compression cannot help.
    return !compression_enabled;
}
