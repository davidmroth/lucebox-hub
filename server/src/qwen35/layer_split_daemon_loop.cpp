// Layer-split daemon loop — uses LayerSplitBackend + generic run_daemon().

#include "layer_split_daemon_loop.h"
#include "common/backend_factory.h"
#include "common/daemon_loop.h"
#include "placement/placement_config.h"

#include <cstdio>
#include <memory>

namespace dflash::common {

int run_layer_split_daemon(const LayerSplitDaemonConfig & cfg) {
    if (!cfg.target_path) {
        std::fprintf(stderr, "[target-split] missing target path\n");
        return 1;
    }
    if (cfg.target_gpus.size() < 2) {
        std::fprintf(stderr, "[target-split] requires >=2 target GPUs\n");
        return 2;
    }

    BackendArgs args;
    args.model_path = cfg.target_path;
    args.draft_path = cfg.load_draft ? cfg.draft_path : nullptr;
    args.stream_fd  = cfg.stream_fd;
    args.fa_window  = cfg.fa_window;
    args.kq_stride_pad = cfg.kq_stride_pad;
    args.draft_ctx_max = cfg.draft_ctx_max;
    args.chunk      = 512;
    args.ddtree_mode = false;

    args.device.max_ctx = cfg.max_ctx;
    args.device.peer_access = cfg.peer_access;
    args.device.layer_split_gpus = cfg.target_gpus;
    args.device.layer_split_weights = cfg.split_weights;
    args.device.layer_split_backends.assign(
        cfg.target_gpus.size(), PlacementBackend::Cuda);
    args.device.backend = PlacementBackend::Cuda;
    args.device.gpu = cfg.target_gpus.front();
    args.draft_device.gpu = cfg.draft_gpu;

    auto backend = create_backend(args);
    if (!backend) {
        std::fprintf(stderr, "[target-split] backend init failed\n");
        return 1;
    }

    DaemonLoopArgs dargs;
    dargs.stream_fd = cfg.stream_fd;
    dargs.chunk     = 512;
    dargs.max_ctx   = cfg.max_ctx;

    const int rc = run_daemon(*backend, dargs);
    backend->shutdown();
    return rc;
}

}  // namespace dflash::common
