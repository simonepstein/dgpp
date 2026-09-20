#pragma once

#include <string>

namespace dgpp {

// HuggingFace hub cache resolution: models live in the canonical hub cache
// inside the user's home directory, and the application treats that as THE
// checkpoint location (deployment decision, M5 exit gates: every node
// carries the full weights there; --checkpoint-dir stays for fixtures and
// staged directories).
//
// The cache layout follows huggingface_hub exactly:
//   <root>/models--<org>--<name>/refs/main       -> current commit of main
//   <root>/models--<org>--<name>/snapshots/<sha>/ -> symlink farm into blobs/
// so resolved snapshots are consumed through ordinary file paths (mmap and
// fopen follow the symlinks; the loaders need no cache awareness of their
// own).
namespace hf {

// The cache root, by huggingface_hub's own precedence: $HF_HUB_CACHE, then
// $HF_HOME/hub, then ~/.cache/huggingface/hub. Empty string when no home
// can be located (the caller errors legibly).
std::string default_cache_root();

// Resolves a model id (org/name) against an explicit cache root. Empty
// return + *error set on any failure; ambiguities (multiple snapshots with
// no ref, a ref whose snapshot is missing) are REFUSED rather than guessed.
std::string model_dir_in_root(const std::string& cache_root,
                              const std::string& model_id,
                              std::string* error);

// model_dir_in_root against default_cache_root().
std::string model_dir(const std::string& model_id, std::string* error);

// Both take "org/name" (refs/main) or "org/name@revision", where the
// revision is a snapshot directory name in full or by unambiguous prefix.

}  // namespace hf
}  // namespace dgpp
