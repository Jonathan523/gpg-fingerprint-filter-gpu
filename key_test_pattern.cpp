#include "key_test.hpp"

#include <map>
#include <memory>
#include <sstream>

#include <nvrtc.h>
#define NVRTC_CALL(func, args...) error_wrapper<nvrtcResult>(#func, (func)(args), NVRTC_SUCCESS, nvrtcGetErrorString)

static std::string compile_single_pattern(const std::string &pattern) {
    std::vector<int> tmp_out;
    std::map<char, int> symbol_map;

    int bra = -1, ket = -1;
    const char *p;

    for (char i = '0'; i <= '9'; i++)
        symbol_map[i] = (i - '0') - 100;

    for (char i = 'A'; i <= 'F'; i++)
        symbol_map[i] = (i - 'A' + 10) - 100;

    for (p = pattern.c_str(); *p; p++) {
        switch (*p) {
            case '(': {
                if (bra == -1)
                    bra = tmp_out.size();
                else
                    return "";

                break;
            };
            case ')': {
                if (bra != -1)
                    ket = tmp_out.size();
                else
                    return "";

                break;
            }
            case '{': {
                unsigned long num = strtoul(++p, (char**)&p, 10);

                if (*p != '}' || num == 0)
                    return "";

                int i0, i1;
                if (ket == -1) {
                    i0 = tmp_out.size() - 1;
                    i1 = tmp_out.size();
                } else {
                    i0 = bra;
                    i1 = ket;
                    bra = ket = -1;
                }

                for (unsigned long i = 0; i < num - 1; i++)
                    for (int j = i0; j < i1; j++)
                        tmp_out.push_back(tmp_out[j]);

                break;
            }
            default: {
                if (isdigit(*p) || isalpha(*p)) {
                    char symbol = toupper(*p);

                    if (symbol_map.count(symbol) == 0)
                        symbol_map[symbol] = tmp_out.size();

                    tmp_out.push_back(symbol_map[symbol]);
                } else
                    return "";

                if (ket != -1)
                    bra = ket = -1;
            }
        };
    }

    int offset = 40 - tmp_out.size();
    if ((offset < 0) || (bra != -1 && ket == -1))
        return "";

    std::stringstream ss;
    for (auto i = 0u; i < tmp_out.size(); i++) {
        auto item = tmp_out[i];

        if (item != static_cast<int>(i)) {
            ss << "w[" << i + offset << "] == ";

            if (item < 0)
                ss << tmp_out[i] + 100;
            else
                ss << "w[" << item + offset << "]";

            ss << " && ";
        }
    }

    std::string ret = ss.str();

    if (ret == "")
        return "1";
    else
        return ret.substr(0, ret.size() - 4);
}

// Append nibble-extraction CUDA C++ code for five 32-bit hash words into w[40].
// src_fmt: either "h%d[index]" (reads from global-memory arrays) or a single
// register-variable name like "a","b","c","d","e" (used by the fused kernel).
static void gen_nibble_extract(std::stringstream &ss,
                               const char *const srcs[5]) {
    for (int i = 0; i < 5; i++) {
        ss << "  tmp = " << srcs[i] << ";\n";
        for (int j = 0; j < 7; j++)
            ss << "  w[" << i * 8 + j << "] = (tmp >> " << (7 - j) * 4 << ") & 0x0F;\n";
        ss << "  w[" << i * 8 + 7 << "] = tmp & 0x0F;\n";
    }
}

static std::string compile_patterns(const std::string &input) {
    // --- Parse all sub-patterns and build the combined boolean condition ---
    std::vector<std::string> pattern_codes;
    std::string buffer = input + "|";
    std::string::size_type pos;

    while ((pos = buffer.find("|")) != std::string::npos) {
        auto pattern = buffer.substr(0, pos);
        auto code = compile_single_pattern(pattern);
        if (code == "")
            return "";
        pattern_codes.push_back(code);
        buffer.erase(0, pos + 1);
    }

    std::stringstream cond_ss;
    for (auto &c : pattern_codes)
        cond_ss << "(" << c << ") || ";
    std::string condition = cond_ss.str();
    condition = condition.substr(0, condition.size() - 4); // strip trailing " || "

    // --- Build the full NVRTC source ---
    std::stringstream ss;

    // Common type definitions and helper macros
    ss << "typedef unsigned int u32;\n";
    // Inline function instead of a macro: cleaner dependency information for
    // the compiler and easier to swap in a PTX intrinsic if desired.
    ss << "__device__ __forceinline__ static u32 rotl32(u32 x, int n) { return (x << n) | (x >> (32 - n)); }\n\n";

    // Constant memory for key data — used exclusively by the fused kernel.
    // The host writes one padded SHA-1 block (64 bytes / 16 u32 words) here
    // before launching pattern_check_fused.
    ss << "__constant__ u32 key_chunk0[16];\n\n";

    // SHA-1 device function used by the fused kernel.
    // Split into five explicit loop phases so each phase has a fixed round
    // function and constant — the compiler can inline everything with zero
    // branches after unrolling.  The MAJ function is written as
    // (b&c)|((b|c)&d) rather than (b&c)|(b&d)|(c&d) to save one OR per round.
    // &15 replaces %16 throughout; the read and write of s[j] are split to give
    // the compiler unambiguous data-flow for the rolling message schedule.
    ss << "__forceinline__ __device__ static\n";
    ss << "void sha1_80rounds(u32 s[16], u32 &a, u32 &b, u32 &c, u32 &d, u32 &e) {\n";
    // Rounds 0-15: direct s[i], CH function
    ss << "    #pragma unroll\n";
    ss << "    for (int i = 0; i < 16; i++) {\n";
    ss << "        u32 temp = rotl32(a,5)+(d^(b&(c^d)))+e+0x5A827999U+s[i];\n";
    ss << "        e=d; d=c; c=rotl32(b,30); b=a; a=temp;\n";
    ss << "    }\n";
    // Rounds 16-19: w expansion starts, still CH function
    ss << "    #pragma unroll\n";
    ss << "    for (int i = 16; i < 20; i++) {\n";
    ss << "        int j=i&15;\n";
    ss << "        u32 x=s[(i-3)&15]^s[(i-8)&15]^s[(i-14)&15]^s[j];\n";
    ss << "        u32 wi=rotl32(x,1); s[j]=wi;\n";
    ss << "        u32 temp=rotl32(a,5)+(d^(b&(c^d)))+e+0x5A827999U+wi;\n";
    ss << "        e=d; d=c; c=rotl32(b,30); b=a; a=temp;\n";
    ss << "    }\n";
    // Rounds 20-39: PARITY function
    ss << "    #pragma unroll\n";
    ss << "    for (int i = 20; i < 40; i++) {\n";
    ss << "        int j=i&15;\n";
    ss << "        u32 x=s[(i-3)&15]^s[(i-8)&15]^s[(i-14)&15]^s[j];\n";
    ss << "        u32 wi=rotl32(x,1); s[j]=wi;\n";
    ss << "        u32 temp=rotl32(a,5)+(b^c^d)+e+0x6ED9EBA1U+wi;\n";
    ss << "        e=d; d=c; c=rotl32(b,30); b=a; a=temp;\n";
    ss << "    }\n";
    // Rounds 40-59: MAJ function (simplified)
    ss << "    #pragma unroll\n";
    ss << "    for (int i = 40; i < 60; i++) {\n";
    ss << "        int j=i&15;\n";
    ss << "        u32 x=s[(i-3)&15]^s[(i-8)&15]^s[(i-14)&15]^s[j];\n";
    ss << "        u32 wi=rotl32(x,1); s[j]=wi;\n";
    ss << "        u32 temp=rotl32(a,5)+((b&c)|((b|c)&d))+e+0x8F1BBCDCU+wi;\n";
    ss << "        e=d; d=c; c=rotl32(b,30); b=a; a=temp;\n";
    ss << "    }\n";
    // Rounds 60-79: PARITY function
    ss << "    #pragma unroll\n";
    ss << "    for (int i = 60; i < 80; i++) {\n";
    ss << "        int j=i&15;\n";
    ss << "        u32 x=s[(i-3)&15]^s[(i-8)&15]^s[(i-14)&15]^s[j];\n";
    ss << "        u32 wi=rotl32(x,1); s[j]=wi;\n";
    ss << "        u32 temp=rotl32(a,5)+(b^c^d)+e+0xCA62C1D6U+wi;\n";
    ss << "        e=d; d=c; c=rotl32(b,30); b=a; a=temp;\n";
    ss << "    }\n";
    ss << "}\n\n";

    // --- Kernel 1: pattern_check (multi-chunk path) ---
    // Reads the final SHA-1 state from global-memory arrays h0..h4.
    ss << "extern \"C\" __global__\n";
    ss << "void pattern_check(u32 *result";
    for (int i = 0; i < 5; i++)
        ss << ", const u32* __restrict__ h" << i;
    ss << ") {\n";
    ss << "  u32 index = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ss << "  u32 tmp;\n";
    ss << "  unsigned char w[40];\n";
    {
        const char *srcs[] = { "h0[index]", "h1[index]", "h2[index]", "h3[index]", "h4[index]" };
        gen_nibble_extract(ss, srcs);
    }
    ss << "  if (" << condition << ") *result = index;\n";
    ss << "}\n\n";

    // --- Kernel 2: pattern_check_fused (single-chunk path) ---
    // Computes SHA-1 inline from key_chunk0 constant memory, keeping the full
    // hash state (a,b,c,d,e) in registers.  No global-memory write/read of
    // h[0-4] means ~40 bytes * n_threads less global-memory traffic per call.
    ss << "extern \"C\" __global__\n";
    ss << "void pattern_check_fused(u32 *result, u32 t0) {\n";
    ss << "  constexpr u32 a0 = 0x67452301U;\n";
    ss << "  constexpr u32 b0 = 0xEFCDAB89U;\n";
    ss << "  constexpr u32 c0 = 0x98BADCFEU;\n";
    ss << "  constexpr u32 d0 = 0x10325476U;\n";
    ss << "  constexpr u32 e0 = 0xC3D2E1F0U;\n";
    ss << "  u32 index = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ss << "  u32 s[16];\n";
    ss << "  #pragma unroll\n";
    ss << "  for (int i = 0; i < 16; i++) s[i] = key_chunk0[i];\n";
    // s[1] holds the 4-byte creation timestamp (bytes 4-7 of the OpenPGP
    // fingerprint hash packet, byte-swapped to little-endian u32 by load_key).
    // Each thread tests a different timestamp: t0 is the base and index is the
    // offset subtracted from it, matching the convention in proc_all_chunks.
    ss << "  s[1] = t0 - index;\n";
    ss << "  u32 a = a0, b = b0, c = c0, d = d0, e = e0;\n";
    ss << "  sha1_80rounds(s, a, b, c, d, e);\n";
    ss << "  a += a0; b += b0; c += c0; d += d0; e += e0;\n";
    ss << "  u32 tmp;\n";
    ss << "  unsigned char w[40];\n";
    {
        const char *srcs[] = { "a", "b", "c", "d", "e" };
        gen_nibble_extract(ss, srcs);
    }
    ss << "  if (" << condition << ") *result = index;\n";
    ss << "}\n";

    return ss.str();
}

void CudaManager::load_patterns(const std::string &input) {
    auto cuda_src = compile_patterns(input);

    nvrtcProgram prog;
    NVRTC_CALL(nvrtcCreateProgram, &prog, cuda_src.c_str(), NULL, 0, NULL, NULL);

    int dev_major, dev_minor;
    CU_CALL(cuDeviceGetAttribute, &dev_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_device);
    CU_CALL(cuDeviceGetAttribute, &dev_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_device);

    std::string arch = "-arch=compute_";
    arch += std::to_string(dev_major);
    arch += std::to_string(dev_minor);

    try {
        const char *opts[] = { arch.c_str() };
        NVRTC_CALL(nvrtcCompileProgram, prog, 1, opts);
    } catch (const std::runtime_error &e) {
        size_t log_size;
        NVRTC_CALL(nvrtcGetProgramLogSize, prog, &log_size);
        auto log = std::vector<char>(log_size);
        NVRTC_CALL(nvrtcGetProgramLog, prog, log.data());
        fprintf(stderr, "nvrtcCompileProgram failed:\n%s\n", log.data());
        throw;
    }

    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);

    auto ptx = std::vector<char>(ptx_size);
    NVRTC_CALL(nvrtcGetPTX, prog, ptx.data());

    NVRTC_CALL(nvrtcDestroyProgram, &prog);

    CU_CALL(cuModuleLoadData, &cu_module, ptx.data());
    CU_CALL(cuModuleGetFunction, &cu_kernel, cu_module, "pattern_check");

    CU_CALL(cuMemAlloc, &cu_result, sizeof(uint32_t));
    CU_CALL(cuMemsetD32, cu_result, UINT32_MAX, 1);

    // Retrieve the fused kernel and the constant-memory symbol it reads from.
    CU_CALL(cuModuleGetFunction, &cu_kernel_fused, cu_module, "pattern_check_fused");
    size_t sym_size;
    CU_CALL(cuModuleGetGlobal, &cu_key_chunk0, &sym_size, cu_module, "key_chunk0");
};

void CudaManager::gpu_pattern_check() {
    void *args[] = {&cu_result, h + 0, h + 1, h + 2, h + 3, h + 4};
    CU_CALL(cuLaunchKernel,
            cu_kernel,
            n_block_, 1, 1,
            thread_per_block_, 1, 1,
            0, 0, args, 0);
}

void CudaManager::gpu_pattern_check_fused(u32 t0) {
    void *args[] = {&cu_result, &t0};
    CU_CALL(cuLaunchKernel,
            cu_kernel_fused,
            n_block_, 1, 1,
            thread_per_block_, 1, 1,
            0, 0, args, 0);
}
