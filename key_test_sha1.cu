#include "key_test.hpp"

// DSA, no key should be longer
__constant__ static u32 chunk_buffer[208];

// SHA-1 initial hash values (IV), shared by both proc_chunks and proc_all_chunks.
static constexpr u32 SHA1_A0 = 0x67452301U;
static constexpr u32 SHA1_B0 = 0xEFCDAB89U;
static constexpr u32 SHA1_C0 = 0x98BADCFEU;
static constexpr u32 SHA1_D0 = 0x10325476U;
static constexpr u32 SHA1_E0 = 0xC3D2E1F0U;

// ---------------------------------------------------------------------------
// Hardware rotate: __funnelshift_l maps to a single SHF.L.WRAP instruction
// in PTX/SASS, removing any ambiguity in recognising the shift+OR idiom.
// ---------------------------------------------------------------------------
__device__ __forceinline__ static u32 rotl32(u32 x, int n) {
    return __funnelshift_l(x, x, n);
}

// ---------------------------------------------------------------------------
// SHA-1 round macros — register-rotation approach.
//
// A standard SHA-1 round assigns 5 registers:
//   e=d; d=c; c=rotl(b,30); b=a; a=temp
//
// Instead we use two operations per round:
//   5th arg (e_role) accumulates the new hash value
//   2nd arg (b_role) is rotated left 30 in place
//
// The caller rotates which C++ variable occupies each role on every call.
// The mapping cycles with period 5:
//   round%5=0  SHA1_*(a,b,c,d,e, k, w)
//   round%5=1  SHA1_*(e,a,b,c,d, k, w)
//   round%5=2  SHA1_*(d,e,a,b,c, k, w)
//   round%5=3  SHA1_*(c,d,e,a,b, k, w)
//   round%5=4  SHA1_*(b,c,d,e,a, k, w)
//
// After 80 rounds (= 16 × 5 full cycles) SHA-1 (A,B,C,D,E) = (a,b,c,d,e).
// ---------------------------------------------------------------------------
#define SHA1_F_CH(b,c,d)  ((d) ^ ((b) & ((c) ^ (d))))
#define SHA1_F_MAJ(b,c,d) (((b) & (c)) | (((b) | (c)) & (d)))
#define SHA1_F_PAR(b,c,d) ((b) ^ (c) ^ (d))

#define SHA1_CH( a,b,c,d,e,k,w) (e) += rotl32((a),5) + SHA1_F_CH( (b),(c),(d)) + (k) + (w); (b) = rotl32((b),30);
#define SHA1_PAR(a,b,c,d,e,k,w) (e) += rotl32((a),5) + SHA1_F_PAR((b),(c),(d)) + (k) + (w); (b) = rotl32((b),30);
#define SHA1_MAJ(a,b,c,d,e,k,w) (e) += rotl32((a),5) + SHA1_F_MAJ((b),(c),(d)) + (k) + (w); (b) = rotl32((b),30);

// Rolling message-schedule expansion (j = i & 15 for round i >= 16).
// Index arithmetic: (j+13)&15 = (i-3)&15, (j+8)&15 = (i-8)&15,
//                   (j+2)&15  = (i-14)&15, j = (i-16)&15.
#define SHA1_EXPAND(W,j) (W)[j] = rotl32((W)[(j+13)&15]^(W)[(j+8)&15]^(W)[(j+2)&15]^(W)[j],1)

// ---------------------------------------------------------------------------
// 80-round SHA-1 compression.  All round functions, key constants, and
// message-word indices are statically known — no loop control, no runtime
// conditionals.  The register-rotation macros above eliminate the 5-way
// variable shuffle that a naive loop produces.
// ---------------------------------------------------------------------------
__forceinline__ __device__ static
void sha1_80rounds(u32 w[16], u32 &a, u32 &b, u32 &c, u32 &d, u32 &e)
{
    // Rounds 0-15: direct w[0..15], CH function, K = 0x5A827999
    SHA1_CH( a,b,c,d,e, 0x5A827999U, w[ 0])
    SHA1_CH( e,a,b,c,d, 0x5A827999U, w[ 1])
    SHA1_CH( d,e,a,b,c, 0x5A827999U, w[ 2])
    SHA1_CH( c,d,e,a,b, 0x5A827999U, w[ 3])
    SHA1_CH( b,c,d,e,a, 0x5A827999U, w[ 4])
    SHA1_CH( a,b,c,d,e, 0x5A827999U, w[ 5])
    SHA1_CH( e,a,b,c,d, 0x5A827999U, w[ 6])
    SHA1_CH( d,e,a,b,c, 0x5A827999U, w[ 7])
    SHA1_CH( c,d,e,a,b, 0x5A827999U, w[ 8])
    SHA1_CH( b,c,d,e,a, 0x5A827999U, w[ 9])
    SHA1_CH( a,b,c,d,e, 0x5A827999U, w[10])
    SHA1_CH( e,a,b,c,d, 0x5A827999U, w[11])
    SHA1_CH( d,e,a,b,c, 0x5A827999U, w[12])
    SHA1_CH( c,d,e,a,b, 0x5A827999U, w[13])
    SHA1_CH( b,c,d,e,a, 0x5A827999U, w[14])
    SHA1_CH( a,b,c,d,e, 0x5A827999U, w[15])
    // Rounds 16-19: expansion begins, CH, K = 0x5A827999
    SHA1_EXPAND(w, 0); SHA1_CH( e,a,b,c,d, 0x5A827999U, w[ 0])
    SHA1_EXPAND(w, 1); SHA1_CH( d,e,a,b,c, 0x5A827999U, w[ 1])
    SHA1_EXPAND(w, 2); SHA1_CH( c,d,e,a,b, 0x5A827999U, w[ 2])
    SHA1_EXPAND(w, 3); SHA1_CH( b,c,d,e,a, 0x5A827999U, w[ 3])
    // Rounds 20-39: PAR function, K = 0x6ED9EBA1
    SHA1_EXPAND(w, 4); SHA1_PAR(a,b,c,d,e, 0x6ED9EBA1U, w[ 4])
    SHA1_EXPAND(w, 5); SHA1_PAR(e,a,b,c,d, 0x6ED9EBA1U, w[ 5])
    SHA1_EXPAND(w, 6); SHA1_PAR(d,e,a,b,c, 0x6ED9EBA1U, w[ 6])
    SHA1_EXPAND(w, 7); SHA1_PAR(c,d,e,a,b, 0x6ED9EBA1U, w[ 7])
    SHA1_EXPAND(w, 8); SHA1_PAR(b,c,d,e,a, 0x6ED9EBA1U, w[ 8])
    SHA1_EXPAND(w, 9); SHA1_PAR(a,b,c,d,e, 0x6ED9EBA1U, w[ 9])
    SHA1_EXPAND(w,10); SHA1_PAR(e,a,b,c,d, 0x6ED9EBA1U, w[10])
    SHA1_EXPAND(w,11); SHA1_PAR(d,e,a,b,c, 0x6ED9EBA1U, w[11])
    SHA1_EXPAND(w,12); SHA1_PAR(c,d,e,a,b, 0x6ED9EBA1U, w[12])
    SHA1_EXPAND(w,13); SHA1_PAR(b,c,d,e,a, 0x6ED9EBA1U, w[13])
    SHA1_EXPAND(w,14); SHA1_PAR(a,b,c,d,e, 0x6ED9EBA1U, w[14])
    SHA1_EXPAND(w,15); SHA1_PAR(e,a,b,c,d, 0x6ED9EBA1U, w[15])
    SHA1_EXPAND(w, 0); SHA1_PAR(d,e,a,b,c, 0x6ED9EBA1U, w[ 0])
    SHA1_EXPAND(w, 1); SHA1_PAR(c,d,e,a,b, 0x6ED9EBA1U, w[ 1])
    SHA1_EXPAND(w, 2); SHA1_PAR(b,c,d,e,a, 0x6ED9EBA1U, w[ 2])
    SHA1_EXPAND(w, 3); SHA1_PAR(a,b,c,d,e, 0x6ED9EBA1U, w[ 3])
    SHA1_EXPAND(w, 4); SHA1_PAR(e,a,b,c,d, 0x6ED9EBA1U, w[ 4])
    SHA1_EXPAND(w, 5); SHA1_PAR(d,e,a,b,c, 0x6ED9EBA1U, w[ 5])
    SHA1_EXPAND(w, 6); SHA1_PAR(c,d,e,a,b, 0x6ED9EBA1U, w[ 6])
    SHA1_EXPAND(w, 7); SHA1_PAR(b,c,d,e,a, 0x6ED9EBA1U, w[ 7])
    // Rounds 40-59: MAJ function, K = 0x8F1BBCDC
    SHA1_EXPAND(w, 8); SHA1_MAJ(a,b,c,d,e, 0x8F1BBCDCU, w[ 8])
    SHA1_EXPAND(w, 9); SHA1_MAJ(e,a,b,c,d, 0x8F1BBCDCU, w[ 9])
    SHA1_EXPAND(w,10); SHA1_MAJ(d,e,a,b,c, 0x8F1BBCDCU, w[10])
    SHA1_EXPAND(w,11); SHA1_MAJ(c,d,e,a,b, 0x8F1BBCDCU, w[11])
    SHA1_EXPAND(w,12); SHA1_MAJ(b,c,d,e,a, 0x8F1BBCDCU, w[12])
    SHA1_EXPAND(w,13); SHA1_MAJ(a,b,c,d,e, 0x8F1BBCDCU, w[13])
    SHA1_EXPAND(w,14); SHA1_MAJ(e,a,b,c,d, 0x8F1BBCDCU, w[14])
    SHA1_EXPAND(w,15); SHA1_MAJ(d,e,a,b,c, 0x8F1BBCDCU, w[15])
    SHA1_EXPAND(w, 0); SHA1_MAJ(c,d,e,a,b, 0x8F1BBCDCU, w[ 0])
    SHA1_EXPAND(w, 1); SHA1_MAJ(b,c,d,e,a, 0x8F1BBCDCU, w[ 1])
    SHA1_EXPAND(w, 2); SHA1_MAJ(a,b,c,d,e, 0x8F1BBCDCU, w[ 2])
    SHA1_EXPAND(w, 3); SHA1_MAJ(e,a,b,c,d, 0x8F1BBCDCU, w[ 3])
    SHA1_EXPAND(w, 4); SHA1_MAJ(d,e,a,b,c, 0x8F1BBCDCU, w[ 4])
    SHA1_EXPAND(w, 5); SHA1_MAJ(c,d,e,a,b, 0x8F1BBCDCU, w[ 5])
    SHA1_EXPAND(w, 6); SHA1_MAJ(b,c,d,e,a, 0x8F1BBCDCU, w[ 6])
    SHA1_EXPAND(w, 7); SHA1_MAJ(a,b,c,d,e, 0x8F1BBCDCU, w[ 7])
    SHA1_EXPAND(w, 8); SHA1_MAJ(e,a,b,c,d, 0x8F1BBCDCU, w[ 8])
    SHA1_EXPAND(w, 9); SHA1_MAJ(d,e,a,b,c, 0x8F1BBCDCU, w[ 9])
    SHA1_EXPAND(w,10); SHA1_MAJ(c,d,e,a,b, 0x8F1BBCDCU, w[10])
    SHA1_EXPAND(w,11); SHA1_MAJ(b,c,d,e,a, 0x8F1BBCDCU, w[11])
    // Rounds 60-79: PAR function, K = 0xCA62C1D6
    SHA1_EXPAND(w,12); SHA1_PAR(a,b,c,d,e, 0xCA62C1D6U, w[12])
    SHA1_EXPAND(w,13); SHA1_PAR(e,a,b,c,d, 0xCA62C1D6U, w[13])
    SHA1_EXPAND(w,14); SHA1_PAR(d,e,a,b,c, 0xCA62C1D6U, w[14])
    SHA1_EXPAND(w,15); SHA1_PAR(c,d,e,a,b, 0xCA62C1D6U, w[15])
    SHA1_EXPAND(w, 0); SHA1_PAR(b,c,d,e,a, 0xCA62C1D6U, w[ 0])
    SHA1_EXPAND(w, 1); SHA1_PAR(a,b,c,d,e, 0xCA62C1D6U, w[ 1])
    SHA1_EXPAND(w, 2); SHA1_PAR(e,a,b,c,d, 0xCA62C1D6U, w[ 2])
    SHA1_EXPAND(w, 3); SHA1_PAR(d,e,a,b,c, 0xCA62C1D6U, w[ 3])
    SHA1_EXPAND(w, 4); SHA1_PAR(c,d,e,a,b, 0xCA62C1D6U, w[ 4])
    SHA1_EXPAND(w, 5); SHA1_PAR(b,c,d,e,a, 0xCA62C1D6U, w[ 5])
    SHA1_EXPAND(w, 6); SHA1_PAR(a,b,c,d,e, 0xCA62C1D6U, w[ 6])
    SHA1_EXPAND(w, 7); SHA1_PAR(e,a,b,c,d, 0xCA62C1D6U, w[ 7])
    SHA1_EXPAND(w, 8); SHA1_PAR(d,e,a,b,c, 0xCA62C1D6U, w[ 8])
    SHA1_EXPAND(w, 9); SHA1_PAR(c,d,e,a,b, 0xCA62C1D6U, w[ 9])
    SHA1_EXPAND(w,10); SHA1_PAR(b,c,d,e,a, 0xCA62C1D6U, w[10])
    SHA1_EXPAND(w,11); SHA1_PAR(a,b,c,d,e, 0xCA62C1D6U, w[11])
    SHA1_EXPAND(w,12); SHA1_PAR(e,a,b,c,d, 0xCA62C1D6U, w[12])
    SHA1_EXPAND(w,13); SHA1_PAR(d,e,a,b,c, 0xCA62C1D6U, w[13])
    SHA1_EXPAND(w,14); SHA1_PAR(c,d,e,a,b, 0xCA62C1D6U, w[14])
    SHA1_EXPAND(w,15); SHA1_PAR(b,c,d,e,a, 0xCA62C1D6U, w[15])
}

// ---------------------------------------------------------------------------
// Template kernel: NCHUNK compile-time chunk count, TPB compile-time block
// size.  Benefits over the old proc_all_chunks with runtime n_chunk:
//
//   1. The compiler can statically unroll `for (int chunk=1; chunk<NCHUNK; ...)`
//      and fold all chunk_buffer[chunk*16+i] into constant offsets.
//   2. __launch_bounds__(TPB) anchors the register budget to the actual block
//      size, giving the compiler more room to reduce regs without spilling.
//
// Dispatched for NCHUNK = 1..4 and TPB ∈ {128, 256, 512}.
// Uncommon values fall through to proc_all_chunks (no __launch_bounds__).
// ---------------------------------------------------------------------------
template<int NCHUNK, int TPB>
__launch_bounds__(TPB)
__global__ static
void proc_chunks(u32 t0,
                 u32* __restrict__ h0, u32* __restrict__ h1,
                 u32* __restrict__ h2, u32* __restrict__ h3,
                 u32* __restrict__ h4)
{
    u32 index = blockIdx.x * blockDim.x + threadIdx.x;

    u32 ha = SHA1_A0, hb = SHA1_B0, hc = SHA1_C0, hd = SHA1_D0, he = SHA1_E0;

    // Chunk 0: explicit unrolled constant-memory load + per-thread timestamp.
    // All warp threads hit the same constant-memory addresses → broadcast.
    {
        u32 w[16];
#pragma unroll
        for (int i = 0; i < 16; i++)
            w[i] = chunk_buffer[i];
        // w[1] carries the creation timestamp
        // (bytes 4-7 of the OpenPGP fingerprint hash packet as a big-endian u32).
        w[1] = t0 - index;
        u32 a = ha, b = hb, c = hc, d = hd, e = he;
        sha1_80rounds(w, a, b, c, d, e);
        ha += a; hb += b; hc += c; hd += d; he += e;
    }

    // Remaining chunks: compile-time count → fully unrollable.
    // chunk_buffer[chunk * 16 + i] folds into a constant offset after unroll.
#pragma unroll
    for (int chunk = 1; chunk < NCHUNK; chunk++) {
        u32 w[16];
#pragma unroll
        for (int i = 0; i < 16; i++)
            w[i] = chunk_buffer[chunk * 16 + i];
        u32 a = ha, b = hb, c = hc, d = hd, e = he;
        sha1_80rounds(w, a, b, c, d, e);
        ha += a; hb += b; hc += c; hd += d; he += e;
    }

    h0[index] = ha;
    h1[index] = hb;
    h2[index] = hc;
    h3[index] = hd;
    h4[index] = he;
}

// Fallback for runtime n_chunk (> 4) or non-standard block sizes.
// No __launch_bounds__ so it can be launched with any block size.
__global__ static
void proc_all_chunks(u32 n_chunk, u32 t0,
                     u32* __restrict__ h0, u32* __restrict__ h1,
                     u32* __restrict__ h2, u32* __restrict__ h3,
                     u32* __restrict__ h4)
{
    u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    u32 ha = SHA1_A0, hb = SHA1_B0, hc = SHA1_C0, hd = SHA1_D0, he = SHA1_E0;

    {
        u32 w[16];
#pragma unroll
        for (int i = 0; i < 16; i++)
            w[i] = chunk_buffer[i];
        w[1] = t0 - index;
        u32 a = ha, b = hb, c = hc, d = hd, e = he;
        sha1_80rounds(w, a, b, c, d, e);
        ha += a; hb += b; hc += c; hd += d; he += e;
    }

    for (u32 chunk = 1; chunk < n_chunk; chunk++) {
        u32 w[16];
#pragma unroll
        for (int i = 0; i < 16; i++)
            w[i] = chunk_buffer[chunk * 16 + i];
        u32 a = ha, b = hb, c = hc, d = hd, e = he;
        sha1_80rounds(w, a, b, c, d, e);
        ha += a; hb += b; hc += c; hd += d; he += e;
    }

    h0[index] = ha;
    h1[index] = hb;
    h2[index] = hc;
    h3[index] = hd;
    h4[index] = he;
}

void CudaManager::gpu_proc_chunk(u32 n_chunk, u32 key_time0) const {
    // Dispatch to template specialisations for the common (n_chunk, block_size)
    // pairs so the compiler can fold constant-memory offsets and honour
    // __launch_bounds__ for the actual block size.  All other combinations
    // fall through to proc_all_chunks (no __launch_bounds__, runtime n_chunk).
#define DISPATCH_N(NC) \
    do { switch (thread_per_block_) { \
    case 128: proc_chunks<(NC),128><<<n_block_,128>>>(key_time0,h[0],h[1],h[2],h[3],h[4]); break; \
    case 256: proc_chunks<(NC),256><<<n_block_,256>>>(key_time0,h[0],h[1],h[2],h[3],h[4]); break; \
    case 512: proc_chunks<(NC),512><<<n_block_,512>>>(key_time0,h[0],h[1],h[2],h[3],h[4]); break; \
    default: proc_all_chunks<<<n_block_,thread_per_block_>>>((NC),key_time0,h[0],h[1],h[2],h[3],h[4]); break; \
    } } while (0)

    switch (n_chunk) {
    case 1:  DISPATCH_N(1); break;
    case 2:  DISPATCH_N(2); break;
    case 3:  DISPATCH_N(3); break;
    case 4:  DISPATCH_N(4); break;
    default: proc_all_chunks<<<n_block_,thread_per_block_>>>(n_chunk,key_time0,h[0],h[1],h[2],h[3],h[4]); break;
    }
#undef DISPATCH_N
}

u32 CudaManager::load_key(const std::vector<u8> &pubkey) const {
    std::vector<u8> buf = pubkey;
    u32 buf_len = buf.size();

    // sha-1 padding
    int pad_zero = (56 - (buf_len + 1) % 64) % 64;
    u32 buf_len2 = buf_len + 1 + pad_zero + 8;

    buf.push_back(0x80);
    buf.resize(buf_len2, 0);

    buf_len *= 8;
    for (auto it = buf.rbegin(); buf_len && it != buf.rend(); it++) {
        *it = buf_len & 0xff;
        buf_len >>= 8;
    }

    // group buffer to 32-bit words (byte-swap to produce native u32 big-endian words)
    for (u32 i = 0; i < buf_len2; i += 4) {
        std::swap(buf[i], buf[i + 3]);
        std::swap(buf[i + 1], buf[i + 2]);
    }

    DIE_ON_ERR(sizeof(chunk_buffer) >= buf_len2);

    u32 n_chunk = buf_len2 / 64;

    // Always keep chunk_buffer current so proc_chunks / proc_all_chunks are
    // always ready to run (fixes the previous split where single-chunk keys
    // only updated key_chunk0 and left chunk_buffer stale).
    CUDA_CALL(cudaMemcpyToSymbol, chunk_buffer, buf.data(), buf_len2);

    // Additionally update the NVRTC key_chunk0 symbol used by the fused
    // pattern_check_fused kernel for the single-chunk path.
    if (n_chunk == 1 && cu_key_chunk0 != 0)
        CU_CALL(cuMemcpyHtoD, cu_key_chunk0, buf.data(), 64);

    return n_chunk;
}
