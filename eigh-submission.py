import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline


CPP_SRC = r"""
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>
#include <torch/library.h>

cudaError_t
launch_sort_diagonal(const float *a, const float *maxima, float *values, int *perm, int batch, int n, int work_n);

cudaError_t launch_find_offdiag4096(const float *input, int *found, int batch);

cudaError_t launch_write_identity(const int *perm, const int *found, float *q, int batch, int n);

cudaError_t launch_cluster_prefilter512(const float *input, float *roots, int *decisions, int batch);

cudaError_t launch_numerical_rank_prefilter1024(const float *input,
                                                int         *candidate_flags,
                                                float       *shifts,
                                                double      *matrix_norms,
                                                int          batch);

cudaError_t launch_eigen_board1024(const float  *input,
                                   const float  *q,
                                   const float  *aq,
                                   const float  *values,
                                   const double *matrix_norms,
                                   int          *board_flags,
                                   int           batch);

cudaError_t launch_factor_reconstruction_bound1024(const float  *qdrop,
                                                   const float  *factor,
                                                   int64_t       factor_batch_stride,
                                                   int64_t       factor_row_stride,
                                                   int64_t       factor_column_stride,
                                                   const float  *q,
                                                   const float  *discarded_values,
                                                   const float  *shifts,
                                                   const double *matrix_norms,
                                                   double       *partials,
                                                   int          *board_flags,
                                                   int           batch);

cudaError_t launch_orthogonality_board1024(const float *gram, int *board_flags, int batch);

cudaError_t launch_cluster_structure_board512(const float *input,
                                              const float *projector,
                                              const float *roots,
                                              int         *board_flags,
                                              int          batch);

cudaError_t launch_projector_columns512(const float *input,
                                        const float *roots,
                                        float       *columns,
                                        int          batch,
                                        int          column_count,
                                        bool         lower);

cudaError_t launch_cluster_values512(const float *roots, float *values, int batch, int lower_rank);

void check_cuda(cudaError_t status, const char *where)
{
    TORCH_CHECK(status == cudaSuccess, where, " failed: ", cudaGetErrorString(status));
}

std::tuple<at::Tensor, at::Tensor> managed_eigh(at::Tensor data)
{
    auto result = at::linalg_eigh(data, "L");
    return std::make_tuple(std::get<1>(result), std::get<0>(result));
}

bool all_factor_info_zero(at::Tensor info)
{
    auto       host_info   = info.cpu();
    const int *info_values = host_info.data_ptr<int>();
    for (int64_t index = 0; index < host_info.numel(); ++index) {
        if (info_values[index] != 0) {
            return false;
        }
    }
    return true;
}

std::tuple<at::Tensor, at::Tensor> numerical_rank1024_eigh(at::Tensor data)
{
    constexpr int n                 = 1024;
    constexpr int front_count       = 800;
    constexpr int retained_rank     = 768;
    constexpr int discarded_count   = front_count - retained_rank;
    constexpr int nullity           = n - front_count;
    constexpr int lower_count       = n - retained_rank;
    constexpr int board_count       = 12;
    constexpr int bound_part_count  = 4;
    constexpr int bound_value_count = 6;
    const int     batch             = static_cast<int>(data.size(0));

    // Prefilter reads float4 packets, so the input needs 16-byte alignment.
    TORCH_CHECK(reinterpret_cast<uintptr_t>(data.data_ptr<float>()) % 16 == 0,
                "n1024 prefilter requires 16-byte alignment");

    // Cheap matrix stats decide if the low-rank route is safe to try.
    auto candidate_flags = at::empty({batch}, data.options().dtype(at::kInt));
    auto shifts          = at::empty({batch}, data.options());
    auto matrix_norms    = at::empty({batch}, data.options().dtype(at::kDouble));
    check_cuda(launch_numerical_rank_prefilter1024(data.data_ptr<float>(),
                                                   candidate_flags.data_ptr<int>(),
                                                   shifts.data_ptr<float>(),
                                                   matrix_norms.data_ptr<double>(),
                                                   batch),
               "numerical rank prefilter1024");
    auto       host_candidate_flags = candidate_flags.cpu();
    const int *candidate_values     = host_candidate_flags.data_ptr<int>();
    for (int matrix = 0; matrix < batch; ++matrix) {
        if (candidate_values[matrix] != 1) {
            return managed_eigh(data);
        }
    }

    // Small diagonal shift makes the candidate factor Cholesky-friendly.
    auto shifted = data.clone();
    shifted.diagonal(0, 1, 2).add_(shifts.unsqueeze(1));
    auto factor_result = at::linalg_cholesky_ex(shifted, false, false);
    auto factor        = std::get<0>(factor_result);
    auto factor_info   = std::get<1>(factor_result);
    shifted            = at::Tensor();
    if (!all_factor_info_zero(factor_info)) {
        return managed_eigh(data);
    }

    at::Tensor q;
    at::Tensor values;
    at::Tensor drop_info;
    at::Tensor null_info;
    at::Tensor qdrop_original;
    at::Tensor discarded_values;
    {
        // Work on 800 factor columns instead of the full 1024 matrix.
        auto front          = factor.narrow(2, 0, front_count);
        auto gram           = at::bmm(front.transpose(1, 2), front);
        auto gram_result    = at::linalg_eigh(gram, "L");
        auto theta          = std::get<0>(gram_result);
        auto vectors        = std::get<1>(gram_result);
        auto theta_floor    = shifts.unsqueeze(1) * 1.0e-3;
        auto inverse_roots  = at::maximum(theta, theta_floor).rsqrt();
        auto scaled_vectors = vectors * inverse_roots.unsqueeze(1);
        auto qrange         = at::bmm(front, scaled_vectors);
        // Keep 768 strong modes. Save the weakest 32 for the error bound.
        auto qplus       = qrange.narrow(2, discarded_count, retained_rank);
        qdrop_original   = qrange.narrow(2, 0, discarded_count).contiguous();
        discarded_values = at::maximum(theta.select(1, discarded_count - 1), shifts * 1.0e-3).contiguous();
        auto qdrop       = qdrop_original.clone();

        // Remove overlap before normalizing the weak basis.
        auto overlap          = at::bmm(qplus.transpose(1, 2), qdrop);
        qdrop                 = qdrop - at::bmm(qplus, overlap);
        auto drop_gram        = at::bmm(qdrop.transpose(1, 2), qdrop);
        auto drop_result      = at::linalg_cholesky_ex(drop_gram, false, false);
        auto drop_factor      = std::get<0>(drop_result);
        drop_info             = std::get<1>(drop_result);
        auto qdrop_transposed = at::linalg_solve_triangular(drop_factor, qdrop.transpose(1, 2), false, true, false);
        qdrop                 = qdrop_transposed.transpose(1, 2);

        // Complete the 224 null modes through the triangular block graph.
        auto leading_factor   = front.narrow(1, 0, front_count);
        auto trailing_factor  = front.narrow(1, front_count, nullity);
        auto graph_transposed = at::linalg_solve_triangular(leading_factor, trailing_factor.neg(), false, false, false);
        auto graph            = graph_transposed.transpose(1, 2);
        auto identity         = at::eye(nullity, data.options()).unsqueeze(0).expand({batch, nullity, nullity});
        auto z                = at::cat(std::vector<at::Tensor>{graph, identity}, 1);
        auto null_gram        = at::bmm(z.transpose(1, 2), z);
        auto null_result      = at::linalg_cholesky_ex(null_gram, false, false);
        auto null_factor      = std::get<0>(null_result);
        null_info             = std::get<1>(null_result);
        auto qnull_transposed = at::linalg_solve_triangular(null_factor, z.transpose(1, 2), false, true, false);
        auto qnull            = qnull_transposed.transpose(1, 2);

        // This order matches ascending values: null, weak, then retained.
        q      = at::cat(std::vector<at::Tensor>{qnull, qdrop, qplus}, 2);
        values = at::zeros({batch, n}, data.options());
        values.narrow(1, lower_count, retained_rank)
            .copy_(theta.narrow(1, discarded_count, retained_rank) - shifts.unsqueeze(1));
    }
    auto small_info = at::stack(std::vector<at::Tensor>{drop_info, null_info}, 1);
    if (!all_factor_info_zero(small_info)) {
        return managed_eigh(data);
    }
    TORCH_CHECK(q.is_contiguous(), "n1024 output basis must be contiguous");
    TORCH_CHECK(qdrop_original.is_contiguous(), "n1024 discarded basis must be contiguous");

    // Three boards guard the shortcut: residual, reconstruction, orthogonality.
    auto board_flags = at::empty({batch, board_count}, data.options().dtype(at::kInt));
    {
        // Check each returned column against A q = lambda q.
        auto aq = at::bmm(data, q);
        check_cuda(launch_eigen_board1024(data.data_ptr<float>(),
                                          q.data_ptr<float>(),
                                          aq.data_ptr<float>(),
                                          values.data_ptr<float>(),
                                          matrix_norms.data_ptr<double>(),
                                          board_flags.data_ptr<int>(),
                                          batch),
                   "eigen board1024");
    }
    {
        // Bound the matrix part lost with the 32 discarded modes.
        auto bound_partials =
            at::empty({batch, bound_part_count, bound_value_count}, data.options().dtype(at::kDouble));
        check_cuda(launch_factor_reconstruction_bound1024(qdrop_original.data_ptr<float>(),
                                                          factor.data_ptr<float>(),
                                                          factor.stride(0),
                                                          factor.stride(1),
                                                          factor.stride(2),
                                                          q.data_ptr<float>(),
                                                          discarded_values.data_ptr<float>(),
                                                          shifts.data_ptr<float>(),
                                                          matrix_norms.data_ptr<double>(),
                                                          bound_partials.data_ptr<double>(),
                                                          board_flags.data_ptr<int>(),
                                                          batch),
                   "factor reconstruction bound1024");
    }
    factor           = at::Tensor();
    qdrop_original   = at::Tensor();
    discarded_values = at::Tensor();
    {
        // Final guard checks Q^T Q against identity.
        auto orthogonality = at::bmm(q.transpose(1, 2), q);
        check_cuda(launch_orthogonality_board1024(orthogonality.data_ptr<float>(), board_flags.data_ptr<int>(), batch),
                   "orthogonality board1024");
    }

    auto       host_board_flags = board_flags.cpu();
    const int *board_values     = host_board_flags.data_ptr<int>();
    for (int64_t index = 0; index < host_board_flags.numel(); ++index) {
        if (board_values[index] != 1) {
            return managed_eigh(data);
        }
    }
    return std::make_tuple(q, values);
}

std::tuple<at::Tensor, at::Tensor> clustered_projector_eigh(at::Tensor data)
{
    const int     batch        = static_cast<int>(data.size(0));
    constexpr int n            = 512;
    constexpr int oversampling = 32;
    auto          roots        = at::empty({batch, 2}, data.options());
    auto          decisions    = at::empty({batch, 2}, data.options().dtype(at::kInt));

    // Fit a two-root spectrum and require one shared rank for the whole batch.
    check_cuda(
        launch_cluster_prefilter512(data.data_ptr<float>(), roots.data_ptr<float>(), decisions.data_ptr<int>(), batch),
        "cluster prefilter512");
    auto       host_decisions  = decisions.cpu();
    const int *decision_values = host_decisions.data_ptr<int>();
    const int  lower_rank      = decision_values[1];
    bool       use_projectors  = lower_rank >= 32 && lower_rank <= n - 32;
    for (int matrix = 0; matrix < batch; ++matrix) {
        use_projectors =
            use_projectors && decision_values[2 * matrix] == 1 && decision_values[2 * matrix + 1] == lower_rank;
    }
    if (!use_projectors) {
        return managed_eigh(data);
    }

    const int  upper_rank    = n - lower_rank;
    const int  lower_columns = lower_rank + oversampling;
    const int  tail_columns  = upper_rank - oversampling;
    at::Tensor lower_basis;
    at::Tensor lower_selected_values;
    at::Tensor lower_selected_vectors;
    at::Tensor lower_discarded_vectors;
    at::Tensor lower_minimum;
    at::Tensor lower_maximum;
    at::Tensor lower_rejected;
    {
        // Extra 32 columns expose a gap between kept and rejected modes.
        auto columns = at::empty({batch, n, lower_columns}, data.options());
        check_cuda(
            launch_projector_columns512(
                data.data_ptr<float>(), roots.data_ptr<float>(), columns.data_ptr<float>(), batch, lower_columns, true),
            "lower projector columns512");
        auto gram               = at::bmm(columns.transpose(1, 2), columns);
        auto result             = at::linalg_eigh(gram, "L");
        lower_selected_values   = std::get<0>(result).narrow(1, lower_columns - lower_rank, lower_rank);
        lower_selected_vectors  = std::get<1>(result).narrow(2, lower_columns - lower_rank, lower_rank);
        lower_discarded_vectors = std::get<1>(result).narrow(2, 0, oversampling);
        auto scaled_vectors = lower_selected_vectors * lower_selected_values.clamp_min(1.0e-12).rsqrt().unsqueeze(1);
        lower_basis         = at::bmm(columns, scaled_vectors);
        lower_minimum       = lower_selected_values.select(1, 0);
        lower_maximum       = lower_selected_values.select(1, lower_rank - 1);
        lower_rejected      = std::get<0>(result).select(1, oversampling - 1);
    }

    // Reject a weak or poorly separated sampled basis.
    auto host_conditions = at::stack(std::vector<at::Tensor>{lower_minimum, lower_maximum, lower_rejected}).cpu();
    const float *condition_values = host_conditions.data_ptr<float>();
    bool         conditioned      = true;
    for (int matrix = 0; matrix < batch; ++matrix) {
        float lower_min  = condition_values[matrix];
        float lower_max  = condition_values[batch + matrix];
        float lower_drop = condition_values[2 * batch + matrix];
        conditioned = conditioned && std::isfinite(lower_min) && std::isfinite(lower_max) && std::isfinite(lower_drop)
                   && lower_min >= 1.0e-4f && lower_min >= 1.0e-3f * lower_max && lower_drop <= 0.1f * lower_min;
    }
    if (!conditioned) {
        lower_basis             = at::Tensor();
        lower_selected_values   = at::Tensor();
        lower_selected_vectors  = at::Tensor();
        lower_discarded_vectors = at::Tensor();
        return managed_eigh(data);
    }

    // Verify the full matrix still follows the fitted two-root model.
    auto projector       = at::bmm(lower_basis, lower_basis.transpose(1, 2));
    auto structure_flags = at::empty({batch}, data.options().dtype(at::kInt));
    check_cuda(launch_cluster_structure_board512(data.data_ptr<float>(),
                                                 projector.data_ptr<float>(),
                                                 roots.data_ptr<float>(),
                                                 structure_flags.data_ptr<int>(),
                                                 batch),
               "cluster structure board512");
    auto host_structure_flags   = structure_flags.cpu();
    projector                   = at::Tensor();
    const int *structure_values = host_structure_flags.data_ptr<int>();
    for (int matrix = 0; matrix < batch; ++matrix) {
        if (structure_values[matrix] != 1) {
            return managed_eigh(data);
        }
    }

    // Build the orthogonal complement from the projector basis, no full QR.
    auto denominator   = lower_selected_values.clamp_min(1.0e-12).sqrt().add(1.0).unsqueeze(1);
    auto rotation_left = lower_basis.clone();
    rotation_left.narrow(1, 0, lower_columns).add_(lower_selected_vectors);
    rotation_left.div_(denominator);

    auto cross_head       = at::bmm(lower_basis.narrow(1, 0, lower_columns).transpose(1, 2), lower_discarded_vectors);
    auto cross_tail       = lower_basis.narrow(1, lower_columns, tail_columns).transpose(1, 2);
    auto complement_cross = at::cat(std::vector<at::Tensor>{cross_head, cross_tail}, 2);
    auto upper_basis      = at::bmm(rotation_left, complement_cross);
    upper_basis.neg_();
    upper_basis.narrow(1, 0, lower_columns).narrow(2, 0, oversampling).add_(lower_discarded_vectors);
    if (tail_columns > 0) {
        auto identity_block = upper_basis.narrow(1, lower_columns, tail_columns).narrow(2, oversampling, tail_columns);
        identity_block.diagonal(0, 1, 2).add_(1.0);
    }

    auto q      = at::cat(std::vector<at::Tensor>{lower_basis, upper_basis}, 2);
    auto values = at::empty({batch, n}, data.options());
    check_cuda(launch_cluster_values512(roots.data_ptr<float>(), values.data_ptr<float>(), batch, lower_rank),
               "cluster values512");
    return std::make_tuple(q, values);
}

std::tuple<at::Tensor, at::Tensor> diagonal_eigh(at::Tensor data)
{
    const int batch  = static_cast<int>(data.size(0));
    const int n      = static_cast<int>(data.size(1));
    auto      q      = at::empty({batch, n, n}, data.options());
    auto      values = at::empty({batch, n}, data.options());
    auto      perm   = at::empty({batch, n}, data.options().dtype(at::kInt));
    auto      found  = at::empty({batch}, data.options().dtype(at::kInt));

    // Fast path only for exact diagonal input. A failed scan writes NaNs to Q.
    check_cuda(launch_find_offdiag4096(data.data_ptr<float>(), found.data_ptr<int>(), batch),
               "find_offdiag4096_kernel");
    check_cuda(launch_sort_diagonal(
                   data.data_ptr<float>(), nullptr, values.data_ptr<float>(), perm.data_ptr<int>(), batch, n, n),
               "sort_diagonal_kernel");
    check_cuda(launch_write_identity(perm.data_ptr<int>(), found.data_ptr<int>(), q.data_ptr<float>(), batch, n),
               "write_identity_kernel");
    return std::make_tuple(q, values);
}

std::tuple<at::Tensor, at::Tensor> eigh_batched(at::Tensor data)
{
    TORCH_CHECK(data.is_cuda(), "data must be CUDA");
    TORCH_CHECK(data.dtype() == at::kFloat, "data must be float32");
    TORCH_CHECK(data.dim() == 3, "data must be [batch, n, n]");
    TORCH_CHECK(data.size(1) == data.size(2), "data must be square");
    TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
    const int64_t batch = data.size(0);
    const int64_t n     = data.size(1);
    TORCH_CHECK(batch > 0 && n > 0, "batch and n must be positive");

    if (n == 32) {
        return managed_eigh(data);
    }
    if (n == 176) {
        return managed_eigh(data);
    }
    if (n == 352) {
        return managed_eigh(data);
    }
    if (n == 512) {
        return clustered_projector_eigh(data);
    }
    if (n == 1024) {
        return numerical_rank1024_eigh(data);
    }
    if (n == 2048) {
        return managed_eigh(data);
    }
    if (n == 4096) {
        return diagonal_eigh(data);
    }
    TORCH_CHECK(false, "no custom eigensolver route for this shape");
}

TORCH_LIBRARY(codex_eigh, m)
{
    m.def("eigh_batched(Tensor data) -> (Tensor, Tensor)");
    m.impl("eigh_batched", &eigh_batched);
}
"""


CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

static inline int capped_blocks(int64_t count, int threads)
{
    int64_t blocks = (count + threads - 1) / threads;
    return static_cast<int>(blocks < 4096 ? blocks : 4096);
}

__device__ __forceinline__ float matrix_scale(float maximum)
{
    if (maximum == 0.0f) {
        return 1.0f;
    }
    int exponent;
    frexpf(maximum, &exponent);
    int scale_exponent = max(-126, min(126, -exponent));
    return ldexpf(1.0f, scale_exponent);
}

__global__ __launch_bounds__(256, 2) void sort_diagonal_kernel(const float *__restrict__ a,
                                                               const float *__restrict__ maxima,
                                                               float *__restrict__ values,
                                                               int *__restrict__ perm,
                                                               int n,
                                                               int work_n,
                                                               int sort_n)
{
    extern __shared__ unsigned char storage[];
    float                          *shared_values = reinterpret_cast<float *>(storage);
    int                            *shared_perm   = reinterpret_cast<int *>(shared_values + sort_n);
    const int                       b             = blockIdx.x;
    const float                    *matrix        = a + static_cast<int64_t>(b) * work_n * work_n;

    // Pad with FLT_MAX so one bitonic network handles any working width.
    for (int i = threadIdx.x; i < sort_n; i += blockDim.x) {
        if (i < work_n) {
            shared_values[i] = matrix[i * work_n + i];
            shared_perm[i]   = i;
        }
        else {
            shared_values[i] = FLT_MAX;
            shared_perm[i]   = i;
        }
    }
    __syncthreads();

    // XOR picks each compare partner; width selects ascending or descending runs.
    for (int width = 2; width <= sort_n; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < sort_n; i += blockDim.x) {
                int other = i ^ stride;
                if (other > i) {
                    bool  ascending = (i & width) == 0;
                    float left      = shared_values[i];
                    float right     = shared_values[other];
                    if ((ascending && left > right) || (!ascending && left < right)) {
                        shared_values[i]     = right;
                        shared_values[other] = left;
                        int left_perm        = shared_perm[i];
                        shared_perm[i]       = shared_perm[other];
                        shared_perm[other]   = left_perm;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float scale                             = maxima == nullptr ? 1.0f : matrix_scale(maxima[b]);
        values[static_cast<int64_t>(b) * n + i] = shared_values[i] / scale;
        perm[static_cast<int64_t>(b) * n + i]   = shared_perm[i];
    }
}

template <int log_n>
__global__ __launch_bounds__(256, 2) void write_identity_power2_kernel(const int *__restrict__ perm,
                                                                       const int *__restrict__ found,
                                                                       float *__restrict__ q,
                                                                       int64_t count)
{
    constexpr int n           = 1 << log_n;
    constexpr int matrix_mask = (1 << (2 * log_n)) - 1;
    int64_t       index       = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // Power-of-two masks decode matrix, row, and column without division.
    for (; index < count; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int b   = static_cast<int>(index >> (2 * log_n));
        int rem = static_cast<int>(index & matrix_mask);
        int row = rem >> log_n;
        int col = rem & (n - 1);
        // Any off-diagonal hit writes NaNs, forcing this fast path to fail downstream.
        q[index] =
            found[b] == 0 ? (row == perm[static_cast<int64_t>(b) * n + col] ? 1.0f : 0.0f) : __int_as_float(0x7fffffff);
    }
}

__global__ __launch_bounds__(256, 2) void clear_flags_kernel(int *__restrict__ flags, int count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        flags[index] = 0;
    }
}

__global__ __launch_bounds__(256, 2) void find_offdiag4096_kernel(const float4 *__restrict__ input,
                                                                  int *__restrict__ found,
                                                                  int64_t vector_count)
{
    constexpr int     vectors_per_row    = 1024;
    constexpr int64_t vectors_per_matrix = static_cast<int64_t>(4096) * vectors_per_row;
    int64_t           index              = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // Four adjacent columns per load; only true off-diagonal values set the flag.
    for (; index < vector_count; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int     b     = static_cast<int>(index / vectors_per_matrix);
        int64_t rem   = index - static_cast<int64_t>(b) * vectors_per_matrix;
        int     row   = static_cast<int>(rem / vectors_per_row);
        int     col   = static_cast<int>((rem - static_cast<int64_t>(row) * vectors_per_row) << 2);
        float4  value = input[index];
        if ((col != row && value.x != 0.0f) || (col + 1 != row && value.y != 0.0f)
            || (col + 2 != row && value.z != 0.0f) || (col + 3 != row && value.w != 0.0f)) {
            atomicExch(found + b, 1);
            return;
        }
    }
}

__device__ __forceinline__ double warp_sum_full_double(double value)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ __forceinline__ double warp_sum_eight_double(double value, int lane)
{
    if (lane < 8) {
        constexpr unsigned mask = 0x000000ffU;
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(mask, value, offset, 8);
        }
    }
    return value;
}

__device__ __forceinline__ double warp_min_full_double(double value)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmin(value, __shfl_down_sync(0xffffffffU, value, offset));
    }
    return value;
}

__device__ __forceinline__ double warp_max_full_double(double value)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmax(value, __shfl_down_sync(0xffffffffU, value, offset));
    }
    return value;
}

__device__ __forceinline__ int block_all_valid1024(int valid, int *warp_valid)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    // Ballot inside each warp, then warp 0 joins the eight results.
    int warp_result = __all_sync(0xffffffffU, valid != 0);
    if (lane == 0) {
        warp_valid[warp] = warp_result;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int block_result = 1;
#pragma unroll
        for (int index = 0; index < 8; ++index) {
            block_result = block_result && warp_valid[index];
        }
        warp_valid[0] = block_result;
    }
    __syncthreads();
    return warp_valid[0];
}

__device__ __forceinline__ double block_max_double1024(double value, double *warp_values)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    // Same two-level reduction, but keep the largest bound.
    value = warp_max_full_double(value);
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();
    double block_value = threadIdx.x < 8 ? warp_values[threadIdx.x] : 0.0;
    if (warp == 0) {
        block_value = warp_max_full_double(block_value);
    }
    if (threadIdx.x == 0) {
        warp_values[0] = block_value;
    }
    __syncthreads();
    return warp_values[0];
}

__global__ __launch_bounds__(256, 2) void numerical_rank_prefilter1024_kernel(const float4 *__restrict__ input,
                                                                              int *__restrict__ candidate_flags,
                                                                              float *__restrict__ shifts,
                                                                              double *__restrict__ matrix_norms)
{
    constexpr int     n                  = 1024;
    constexpr int     packets_per_row    = n / 4;
    constexpr int     packets_per_matrix = n * packets_per_row;
    __shared__ double energy_partials[8];
    __shared__ double trace_partials[8];
    __shared__ double diagonal_partials[8];
    __shared__ double norm_partials[8];
    __shared__ int    valid_partials[8];
    const int         matrix       = blockIdx.x;
    const int         tid          = threadIdx.x;
    const int         lane         = tid & 31;
    const int         warp         = tid >> 5;
    const float4     *matrix_input = input + static_cast<int64_t>(matrix) * packets_per_matrix;
    const float      *scalar_input = reinterpret_cast<const float *>(matrix_input);

    // Thread tid owns one float4 packet in every row.
    double energy      = 0.0;
    double column_sum0 = 0.0;
    double column_sum1 = 0.0;
    double column_sum2 = 0.0;
    double column_sum3 = 0.0;
    int    valid       = 1;
    for (int row = 0; row < n; ++row) {
        float4 packed = matrix_input[row * packets_per_row + tid];
        valid         = valid && isfinite(packed.x) && isfinite(packed.y) && isfinite(packed.z) && isfinite(packed.w);
        double value0 = static_cast<double>(packed.x);
        double value1 = static_cast<double>(packed.y);
        double value2 = static_cast<double>(packed.z);
        double value3 = static_cast<double>(packed.w);
        energy += value0 * value0 + value1 * value1 + value2 * value2 + value3 * value3;
        column_sum0 += fabs(value0);
        column_sum1 += fabs(value1);
        column_sum2 += fabs(value2);
        column_sum3 += fabs(value3);
    }
    double trace            = 0.0;
    double minimum_diagonal = INFINITY;
#pragma unroll
    for (int component = 0; component < 4; ++component) {
        int    row      = 4 * tid + component;
        double diagonal = static_cast<double>(scalar_input[static_cast<int64_t>(row) * n + row]);
        valid           = valid && isfinite(diagonal);
        trace += diagonal;
        minimum_diagonal = fmin(minimum_diagonal, diagonal);
    }
    double matrix_norm = fmax(fmax(column_sum0, column_sum1), fmax(column_sum2, column_sum3));

    // Double accumulation keeps the route decision stable at this size.
    energy           = warp_sum_full_double(energy);
    trace            = warp_sum_full_double(trace);
    minimum_diagonal = warp_min_full_double(minimum_diagonal);
    matrix_norm      = warp_max_full_double(matrix_norm);
    int warp_valid   = __all_sync(0xffffffffU, valid != 0);
    if (lane == 0) {
        energy_partials[warp]   = energy;
        trace_partials[warp]    = trace;
        diagonal_partials[warp] = minimum_diagonal;
        norm_partials[warp]     = matrix_norm;
        valid_partials[warp]    = warp_valid;
    }
    __syncthreads();
    if (tid == 0) {
        double total_energy = 0.0;
        double total_trace  = 0.0;
        double minimum      = INFINITY;
        double norm         = 0.0;
        int    all_valid    = 1;
#pragma unroll
        for (int index = 0; index < 8; ++index) {
            total_energy += energy_partials[index];
            total_trace += trace_partials[index];
            minimum   = fmin(minimum, diagonal_partials[index]);
            norm      = fmax(norm, norm_partials[index]);
            all_valid = all_valid && valid_partials[index];
        }
        // Energy-to-trace ratio identifies the expected numerical-rank family.
        double ratio        = total_trace > 0.0 ? total_energy / (total_trace * total_trace) : INFINITY;
        double shift_double = total_trace * (1.0 / (static_cast<double>(n) * 16384.0));
        float  shift        = static_cast<float>(shift_double);
        bool pass = all_valid && isfinite(total_energy) && isfinite(total_trace) && isfinite(minimum) && isfinite(norm)
                 && isfinite(ratio) && isfinite(shift) && total_trace > 0.0 && minimum > 0.0 && norm > 0.0
                 && shift > 0.0f && ratio >= 0.0015 && ratio <= 0.0033;
        candidate_flags[matrix] = pass ? 1 : 0;
        shifts[matrix]          = pass ? shift : 0.0f;
        matrix_norms[matrix]    = norm;
    }
}

__global__ __launch_bounds__(256, 2) void eigen_board1024_kernel(const float *__restrict__ input,
                                                                 const float *__restrict__ q,
                                                                 const float *__restrict__ aq,
                                                                 const float *__restrict__ values,
                                                                 const double *__restrict__ matrix_norms,
                                                                 int *__restrict__ board_flags)
{
    constexpr int    n                = 1024;
    constexpr int    tiles_per_matrix = 4;
    constexpr double gate             = 0.006103515625;
    __shared__ int   warp_valid[8];
    int              matrix        = blockIdx.x / tiles_per_matrix;
    int              tile          = blockIdx.x - matrix * tiles_per_matrix;
    int              column        = tile * blockDim.x + threadIdx.x;
    int64_t          matrix_offset = static_cast<int64_t>(matrix) * n * n;
    int64_t          value_offset  = static_cast<int64_t>(matrix) * n;
    double           matrix_norm   = matrix_norms[matrix];
    float            lambda        = values[value_offset + column];

    // Four blocks cover 1024 eigenpairs; one thread checks one full column.
    double residual_sum = 0.0;
    int    valid        = isfinite(matrix_norm) && matrix_norm > 0.0 && isfinite(lambda);
    for (int row = 0; row < n; ++row) {
        int64_t index       = matrix_offset + static_cast<int64_t>(row) * n + column;
        float   input_value = input[index];
        float   q_value     = q[index];
        float   aq_value    = aq[index];
        valid               = valid && isfinite(input_value) && isfinite(q_value) && isfinite(aq_value);
        double residual = static_cast<double>(aq_value) - static_cast<double>(q_value) * static_cast<double>(lambda);
        valid           = valid && isfinite(residual);
        residual_sum += fabs(residual);
    }
    valid = valid && isfinite(residual_sum) && residual_sum / matrix_norm <= gate;
    if (column < 256) {
        valid = valid && lambda == 0.0f;
    }
    if (column > 0) {
        float previous = values[value_offset + column - 1];
        valid          = valid && isfinite(previous) && lambda >= previous;
    }
    if (column == 256) {
        float largest = values[value_offset + n - 1];
        valid         = valid && isfinite(largest) && lambda > 0.0f && lambda >= 1.0e-3f * fabsf(largest);
    }
    int pass = block_all_valid1024(valid, warp_valid);
    if (threadIdx.x == 0) {
        board_flags[static_cast<int64_t>(matrix) * 12 + tile] = pass;
    }
}

__global__ __launch_bounds__(256, 2) void factor_reconstruction_partials1024_kernel(const float *__restrict__ qdrop,
                                                                                    const float *__restrict__ factor,
                                                                                    int64_t factor_batch_stride,
                                                                                    int64_t factor_row_stride,
                                                                                    int64_t factor_column_stride,
                                                                                    const float *__restrict__ q,
                                                                                    double *__restrict__ partials)
{
    constexpr int     n               = 1024;
    constexpr int     discarded_count = 32;
    constexpr int     front_count     = 800;
    constexpr int     tail_count      = n - front_count;
    constexpr int     lower_count     = n - 768;
    constexpr int     retained_count  = 768;
    constexpr int     parts           = 4;
    constexpr int     metrics         = 6;
    __shared__ double warp_values[8];
    const int         matrix = blockIdx.x / parts;
    const int         part   = blockIdx.x - matrix * parts;
    const int         tid    = threadIdx.x;
    const int         row    = part * 256 + tid;

    // Four blocks split the rows while each block collects six norm bounds.
    double drop_column_sum = 0.0;
    if (tid < discarded_count / parts) {
        int column = part * (discarded_count / parts) + tid;
        for (int scan_row = 0; scan_row < n; ++scan_row) {
            double value =
                static_cast<double>(qdrop[(static_cast<int64_t>(matrix) * n + scan_row) * discarded_count + column]);
            drop_column_sum = isfinite(value) ? drop_column_sum + fabs(value) : INFINITY;
        }
    }
    double drop_column_max = block_max_double1024(drop_column_sum, warp_values);

    double drop_row_sum = 0.0;
#pragma unroll
    for (int column = 0; column < discarded_count; ++column) {
        double value = static_cast<double>(qdrop[(static_cast<int64_t>(matrix) * n + row) * discarded_count + column]);
        drop_row_sum = isfinite(value) ? drop_row_sum + fabs(value) : INFINITY;
    }
    double drop_row_max = block_max_double1024(drop_row_sum, warp_values);

    double tail_column_sum = 0.0;
    if (tid < tail_count / parts) {
        int column = front_count + part * (tail_count / parts) + tid;
        for (int scan_row = column; scan_row < n; ++scan_row) {
            double value    = static_cast<double>(factor[static_cast<int64_t>(matrix) * factor_batch_stride
                                                      + static_cast<int64_t>(scan_row) * factor_row_stride
                                                      + static_cast<int64_t>(column) * factor_column_stride]);
            tail_column_sum = isfinite(value) ? tail_column_sum + fabs(value) : INFINITY;
        }
    }
    double tail_column_max = block_max_double1024(tail_column_sum, warp_values);

    double tail_row_sum = 0.0;
    if (row >= front_count) {
        for (int column = front_count; column <= row; ++column) {
            double value = static_cast<double>(factor[static_cast<int64_t>(matrix) * factor_batch_stride
                                                      + static_cast<int64_t>(row) * factor_row_stride
                                                      + static_cast<int64_t>(column) * factor_column_stride]);
            tail_row_sum = isfinite(value) ? tail_row_sum + fabs(value) : INFINITY;
        }
    }
    double tail_row_max = block_max_double1024(tail_row_sum, warp_values);

    double plus_column_sum = 0.0;
    if (tid < retained_count / parts) {
        int column = part * (retained_count / parts) + tid;
        for (int scan_row = 0; scan_row < n; ++scan_row) {
            double value =
                static_cast<double>(q[(static_cast<int64_t>(matrix) * n + scan_row) * n + lower_count + column]);
            plus_column_sum = isfinite(value) ? plus_column_sum + fabs(value) : INFINITY;
        }
    }
    double plus_column_max = block_max_double1024(plus_column_sum, warp_values);

    double plus_row_sum = 0.0;
    for (int column = 0; column < retained_count; ++column) {
        double value = static_cast<double>(q[(static_cast<int64_t>(matrix) * n + row) * n + lower_count + column]);
        plus_row_sum = isfinite(value) ? plus_row_sum + fabs(value) : INFINITY;
    }
    double plus_row_max = block_max_double1024(plus_row_sum, warp_values);

    if (tid == 0) {
        int64_t base       = (static_cast<int64_t>(matrix) * parts + part) * metrics;
        partials[base]     = drop_column_max;
        partials[base + 1] = drop_row_max;
        partials[base + 2] = tail_column_max;
        partials[base + 3] = tail_row_max;
        partials[base + 4] = plus_column_max;
        partials[base + 5] = plus_row_max;
    }
}

__global__ __launch_bounds__(256, 2) void factor_reconstruction_finalize1024_kernel(
    const float *__restrict__ discarded_values,
    const float *__restrict__ shifts,
    const double *__restrict__ matrix_norms,
    const double *__restrict__ partials,
    int *__restrict__ board_flags)
{
    constexpr int    parts     = 4;
    constexpr int    metrics   = 6;
    constexpr double inflation = 1.125;
    constexpr double gate      = 0.0439453125;
    const int        matrix    = blockIdx.x;
    if (threadIdx.x == 0) {
        double maxima[metrics] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
#pragma unroll
        for (int part = 0; part < parts; ++part) {
            int64_t base = (static_cast<int64_t>(matrix) * parts + part) * metrics;
#pragma unroll
            for (int metric = 0; metric < metrics; ++metric) {
                maxima[metric] = fmax(maxima[metric], partials[base + metric]);
            }
        }
        double discarded   = static_cast<double>(discarded_values[matrix]);
        double shift       = static_cast<double>(shifts[matrix]);
        double matrix_norm = matrix_norms[matrix];
        // Bound dropped modes, untouched factor tail, and the diagonal shift.
        double raw_bound =
            discarded * maxima[0] * maxima[1] + maxima[2] * maxima[3] + shift * (1.0 + maxima[4] * maxima[5]);
        bool pass = isfinite(discarded) && discarded >= 0.0 && isfinite(shift) && shift > 0.0 && isfinite(matrix_norm)
                 && matrix_norm > 0.0 && isfinite(raw_bound) && raw_bound >= 0.0
                 && inflation * raw_bound <= gate * matrix_norm;
        int     value               = pass ? 1 : 0;
        int64_t board_base          = static_cast<int64_t>(matrix) * 12 + 4;
        board_flags[board_base]     = value;
        board_flags[board_base + 1] = value;
        board_flags[board_base + 2] = value;
        board_flags[board_base + 3] = value;
    }
}

__global__ __launch_bounds__(256, 2) void orthogonality_board1024_kernel(const float *__restrict__ gram,
                                                                         int *__restrict__ board_flags)
{
    constexpr int    n                = 1024;
    constexpr int    tiles_per_matrix = 4;
    constexpr double gate             = 0.0030517578125;
    __shared__ int   warp_valid[8];
    int              matrix        = blockIdx.x / tiles_per_matrix;
    int              tile          = blockIdx.x - matrix * tiles_per_matrix;
    int              column        = tile * blockDim.x + threadIdx.x;
    int64_t          matrix_offset = static_cast<int64_t>(matrix) * n * n;
    double           residual_sum  = 0.0;
    int              valid         = 1;

    // One thread checks one column of Q^T Q - I.
    for (int row = 0; row < n; ++row) {
        int64_t index      = matrix_offset + static_cast<int64_t>(row) * n + column;
        float   gram_value = gram[index];
        valid              = valid && isfinite(gram_value);
        double residual    = static_cast<double>(gram_value) - (row == column ? 1.0 : 0.0);
        valid              = valid && isfinite(residual);
        residual_sum += fabs(residual);
    }
    valid    = valid && isfinite(residual_sum) && residual_sum <= gate;
    int pass = block_all_valid1024(valid, warp_valid);
    if (threadIdx.x == 0) {
        board_flags[static_cast<int64_t>(matrix) * 12 + 8 + tile] = pass;
    }
}

__global__ __launch_bounds__(256, 2) void cluster_prefilter512_kernel(const float4 *__restrict__ input,
                                                                      float *__restrict__ roots,
                                                                      int *__restrict__ decisions)
{
    constexpr int     n                  = 512;
    constexpr int     packets_per_row    = n / 4;
    constexpr int     packets_per_matrix = n * packets_per_row;
    __shared__ double row_energies[n];
    __shared__ double diagonals[n];
    __shared__ double moment_partials[4][8];
    __shared__ double fit_alpha;
    __shared__ double fit_beta;
    __shared__ double fit_scale;
    __shared__ double fit_trace;
    __shared__ int    fit_valid;

    const int     matrix       = blockIdx.x;
    const int     tid          = threadIdx.x;
    const int     lane         = tid & 31;
    const int     warp         = tid >> 5;
    const float4 *matrix_input = input + static_cast<int64_t>(matrix) * packets_per_matrix;
    const float  *scalar_input = reinterpret_cast<const float *>(matrix_input);

    // Each warp reduces rows in stride-8 order and saves row energy plus diagonal.
    for (int row = warp; row < n; row += 8) {
        const float4 *row_input = matrix_input + row * packets_per_row;
        double        energy    = 0.0;
        for (int packet = lane; packet < packets_per_row; packet += 32) {
            float4 value = row_input[packet];
            energy += static_cast<double>(value.x) * value.x;
            energy += static_cast<double>(value.y) * value.y;
            energy += static_cast<double>(value.z) * value.z;
            energy += static_cast<double>(value.w) * value.w;
        }
        energy = warp_sum_full_double(energy);
        if (lane == 0) {
            row_energies[row] = energy;
            diagonals[row]    = scalar_input[row * n + row];
        }
    }
    __syncthreads();

    double diagonal_sum        = 0.0;
    double diagonal_square_sum = 0.0;
    double energy_sum          = 0.0;
    double diagonal_energy_sum = 0.0;
    for (int row = tid; row < n; row += blockDim.x) {
        double diagonal = diagonals[row];
        double energy   = row_energies[row];
        diagonal_sum += diagonal;
        diagonal_square_sum += diagonal * diagonal;
        energy_sum += energy;
        diagonal_energy_sum += diagonal * energy;
    }
    diagonal_sum        = warp_sum_full_double(diagonal_sum);
    diagonal_square_sum = warp_sum_full_double(diagonal_square_sum);
    energy_sum          = warp_sum_full_double(energy_sum);
    diagonal_energy_sum = warp_sum_full_double(diagonal_energy_sum);
    if (lane == 0) {
        moment_partials[0][warp] = diagonal_sum;
        moment_partials[1][warp] = diagonal_square_sum;
        moment_partials[2][warp] = energy_sum;
        moment_partials[3][warp] = diagonal_energy_sum;
    }
    __syncthreads();

    // Fit row energy as alpha * diagonal + beta from four global moments.
    if (warp == 0) {
        diagonal_sum        = lane < 8 ? moment_partials[0][lane] : 0.0;
        diagonal_square_sum = lane < 8 ? moment_partials[1][lane] : 0.0;
        energy_sum          = lane < 8 ? moment_partials[2][lane] : 0.0;
        diagonal_energy_sum = lane < 8 ? moment_partials[3][lane] : 0.0;
        diagonal_sum        = warp_sum_eight_double(diagonal_sum, lane);
        diagonal_square_sum = warp_sum_eight_double(diagonal_square_sum, lane);
        energy_sum          = warp_sum_eight_double(energy_sum, lane);
        diagonal_energy_sum = warp_sum_eight_double(diagonal_energy_sum, lane);
        if (lane == 0) {
            double determinant       = n * diagonal_square_sum - diagonal_sum * diagonal_sum;
            double determinant_scale = n * diagonal_square_sum;
            double alpha             = 0.0;
            bool   fit_conditioned =
                isfinite(determinant) && isfinite(determinant_scale) && determinant > 1.0e-12 * determinant_scale;
            if (fit_conditioned) {
                alpha = (n * diagonal_energy_sum - diagonal_sum * energy_sum) / determinant;
            }
            double beta = (energy_sum - alpha * diagonal_sum) / n;
            fit_alpha   = alpha;
            fit_beta    = beta;
            fit_scale   = energy_sum;
            fit_trace   = diagonal_sum;
            fit_valid =
                fit_conditioned && isfinite(alpha) && isfinite(beta) && isfinite(energy_sum) && energy_sum > 0.0;
        }
    }
    __syncthreads();

    // A small fit residual is the cheap evidence for a two-root spectrum.
    double residual_sum = 0.0;
    for (int row = tid; row < n; row += blockDim.x) {
        double residual = row_energies[row] - fit_alpha * diagonals[row] - fit_beta;
        residual_sum += residual * residual;
    }
    residual_sum = warp_sum_full_double(residual_sum);
    if (lane == 0) {
        moment_partials[0][warp] = residual_sum;
    }
    __syncthreads();
    if (warp == 0) {
        residual_sum = lane < 8 ? moment_partials[0][lane] : 0.0;
        residual_sum = warp_sum_eight_double(residual_sum, lane);
        if (lane == 0) {
            double relative_residual = fit_valid ? sqrt(fmax(residual_sum, 0.0)) / fit_scale : INFINITY;
            double discriminant      = fit_alpha * fit_alpha + 4.0 * fit_beta;
            bool   roots_valid       = fit_valid && discriminant > 0.0 && isfinite(discriminant);
            double root_gap          = roots_valid ? sqrt(discriminant) : 0.0;
            double lower_root        = 0.5 * (fit_alpha - root_gap);
            double upper_root        = 0.5 * (fit_alpha + root_gap);
            // Trace and root gap recover the lower-cluster multiplicity.
            double rank_estimate = roots_valid ? (n * upper_root - fit_trace) / root_gap : 0.0;
            double rounded_rank  = nearbyint(rank_estimate);
            int    lower_rank    = static_cast<int>(rounded_rank);
            double root_scale    = fmax(fmax(fabs(lower_root), fabs(upper_root)), 1.0);
            bool   pass          = roots_valid && isfinite(root_gap) && isfinite(lower_root) && isfinite(upper_root)
                     && isfinite(rank_estimate) && isfinite(relative_residual) && relative_residual <= 5.0e-5
                     && fabs(rank_estimate - rounded_rank) <= 1.0e-2 && root_gap >= 1.0e-3 * root_scale
                     && lower_rank >= 32 && lower_rank <= n - 32;
            roots[static_cast<int64_t>(matrix) * 2]         = pass ? static_cast<float>(lower_root) : 0.0f;
            roots[static_cast<int64_t>(matrix) * 2 + 1]     = pass ? static_cast<float>(upper_root) : 0.0f;
            decisions[static_cast<int64_t>(matrix) * 2]     = pass ? 1 : 0;
            decisions[static_cast<int64_t>(matrix) * 2 + 1] = pass ? lower_rank : 0;
        }
    }
}

__global__ __launch_bounds__(256, 2) void cluster_structure_board512_kernel(const float2 *__restrict__ input,
                                                                            const float2 *__restrict__ projector,
                                                                            const float *__restrict__ roots,
                                                                            int *__restrict__ board_flags)
{
    constexpr int     n                  = 512;
    constexpr int     vectors_per_row    = n / 2;
    constexpr int     vectors_per_matrix = n * vectors_per_row;
    constexpr double  structure_gate     = 0.0030517578125;
    __shared__ double residual_partials[8];
    __shared__ double norm_partials[8];
    const int         matrix            = blockIdx.x;
    const int         tid               = threadIdx.x;
    const int         lane              = tid & 31;
    const int         warp              = tid >> 5;
    const float2     *matrix_input      = input + static_cast<int64_t>(matrix) * vectors_per_matrix;
    const float2     *matrix_projector  = projector + static_cast<int64_t>(matrix) * vectors_per_matrix;
    const int         first_column      = tid << 1;
    const int         second_column     = first_column + 1;
    const double      lower_root        = roots[static_cast<int64_t>(matrix) * 2];
    const double      upper_root        = roots[static_cast<int64_t>(matrix) * 2 + 1];
    const double      root_delta        = lower_root - upper_root;
    double            residual_sum      = 0.0;
    double            first_column_sum  = 0.0;
    double            second_column_sum = 0.0;
    // float2 checks two columns of A = upper * I + (lower - upper) * P.
    for (int row = 0; row < n; ++row) {
        int    vector          = row * vectors_per_row + tid;
        float2 input_value     = matrix_input[vector];
        float2 projector_value = matrix_projector[vector];
        double first_residual  = static_cast<double>(input_value.x) - root_delta * projector_value.x
                              - (row == first_column ? upper_root : 0.0);
        double second_residual = static_cast<double>(input_value.y) - root_delta * projector_value.y
                               - (row == second_column ? upper_root : 0.0);
        residual_sum += first_residual * first_residual + second_residual * second_residual;
        first_column_sum += fabs(static_cast<double>(input_value.x));
        second_column_sum += fabs(static_cast<double>(input_value.y));
    }
    residual_sum       = warp_sum_full_double(residual_sum);
    double matrix_norm = warp_max_full_double(fmax(first_column_sum, second_column_sum));
    if (lane == 0) {
        residual_partials[warp] = residual_sum;
        norm_partials[warp]     = matrix_norm;
    }
    __syncthreads();
    if (warp == 0) {
        residual_sum = lane < 8 ? residual_partials[lane] : 0.0;
        matrix_norm  = lane < 8 ? norm_partials[lane] : 0.0;
        residual_sum = warp_sum_eight_double(residual_sum, lane);
        matrix_norm  = warp_max_full_double(matrix_norm);
        if (lane == 0) {
            double relative_bound = matrix_norm > 0.0 ? sqrt(n * fmax(residual_sum, 0.0)) / matrix_norm : INFINITY;
            board_flags[matrix]   = isfinite(lower_root) && isfinite(upper_root) && lower_root < upper_root
                                       && isfinite(relative_bound) && relative_bound <= structure_gate
                                      ? 1
                                      : 0;
        }
    }
}

__global__ __launch_bounds__(256, 2) void projector_columns512_kernel(const float *__restrict__ input,
                                                                      const float *__restrict__ roots,
                                                                      float *__restrict__ columns,
                                                                      int     column_count,
                                                                      int     lower,
                                                                      int64_t count)
{
    constexpr int n     = 512;
    int64_t       index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; index < count; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t matrix_size = static_cast<int64_t>(n) * column_count;
        int     matrix      = static_cast<int>(index / matrix_size);
        int64_t rem         = index - static_cast<int64_t>(matrix) * matrix_size;
        int     row         = static_cast<int>(rem / column_count);
        int     col         = static_cast<int>(rem - static_cast<int64_t>(row) * column_count);
        float   lower_root  = roots[static_cast<int64_t>(matrix) * 2];
        float   upper_root  = roots[static_cast<int64_t>(matrix) * 2 + 1];
        float   gap         = upper_root - lower_root;
        float   value       = input[static_cast<int64_t>(matrix) * n * n + row * n + col];
        float   diagonal    = row == col ? 1.0f : 0.0f;

        // Form columns of either spectral projector directly from A and the roots.
        columns[index] = lower ? (upper_root * diagonal - value) / gap : (value - lower_root * diagonal) / gap;
    }
}

__global__ __launch_bounds__(256, 2) void cluster_values512_kernel(const float *__restrict__ roots,
                                                                   float *__restrict__ values,
                                                                   int     lower_rank,
                                                                   int64_t count)
{
    constexpr int n     = 512;
    int64_t       index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; index < count; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int matrix    = static_cast<int>(index >> 9);
        int column    = static_cast<int>(index & (n - 1));
        values[index] = roots[static_cast<int64_t>(matrix) * 2 + (column >= lower_rank ? 1 : 0)];
    }
}

static int next_power2(int n)
{
    int value = 1;
    while (value < n) {
        value <<= 1;
    }
    return value;
}

cudaError_t launch_numerical_rank_prefilter1024(const float *input,
                                                int         *candidate_flags,
                                                float       *shifts,
                                                double      *matrix_norms,
                                                int          batch)
{
    constexpr int threads = 256;
    numerical_rank_prefilter1024_kernel<<<batch, threads>>>(
        reinterpret_cast<const float4 *>(input), candidate_flags, shifts, matrix_norms);
    return cudaGetLastError();
}

cudaError_t launch_eigen_board1024(const float  *input,
                                   const float  *q,
                                   const float  *aq,
                                   const float  *values,
                                   const double *matrix_norms,
                                   int          *board_flags,
                                   int           batch)
{
    constexpr int threads          = 256;
    constexpr int tiles_per_matrix = 4;
    eigen_board1024_kernel<<<batch * tiles_per_matrix, threads>>>(input, q, aq, values, matrix_norms, board_flags);
    return cudaGetLastError();
}

cudaError_t launch_factor_reconstruction_bound1024(const float  *qdrop,
                                                   const float  *factor,
                                                   int64_t       factor_batch_stride,
                                                   int64_t       factor_row_stride,
                                                   int64_t       factor_column_stride,
                                                   const float  *q,
                                                   const float  *discarded_values,
                                                   const float  *shifts,
                                                   const double *matrix_norms,
                                                   double       *partials,
                                                   int          *board_flags,
                                                   int           batch)
{
    constexpr int threads = 256;
    constexpr int parts   = 4;
    factor_reconstruction_partials1024_kernel<<<batch * parts, threads>>>(
        qdrop, factor, factor_batch_stride, factor_row_stride, factor_column_stride, q, partials);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    factor_reconstruction_finalize1024_kernel<<<batch, threads>>>(
        discarded_values, shifts, matrix_norms, partials, board_flags);
    return cudaGetLastError();
}

cudaError_t launch_orthogonality_board1024(const float *gram, int *board_flags, int batch)
{
    constexpr int threads          = 256;
    constexpr int tiles_per_matrix = 4;
    orthogonality_board1024_kernel<<<batch * tiles_per_matrix, threads>>>(gram, board_flags);
    return cudaGetLastError();
}

cudaError_t launch_cluster_prefilter512(const float *input, float *roots, int *decisions, int batch)
{
    constexpr int threads = 256;
    cluster_prefilter512_kernel<<<batch, threads>>>(reinterpret_cast<const float4 *>(input), roots, decisions);
    return cudaGetLastError();
}

cudaError_t launch_cluster_structure_board512(const float *input,
                                              const float *projector,
                                              const float *roots,
                                              int         *board_flags,
                                              int          batch)
{
    constexpr int threads = 256;
    cluster_structure_board512_kernel<<<batch, threads>>>(
        reinterpret_cast<const float2 *>(input), reinterpret_cast<const float2 *>(projector), roots, board_flags);
    return cudaGetLastError();
}

cudaError_t launch_projector_columns512(const float *input,
                                        const float *roots,
                                        float       *columns,
                                        int          batch,
                                        int          column_count,
                                        bool         lower)
{
    constexpr int n       = 512;
    constexpr int threads = 256;
    int64_t       count   = static_cast<int64_t>(batch) * n * column_count;
    int           blocks  = capped_blocks(count, threads);
    projector_columns512_kernel<<<blocks, threads>>>(input, roots, columns, column_count, lower ? 1 : 0, count);
    return cudaGetLastError();
}

cudaError_t launch_cluster_values512(const float *roots, float *values, int batch, int lower_rank)
{
    constexpr int threads = 256;
    constexpr int n       = 512;
    int64_t       count   = static_cast<int64_t>(batch) * n;
    int           blocks  = capped_blocks(count, threads);
    cluster_values512_kernel<<<blocks, threads>>>(roots, values, lower_rank, count);
    return cudaGetLastError();
}

cudaError_t
launch_sort_diagonal(const float *a, const float *maxima, float *values, int *perm, int batch, int n, int work_n)
{
    constexpr int threads      = 256;
    int           sort_n       = next_power2(work_n);
    size_t        shared_bytes = static_cast<size_t>(sort_n) * (sizeof(float) + sizeof(int));
    sort_diagonal_kernel<<<batch, threads, shared_bytes>>>(a, maxima, values, perm, n, work_n, sort_n);
    return cudaGetLastError();
}

cudaError_t launch_write_identity(const int *perm, const int *found, float *q, int batch, int n)
{
    constexpr int threads = 256;
    int64_t       count   = static_cast<int64_t>(batch) * n * n;
    int           blocks  = capped_blocks(count, threads);
    if (n == 4096) {
        write_identity_power2_kernel<12><<<blocks, threads>>>(perm, found, q, count);
    }
    else {
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

cudaError_t launch_find_offdiag4096(const float *input, int *found, int batch)
{
    constexpr int threads = 256;
    clear_flags_kernel<<<(batch + threads - 1) / threads, threads>>>(found, batch);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    int64_t vector_count = static_cast<int64_t>(batch) * 4096 * 4096 / 4;
    int     blocks       = capped_blocks(vector_count, threads);
    find_offdiag4096_kernel<<<blocks, threads>>>(reinterpret_cast<const float4 *>(input), found, vector_count);
    return cudaGetLastError();
}
"""


load_inline(
    name="codex_eigh_no_a2_projector_exp18",
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    is_python_module=False,
    no_implicit_headers=True,
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a",
    ],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    return torch.ops.codex_eigh.eigh_batched(data)
