#include <torch/extension.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <iostream>
#include <c10/cuda/CUDAGuard.h>

namespace cg = cooperative_groups;

// ------------------------------------------
// Constant Memory for Gaussian Coefficients
// ------------------------------------------
__constant__ float cGauss[11] = {
    0.001028380123898387f,
    0.0075987582094967365f,
    0.036000773310661316f,
    0.10936068743467331f,
    0.21300552785396576f,
    0.26601171493530273f,
    0.21300552785396576f,
    0.10936068743467331f,
    0.036000773310661316f,
    0.0075987582094967365f,
    0.001028380123898387f
};

// ------------------------------------------
// Block and Shared Memory Dimensions
// ------------------------------------------
#define BLOCK_X 16
#define BLOCK_Y 16
#define HALO    5

#define SHARED_X (BLOCK_X + 2 * HALO)
#define SHARED_Y (BLOCK_Y + 2 * HALO)

// For partial results after horizontal pass
#define CONV_X BLOCK_X
#define CONV_Y SHARED_Y

// ------------------------------------------
// Utility: Safe pixel fetch w/ zero padding
// ------------------------------------------
__device__ __forceinline__ float get_pix_value(
    const float* img, 
    int b, int c, int y, int x,
    int CH, int H, int W
) {
    if (x < 0 || x >= W || y < 0 || y >= H) {
        return 0.0f;
    }
    return img[b * CH * H * W + c * H * W + y * W + x];
}

// ------------------------------------------
// Forward Kernel: Fused SSIM
//  - Two-pass convolution to get mu1, mu2, mu3,
//    sigma1_sq, sigma2_sq, sigma12, etc.
//  - Writes final SSIM map to ssim_map
//  - Optionally writes branch partial derivatives
//    to dl_dmu1, dcs_dmu1, dcs_dsigma1_sq, dcs_dsigma12
// ------------------------------------------
__global__ void fusedssimCUDA(
    int H,
    int W,
    int CH,
    float C1,
    float C2,
    const float* __restrict__ img1,
    const float* __restrict__ img2,
    const float* __restrict__ img3,
    float* __restrict__ luminance_map,
    float* __restrict__ contrast_structure_map,
    float* __restrict__ dl_dmu1,
    float* __restrict__ dl_dmu3,
    float* __restrict__ dcs_dmu1,
    float* __restrict__ dcs_dmu2,
    float* __restrict__ dcs_dsigma1_sq,
    float* __restrict__ dcs_dsigma12
) {
    auto block = cg::this_thread_block();
    const int bIdx   = block.group_index().z;  // batch index
    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;

    // Shared memory for the tile (img1, img2, img3)
    __shared__ float sTile[SHARED_Y][SHARED_X][3];
    // After horizontal pass, store partial sums here
    // xconv[y][x] -> (sumX, sumX^2, sumY, sumY^2, sumXY, sumZ)
    __shared__ float xconv[CONV_Y][CONV_X][6];

    // Each block processes B x C sub-batches. We loop over channels:
    for (int c = 0; c < CH; ++c) {
        // ------------------------------------------------------------
        // 1) Load (img1, img2) tile + halo into shared memory
        // ------------------------------------------------------------
        {
            const int tileSize = SHARED_Y * SHARED_X;
            const int threads = BLOCK_X * BLOCK_Y;
            const int steps = (tileSize + threads - 1) / threads;

            const int tileStartY = block.group_index().y * BLOCK_Y;
            const int tileStartX = block.group_index().x * BLOCK_X;

            for (int s = 0; s < steps; ++s) {
                int tid = s * threads + block.thread_rank();
                if (tid < tileSize) {
                    int local_y = tid / SHARED_X;
                    int local_x = tid % SHARED_X;
                    int gy = tileStartY + local_y - HALO;
                    int gx = tileStartX + local_x - HALO;

                    float X = get_pix_value(img1, bIdx, c, gy, gx, CH, H, W);
                    float Y = get_pix_value(img2, bIdx, c, gy, gx, CH, H, W);
                    float Z = get_pix_value(img3, bIdx, c, gy, gx, CH, H, W);

                    sTile[local_y][local_x][0] = X;
                    sTile[local_y][local_x][1] = Y;
                    sTile[local_y][local_x][2] = Z;
                }
            }
        }
        block.sync();

        // ------------------------------------------------------------
        // 2) Horizontal convolution (11x1) in shared memory
        //    We'll accumulate symmetrical pairs around center.
        // ------------------------------------------------------------
        {
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO;  // skip left halo

            float sumX   = 0.f;
            float sumX2  = 0.f;
            float sumY   = 0.f;
            float sumY2  = 0.f;
            float sumXY  = 0.f;
            float sumZ   = 0.f;

            // #pragma unroll for those 5 pairs
#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                float Xleft  = sTile[ly][lx - d][0];
                float Yleft  = sTile[ly][lx - d][1];
                float Zleft  = sTile[ly][lx - d][2];
                float Xright = sTile[ly][lx + d][0];
                float Yright = sTile[ly][lx + d][1];
                float Zright = sTile[ly][lx + d][2];

                sumX  += (Xleft + Xright) * w;
                sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                sumY  += (Yleft + Yright) * w;
                sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
                sumZ  += (Zleft + Zright) * w;
            }
            // center
            {
                float centerX = sTile[ly][lx][0];
                float centerY = sTile[ly][lx][1];
                float centerZ = sTile[ly][lx][2];
                float wc = cGauss[HALO];
                sumX  += centerX * wc;
                sumX2 += (centerX * centerX) * wc;
                sumY  += centerY * wc;
                sumY2 += (centerY * centerY) * wc;
                sumXY += (centerX * centerY) * wc;
                sumZ  += centerZ * wc;
            }

            // Write out partial sums
            xconv[ly][threadIdx.x][0] = sumX;
            xconv[ly][threadIdx.x][1] = sumX2;
            xconv[ly][threadIdx.x][2] = sumY;
            xconv[ly][threadIdx.x][3] = sumY2;
            xconv[ly][threadIdx.x][4] = sumXY;
            xconv[ly][threadIdx.x][5] = sumZ;

            // Possibly handle second row in same warp
            int ly2 = ly + BLOCK_Y;
            if (ly2 < CONV_Y) {
                sumX   = 0.f; sumX2  = 0.f;
                sumY   = 0.f; sumY2  = 0.f;
                sumXY  = 0.f;
                sumZ   = 0.f;

#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float Xleft  = sTile[ly2][lx - d][0];
                    float Yleft  = sTile[ly2][lx - d][1];
                    float Zleft  = sTile[ly2][lx - d][2];
                    float Xright = sTile[ly2][lx + d][0];
                    float Yright = sTile[ly2][lx + d][1];
                    float Zright = sTile[ly2][lx + d][2];

                    sumX  += (Xleft + Xright) * w;
                    sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                    sumY  += (Yleft + Yright) * w;
                    sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                    sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
                    sumZ  += (Zleft + Zright) * w;
                }
                // center
                {
                    float cx = sTile[ly2][lx][0];
                    float cy = sTile[ly2][lx][1];
                    float cz = sTile[ly2][lx][2];
                    float wc = cGauss[HALO];
                    sumX  += cx * wc;
                    sumX2 += (cx * cx) * wc;
                    sumY  += cy * wc;
                    sumY2 += (cy * cy) * wc;
                    sumXY += (cx * cy) * wc;
                    sumZ  += cz * wc;
                }
                xconv[ly2][threadIdx.x][0] = sumX;
                xconv[ly2][threadIdx.x][1] = sumX2;
                xconv[ly2][threadIdx.x][2] = sumY;
                xconv[ly2][threadIdx.x][3] = sumY2;
                xconv[ly2][threadIdx.x][4] = sumXY;
                xconv[ly2][threadIdx.x][5] = sumZ;
            }
        }
        block.sync();

        // ------------------------------------------------------------
        // 3) Vertical convolution (1x11) + final SSIM
        // ------------------------------------------------------------
        {
            int ly = threadIdx.y + HALO;
            int lx = threadIdx.x;

            float out0 = 0.f, out1 = 0.f, out2 = 0.f, out3 = 0.f, out4 = 0.f, out5 = 0.f;

#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                float* top = xconv[ly - d][lx];
                float* bot = xconv[ly + d][lx];

                out0 += (top[0] + bot[0]) * w;
                out1 += (top[1] + bot[1]) * w;
                out2 += (top[2] + bot[2]) * w;
                out3 += (top[3] + bot[3]) * w;
                out4 += (top[4] + bot[4]) * w;
                out5 += (top[5] + bot[5]) * w;
            }
            // center
            {
                float wC = cGauss[HALO];
                float* ctr = xconv[ly][lx];
                out0 += ctr[0] * wC;
                out1 += ctr[1] * wC;
                out2 += ctr[2] * wC;
                out3 += ctr[3] * wC;
                out4 += ctr[4] * wC;
                out5 += ctr[5] * wC;
            }

            if (pix_x < W && pix_y < H) {
                float mu1 = out0;
                float mu2 = out2;
                float mu3 = out5;
                float mu1_sq = mu1 * mu1;
                float mu2_sq = mu2 * mu2;

                float mu3_sq = mu3 * mu3;

                float sigma1_sq = fmaxf(0.0f, out1 - mu1_sq);
                float sigma2_sq = fmaxf(0.0f, out3 - mu2_sq);
                float sigma12   = out4 - mu1 * mu2;

                // luminance denominator
                float A = mu1_sq + mu3_sq + C1;
                // contrast-structure denominator
                float B = sigma1_sq + sigma2_sq + C2;
                // luminance numerator
                float C_ = 2.f * mu1 * mu3 + C1;
                // contrast-structure numerator
                float D_ = 2.f * sigma12 + C2;

                int global_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                luminance_map[global_idx] = C_ / A;
                contrast_structure_map[global_idx] = D_ / B;

                if (dl_dmu1) {
                    // Branch partials for additive decoupled backward:
                    // dL/dx = conv(Gmu) + 2*x*conv(Gs1) + y*conv(Gs12),
                    // where:
                    //   Gmu  = dL/dl * dl/dmu1 + dL/dcs * dcs/dmu1
                    //   Gs1  = dL/dcs * dcs/dsigma1_sq
                    //   Gs12 = dL/dcs * dcs/dsigma12
                    float dl_mu1 = (2.f * mu3) / A - (2.f * mu1 * C_) / (A * A);
                    float dl_mu3 = (2.f * mu1) / A - (2.f * mu3 * C_) / (A * A);
                    float dcs_mu1 = (-2.f * mu2) / B + (2.f * mu1 * D_) / (B * B);
                    float dcs_mu2 = (-2.f * mu1) / B + (2.f * mu2 * D_) / (B * B);
                    float dcs_s1 = -D_ / (B * B);
                    float dcs_s12 = 2.f / B;

                    dl_dmu1[global_idx]         = dl_mu1;
                    dl_dmu3[global_idx]         = dl_mu3;
                    dcs_dmu1[global_idx]        = dcs_mu1;
                    dcs_dmu2[global_idx]        = dcs_mu2;
                    dcs_dsigma1_sq[global_idx]  = dcs_s1;
                    dcs_dsigma12[global_idx]    = dcs_s12;
                }
            }
        }
    }
}

// ------------------------------------------
// Backward Kernel: Apply chain rule to get
//    dL/d(img1) from branch partial derivatives and
//    upstream gradients dL/dluminance_map and
//    dL/dcontrast_structure_map.
// ------------------------------------------
__global__ void fusedssim_backwardCUDA(
    int H,
    int W,
    int CH,
    float C1,
    float C2,
    const float* __restrict__ img1,
    const float* __restrict__ img2,
    const float* __restrict__ dL_dluminance_map,
    const float* __restrict__ dL_dcontrast_structure_map,
    float* __restrict__ dL_dimg1,
    const float* __restrict__ dl_dmu1,
    const float* __restrict__ dcs_dmu1,
    const float* __restrict__ dcs_dsigma1_sq,
    const float* __restrict__ dcs_dsigma12
) {
    auto block = cg::this_thread_block();

    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x; 
    const int num_pix = H * W;
    const int bIdx   = block.group_index().z;

    // Shared memory for fused chain-rule terms:
    // [0]: Gmu, [1]: Gs1, [2]: Gs12
    __shared__ float sData[3][SHARED_Y][SHARED_X];
    __shared__ float sScratch[CONV_Y][CONV_X][3];

    for (int c = 0; c < CH; ++c) {
        float p1 = 0.f, p2 = 0.f;
        if (pix_x < W && pix_y < H) {
            p1 = get_pix_value(img1, bIdx, c, pix_y, pix_x, CH, H, W);
            p2 = get_pix_value(img2, bIdx, c, pix_y, pix_x, CH, H, W);
        }

        // (1) Load + fuse multiplication
        {
            const int start_y = block.group_index().y * BLOCK_Y;
            const int start_x = block.group_index().x * BLOCK_X;

            int tid = threadIdx.y * blockDim.x + threadIdx.x;
            int warp_id = tid / 32;
            int lane_id = tid % 32;
            int totalThreads = BLOCK_X * BLOCK_Y;
            int num_warps = (totalThreads + 31) / 32;

            for (int row = warp_id; row < SHARED_Y; row += num_warps) {
                int gy = start_y + row - HALO;
                for (int col = lane_id; col < SHARED_X; col += 32) {
                    int gx = start_x + col - HALO;

                    float chain_luminance = get_pix_value(dL_dluminance_map,         bIdx, c, gy, gx, CH, H, W);
                    float chain_contrast  = get_pix_value(dL_dcontrast_structure_map, bIdx, c, gy, gx, CH, H, W);

                    float v_dl_mu1  = get_pix_value(dl_dmu1,         bIdx, c, gy, gx, CH, H, W);
                    float v_dcs_mu1 = get_pix_value(dcs_dmu1,        bIdx, c, gy, gx, CH, H, W);
                    float v_dcs_s1  = get_pix_value(dcs_dsigma1_sq,  bIdx, c, gy, gx, CH, H, W);
                    float v_dcs_s12 = get_pix_value(dcs_dsigma12,    bIdx, c, gy, gx, CH, H, W);

                    float g_mu  = chain_luminance * v_dl_mu1 + chain_contrast * v_dcs_mu1;
                    float g_s1  = chain_contrast * v_dcs_s1;
                    float g_s12 = chain_contrast * v_dcs_s12;

                    sData[0][row][col] = g_mu;
                    sData[1][row][col] = g_s1;
                    sData[2][row][col] = g_s12;
                }
            }
        }
        block.sync();

        // (2) Horizontal pass
        {
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO;

            for (int pass = 0; pass < 2; ++pass) {
                int yy = ly + pass * BLOCK_Y;
                if (yy < CONV_Y) {
                    float accum0 = 0.f, accum1 = 0.f, accum2 = 0.f;

#pragma unroll
                    for (int d = 1; d <= HALO; ++d) {
                        float w = cGauss[HALO - d];
                        float left0  = sData[0][yy][lx - d];
                        float left1  = sData[1][yy][lx - d];
                        float left2  = sData[2][yy][lx - d];

                        float right0 = sData[0][yy][lx + d];
                        float right1 = sData[1][yy][lx + d];
                        float right2 = sData[2][yy][lx + d];

                        accum0 += (left0 + right0) * w;
                        accum1 += (left1 + right1) * w;
                        accum2 += (left2 + right2) * w;
                    }
                    // center
                    {
                        float wc = cGauss[HALO];
                        float c0 = sData[0][yy][lx];
                        float c1 = sData[1][yy][lx];
                        float c2 = sData[2][yy][lx];
                        accum0 += c0 * wc;
                        accum1 += c1 * wc;
                        accum2 += c2 * wc;
                    }

                    sScratch[yy][threadIdx.x][0] = accum0;
                    sScratch[yy][threadIdx.x][1] = accum1;
                    sScratch[yy][threadIdx.x][2] = accum2;
                }
            }
        }
        block.sync();

        // (3) Vertical pass -> finalize dL/d(img1)
        if (pix_x < W && pix_y < H) {
            int ly = threadIdx.y + HALO;
            int lx = threadIdx.x;

            float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f;

#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                float* top = sScratch[ly - d][lx];
                float* bot = sScratch[ly + d][lx];

                sum0 += (top[0] + bot[0]) * w;
                sum1 += (top[1] + bot[1]) * w;
                sum2 += (top[2] + bot[2]) * w;
            }
            // center
            {
                float wc = cGauss[HALO];
                float* ctr = sScratch[ly][lx];
                sum0 += ctr[0] * wc;
                sum1 += ctr[1] * wc;
                sum2 += ctr[2] * wc;
            }

            // final accumulation
            float dL_dpix = sum0 + (2.f * p1) * sum1 + (p2) * sum2;

            int out_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
            dL_dimg1[out_idx] = dL_dpix;
        }
        block.sync();
    }
}

// ------------------------------------------
// PyTorch Interface (Forward)
//   Returns (luminance_map, contrast_structure_map,
//            dl_dmu1, dl_dmu3, dcs_dmu1, dcs_dmu2, dcs_dsigma1_sq, dcs_dsigma12).
//   If train=false, derivative Tensors are empty.
// ------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &img3,
    bool train
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    int B  = img1.size(0);
    int CH = img1.size(1);
    int H  = img1.size(2);
    int W  = img1.size(3);

    // Launch config
    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y,
              B);
    dim3 block(BLOCK_X, BLOCK_Y); 

    // Output SSIM map
    auto luminance_map = torch::zeros_like(img1, img1.options()).contiguous();
    auto contrast_structure_map = torch::zeros_like(img1, img1.options()).contiguous();

    // Optionally allocate branch derivative tensors
    auto dl_dmu1         = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dl_dmu3         = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dcs_dmu1        = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dcs_dmu2        = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dcs_dsigma1_sq  = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dcs_dsigma12    = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());

    fusedssimCUDA<<<grid, block>>>(
        H, W, CH, C1, C2,
        img1.contiguous().data_ptr<float>(),
        img2.contiguous().data_ptr<float>(),
        img3.contiguous().data_ptr<float>(),
        luminance_map.data_ptr<float>(),
        contrast_structure_map.data_ptr<float>(),
        train ? dl_dmu1.data_ptr<float>()        : nullptr,
        train ? dl_dmu3.data_ptr<float>()        : nullptr,
        train ? dcs_dmu1.data_ptr<float>()       : nullptr,
        train ? dcs_dmu2.data_ptr<float>()       : nullptr,
        train ? dcs_dsigma1_sq.data_ptr<float>() : nullptr,
        train ? dcs_dsigma12.data_ptr<float>()   : nullptr
    );

    return std::make_tuple(
        luminance_map,
        contrast_structure_map,
        dl_dmu1,
        dl_dmu3,
        dcs_dmu1,
        dcs_dmu2,
        dcs_dsigma1_sq,
        dcs_dsigma12
    );
}

// ------------------------------------------
// PyTorch Interface (Backward)
//   Takes the gradient wrt the SSIM map and
//   the partial derivatives from forward;
//   returns dL/d(img1), dL/d(img2), dL/d(img3).
// ------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim_backward(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &img3,
    torch::Tensor &dL_dluminance_map,
    torch::Tensor &dL_dcontrast_structure_map,
    torch::Tensor &dl_dmu1,
    torch::Tensor &dl_dmu3,
    torch::Tensor &dcs_dmu1,
    torch::Tensor &dcs_dmu2,
    torch::Tensor &dcs_dsigma1_sq,
    torch::Tensor &dcs_dsigma12
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    int B  = img1.size(0);
    int CH = img1.size(1);
    int H  = img1.size(2);
    int W  = img1.size(3);

    const bool need_img1 = (dl_dmu1.numel() != 0) || (dcs_dmu1.numel() != 0);
    const bool need_img2 = (dcs_dmu2.numel() != 0);
    const bool need_img3 = (dl_dmu3.numel() != 0);

    auto dL_dimg1 = need_img1 ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dL_dimg2 = need_img2 ? torch::zeros_like(img2) : torch::empty({0}, img2.options());
    auto dL_dimg3 = need_img3 ? torch::zeros_like(img3) : torch::empty({0}, img3.options());

    auto zeros = torch::zeros_like(img1);

    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y,
              B);
    dim3 block(BLOCK_X, BLOCK_Y);

    if (need_img1) {
        fusedssim_backwardCUDA<<<grid, block>>>(
            H, W, CH, C1, C2,
            img1.contiguous().data_ptr<float>(),
            img2.contiguous().data_ptr<float>(),
            dL_dluminance_map.contiguous().data_ptr<float>(),
            dL_dcontrast_structure_map.contiguous().data_ptr<float>(),
            dL_dimg1.data_ptr<float>(),
            dl_dmu1.contiguous().data_ptr<float>(),
            dcs_dmu1.contiguous().data_ptr<float>(),
            dcs_dsigma1_sq.contiguous().data_ptr<float>(),
            dcs_dsigma12.contiguous().data_ptr<float>()
        );
    }

    // For img2, only the contrast-structure branch contributes (mu2 / sigma2_sq / sigma12).
    // We compute the gradient wrt img2 by swapping (img1, img2) in the backward kernel:
    //   dL/d(img2) = conv(Gmu2) + 2*img2*conv(Gs2) + img1*conv(Gs12)
    // where:
    //   Gmu2 = dL/dcs * dcs/dmu2
    //   Gs2  = dL/dcs * dcs/dsigma2_sq   (same as dcs/dsigma1_sq)
    //   Gs12 = dL/dcs * dcs/dsigma12
    if (need_img2) {
        fusedssim_backwardCUDA<<<grid, block>>>(
            H, W, CH, C1, C2,
            img2.contiguous().data_ptr<float>(),  // p1 := img2
            img1.contiguous().data_ptr<float>(),  // p2 := img1
            zeros.contiguous().data_ptr<float>(), // no luminance contribution
            dL_dcontrast_structure_map.contiguous().data_ptr<float>(),
            dL_dimg2.data_ptr<float>(),
            zeros.contiguous().data_ptr<float>(),       // dl_dmu1 unused
            dcs_dmu2.contiguous().data_ptr<float>(),     // dcs/dmu2
            dcs_dsigma1_sq.contiguous().data_ptr<float>(), // dcs/dsigma2_sq == dcs/dsigma1_sq
            dcs_dsigma12.contiguous().data_ptr<float>()
        );
    }

    // For img3, only the luminance branch contributes (mu3).
    if (need_img3) {
        fusedssim_backwardCUDA<<<grid, block>>>(
            H, W, CH, C1, C2,
            img3.contiguous().data_ptr<float>(),
            img2.contiguous().data_ptr<float>(),
            dL_dluminance_map.contiguous().data_ptr<float>(),
            dL_dcontrast_structure_map.contiguous().data_ptr<float>(),
            dL_dimg3.data_ptr<float>(),
            dl_dmu3.contiguous().data_ptr<float>(),
            zeros.data_ptr<float>(),
            zeros.data_ptr<float>(),
            zeros.data_ptr<float>()
        );
    }

    return std::make_tuple(dL_dimg1, dL_dimg2, dL_dimg3);
}