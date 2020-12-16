#!/usr/bin/python3
import numpy as np
from skimage import io, transform, data, color
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
import math
import PIL
import threading
import time
from scipy import signal
from numba import cuda, float32



# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16

@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp



laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# input: single channel img; output: 2 channel img, as the gradient field
def differentiate(inputImg):
    fdk1 = np.array([1, -1])
    fdk1 = fdk1.reshape(1, 2)
    fdk2 = fdk1.reshape(2, 1)
    Ix = signal.convolve2d(inputImg, fdk1, mode='same')
    Iy = signal.convolve2d(inputImg, fdk2, mode='same')
    outputImg = np.dstack([Ix, Iy])
    return outputImg


# input: single channel img; output: single channel img
def compute_divergence(inputImg):
    gradientImg = differentiate(inputImg)
    Ix = gradientImg[:, :, 0]
    Iy = gradientImg[:, :, 1]
    gradientIx = differentiate(Ix)
    gradientIy = differentiate(Iy)
    Ixx = gradientIx[:, :, 0]
    Iyy = gradientIy[:, :, 1]
    divI = Ixx + Iyy
    return divI


# input: double channel img; output: single channel img
def grad2divergence(inputImg):
    Ix = inputImg[:, :, 0]
    Iy = inputImg[:, :, 1]
    gradientIx = differentiate(Ix)
    gradientIy = differentiate(Iy)
    Ixx = gradientIx[:, :, 0]
    Iyy = gradientIy[:, :, 1]
    divI = Ixx + Iyy
    return divI

# input: single channel img; out put: filtered img with same shape
def laplacian_filtering(inputImg):
    global laplacian_kernel
    outputImg = signal.convolve2d(inputImg, laplacian_kernel, mode='same')
    return outputImg


def enforce_boundary_condition(inputImg, boundaryMask, boundaryImg):
    reverseBM = 1 - boundaryMask
    return inputImg * reverseBM + boundaryImg * boundaryMask


def cgd(div_grad_I, primary_img, e, N, alpha):
    #  Initialization
    Istar = primary_img
    # Istar = np.zeros(primary_img.shape)
    r = div_grad_I - laplacian_filtering(Istar) + alpha * (primary_img - Istar)
    d = r
    delta_new = np.sum(r * r)
    n = 0
    #  cgd iteration
    while np.sum(r * r) > e * e and n < N:
        q = laplacian_filtering(d) + alpha * (primary_img - Istar)
        eta = delta_new / np.sum(d * q)
        Istar = (Istar + eta * d)
        r = r - eta * q
        delta_old = delta_new
        delta_new = np.sum(r * r)
        beta = delta_new / delta_old
        d = r + beta * d + alpha * (primary_img - Istar)
        n = n + 1
        print(n)

    print("loop end when : n = ", n)
    return Istar

def my_cgd(A, b, x_init, tol, N):
    r = b - np.matmul(A, x_init)
    s = np.matmul(A.T, r)
    # s = np.zeros_like(A.T)
    # fast_matmul(A.T, r, s)
    p = s
    gamma = np.sum(s * s)
    x = x_init
    n = 0

    # q = np.zeros_like(A)
    while np.sum(r * r) > tol * tol and n < N:
        q = np.matmul(A, p)
        # fast_matmul(A, p, q)
        alpha = gamma / np.sum(q * q)
        x = x + alpha * p
        r = r - alpha * q
        s = np.matmul(A.T, r)
        # fast_matmul(A.T, r, s)
        gamma_old = gamma
        gamma = np.sum(s * s)
        beta = gamma / gamma_old
        p = s + beta * p
        n = n + 1
        print(n)

    print("loop end when : n = ", n)
    return x

def main():

    input_Gx = io.imread("sponza_dx.jpg")
    input_Gy = io.imread("sponza_dy.jpg")
    input_P = io.imread("sponza_throughput.jpg")
    input_Gx = input_Gx / 255
    input_Gy = input_Gy / 255
    input_P = input_P / 255

    final_result = np.zeros_like(input_P)

    chunk_size = 100
    chunk_y = 0
    # y_lb = 180
    # y_ub = 280
    # x_lb = 550
    # x_ub = 650
    # y_lb = 200
    # y_ub = 250
    # x_lb = 550
    # x_ub = 600

    # Gx = input_Gx[y_lb:y_ub, x_lb:x_ub, :]
    # Gy = input_Gy[y_lb:y_ub, x_lb:x_ub, :]
    # P = input_P[y_lb:y_ub, x_lb:x_ub, :]
    #
    # pixel_count = P.shape[0] * P.shape[1]
    #
    # e = 0.0001
    # N = 100
    # alpha = 1

    # chunk_result = np.zeros([int(P.shape[0]), int(P.shape[1]), 3])
    # for i in range(0, 3):
    #     # grad_img = np.dstack([Gx[:, :, i], Gy[:, :, i]])
    #     # divI = grad2divergence(grad_img)
    #     # result = cgd(divI, P[:, :, i], e, N, alpha)
    #     # final_result[:, :, i] = result
    #     A = np.zeros([3 * pixel_count, pixel_count], dtype='float32')
    #     b = np.zeros([3 * pixel_count, 1], dtype='float32')
    #     # fill up A and b
    #     for y in range(0, P.shape[0]):
    #         for x in range(0, P.shape[1]):
    #             Gx_cur = Gx[y, x, i]
    #             Gy_cur = Gy[y, x, i]
    #             vec_idx_for_XY = y * P.shape[1] + x
    #             # fill a line in A for Gx_cur
    #             line_num = vec_idx_for_XY
    #             b[line_num] = Gx_cur
    #             if x == 0:
    #                 A[line_num, vec_idx_for_XY] = 1
    #             else:
    #                 A[line_num, vec_idx_for_XY] = 1
    #                 A[line_num, vec_idx_for_XY - 1] = -1
    #
    #             # fill a line in A for Gy_cur
    #             line_num = vec_idx_for_XY + pixel_count
    #             b[line_num] = Gy_cur
    #             if y == 0:
    #                 A[line_num, vec_idx_for_XY] = 1
    #             else:
    #                 A[line_num, vec_idx_for_XY] = 1
    #                 A[line_num, (y - 1) * P.shape[1] + x] = -1
    #
    #             # fill a line in A for P_cur
    #             line_num = vec_idx_for_XY + 2 * pixel_count
    #             b[line_num] = P[y, x, i] * alpha
    #             A[line_num, vec_idx_for_XY] = alpha
    #
    #     x_init = np.zeros([pixel_count, 1])
    #     x_output = my_cgd(A, b, x_init, e, N)
    #     chunk_result[:, :, i] = np.reshape(x_output, (P.shape[0], P.shape[1]), order='C')
    # final_result = chunk_result

    while chunk_y * chunk_size < input_P.shape[0]:
        y_lb = chunk_y * chunk_size
        if (chunk_y + 1) * chunk_size > input_P.shape[0]:
            y_ub = input_P.shape[0]
        else:
            y_ub = (chunk_y + 1) * chunk_size
        chunk_x = 0
        while chunk_x * chunk_size < input_P.shape[1]:
            x_lb = chunk_x * chunk_size
            if (chunk_x + 1) * chunk_size > input_P.shape[1]:
                x_ub = input_P.shape[1]
            else:
                x_ub = (chunk_x + 1) * chunk_size

            Gx = input_Gx[y_lb:y_ub, x_lb:x_ub, :]
            Gy = input_Gy[y_lb:y_ub, x_lb:x_ub, :]
            P = input_P[y_lb:y_ub, x_lb:x_ub, :]

            print("Calculating chunk, y: " + str(chunk_y) + " x: " + str(chunk_x))

            pixel_count = P.shape[0] * P.shape[1]

            e = 0.001
            N = 100
            alpha = 1

            chunk_result = np.zeros([int(P.shape[0]), int(P.shape[1]), 3])
            for i in range(0, 3):
                # grad_img = np.dstack([Gx[:, :, i], Gy[:, :, i]])
                # divI = grad2divergence(grad_img)
                # result = cgd(divI, P[:, :, i], e, N, alpha)
                # final_result[:, :, i] = result
                A = np.zeros([3 * pixel_count, pixel_count], dtype='float32')
                b = np.zeros([3 * pixel_count, 1], dtype='float32')
                # fill up A and b
                for y in range(0, P.shape[0]):
                    for x in range(0, P.shape[1]):
                        Gx_cur = Gx[y, x, i]
                        Gy_cur = Gy[y, x, i]
                        vec_idx_for_XY = y * P.shape[1] + x
                        # fill a line in A for Gx_cur
                        line_num = vec_idx_for_XY
                        b[line_num] = Gx_cur
                        if x == 0:
                            A[line_num, vec_idx_for_XY] = 1
                        else:
                            A[line_num, vec_idx_for_XY] = 1
                            A[line_num, vec_idx_for_XY - 1] = -1

                        # fill a line in A for Gy_cur
                        line_num = vec_idx_for_XY + pixel_count
                        b[line_num] = Gy_cur
                        if y == 0:
                            A[line_num, vec_idx_for_XY] = 1
                        else:
                            A[line_num, vec_idx_for_XY] = 1
                            A[line_num, (y - 1) * P.shape[1] + x] = -1

                        # fill a line in A for P_cur
                        line_num = vec_idx_for_XY + 2 * pixel_count
                        b[line_num] = P[y, x, i] * alpha
                        A[line_num, vec_idx_for_XY] = alpha

                x_init = np.zeros([pixel_count, 1])
                x_output = my_cgd(A, b, x_init, e, N)
                chunk_result[:, :, i] = np.reshape(x_output, (P.shape[0], P.shape[1]), order='C')

            final_result[y_lb:y_ub, x_lb:x_ub, :] = chunk_result

            chunk_x = chunk_x + 1
        chunk_y = chunk_y + 1

    final_result = final_result * 255
    final_result = final_result.astype(np.int)
    final_result = np.clip(final_result, 0, 255)

    io.imsave("reconstruction.jpg", final_result)





if __name__ == "__main__":
    main()