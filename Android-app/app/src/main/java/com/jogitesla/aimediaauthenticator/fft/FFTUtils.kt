package com.jogitesla.aimediaauthenticator.fft

import kotlin.math.*

object FFTUtils {

    // Complex number representation
    data class Complex(val re: Double, val im: Double) {
        operator fun plus(other: Complex) =
            Complex(re + other.re, im + other.im)

        operator fun minus(other: Complex) =
            Complex(re - other.re, im - other.im)

        operator fun times(other: Complex) =
            Complex(
                re * other.re - im * other.im,
                re * other.im + im * other.re
            )

        fun magnitude() = sqrt(re * re + im * im)
    }

    // 1D FFT (Cooley–Tukey, recursive)
    fun fft1D(x: Array<Complex>): Array<Complex> {
        val n = x.size
        if (n == 1) return arrayOf(x[0])

        val even = Array(n / 2) { x[2 * it] }
        val odd = Array(n / 2) { x[2 * it + 1] }

        val fftEven = fft1D(even)
        val fftOdd = fft1D(odd)

        val result = Array(n) { Complex(0.0, 0.0) }

        for (k in 0 until n / 2) {
            val angle = -2.0 * Math.PI * k / n
            val wk = Complex(cos(angle), sin(angle))
            val t = wk * fftOdd[k]

            result[k] = fftEven[k] + t
            result[k + n / 2] = fftEven[k] - t
        }

        return result
    }

    // 2D FFT (rows → columns)
    fun fft2D(input: Array<Array<Double>>): Array<Array<Double>> {
        val h = input.size
        val w = input[0].size

        // FFT rows
        val rowFFT = Array(h) { y ->
            fft1D(Array(w) { x ->
                Complex(input[y][x], 0.0)
            })
        }

        // FFT columns
        val result = Array(h) { Array(w) { 0.0 } }

        for (x in 0 until w) {
            val col = Array(h) { y -> rowFFT[y][x] }
            val fftCol = fft1D(col)

            for (y in 0 until h) {
                result[y][x] = fftCol[y].magnitude()
            }
        }

        return result
    }
}
