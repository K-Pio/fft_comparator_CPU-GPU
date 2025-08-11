#pragma once
#include <complex>
#include <vector>

void cpu_fft( const std::vector<std::complex<double>>& in,
             std::vector<std::complex<double>>& out,
             bool inverse );
