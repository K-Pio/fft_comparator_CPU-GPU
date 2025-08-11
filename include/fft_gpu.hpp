#pragma once
#include <complex>
#include <string>
#include <vector>

// True if success, else set 'err'
bool gpu_fft( const std::vector<std::complex<double>>& in,
             std::vector<std::complex<double>>& out,
             bool inverse,
             std::string& err );
