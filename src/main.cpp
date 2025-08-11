#include "fft_cpu.hpp"
#include "fft_gpu.hpp"

#include <CLI/CLI.hpp>
#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <fstream>

#include "wav_signal.hpp"

constexpr double M_PI = 3.14159265358979323846;

struct TimedResult
{
    std::vector<std::complex<double>> data;
    double ms = 0.0;
    bool ok = true;
    std::string err;
};

static std::vector<std::complex<double>> make_signal( std::size_t n )
{
    std::vector<std::complex<double>> x( n );
    const double twoPi = 2.0 * M_PI;
    
    for ( std::size_t k = 0; k < n; ++k )
    {
        double t = static_cast<double>( k ) / static_cast<double>( n );
        double re = std::cos( twoPi * t ) + 0.1 * std::cos( 2 * twoPi * t );
        double im = std::sin( twoPi * t ) + 0.1 * std::sin( 2 * twoPi * t );
        x[k] = {re, im};
    }
    return x;
}

static double max_abs_error(const std::vector<std::complex<double>>& a,
                            const std::vector<std::complex<double>>& b)
{
    double m = 0.0;
    const std::size_t n = a.size();
    for ( std::size_t i = 0; i < n; ++i )
    {
        double er = std::abs(a[i].real() - b[i].real());
        double ei = std::abs(a[i].imag() - b[i].imag());
        m = std::max( m, std::max( er, ei ));
    }
    return m;
}

int main( int argc, char** argv )
{
    CLI::App app{"fft_cli — test FFTW (CPU) i cuFFT (GPU)"};

    std::size_t n = 1 << 20;        // 1,048,576
    bool inverse = false;
    int repeats = 1;
    bool run_cpu = true;
    bool run_gpu = true;
    bool check = true;
    std::string wav_path;

    app.add_option( "-n,--size", n, "Basic signal length" );
    app.add_flag( "--inverse", inverse, "Inverse transform" );
    app.add_option( "-r,--repeats", repeats, "Number of repeats of measurement" )->check( CLI::PositiveNumber );

    app.add_flag_function( "--cpu-only", [&](size_t) {run_cpu = true; run_gpu = false;} );

    app.add_flag_function( "--gpu-only", [&](size_t) {run_cpu = false; run_gpu = true;} );

    app.add_flag_function( "--no-check", [&](size_t) {check = false;} );

    app.add_option("-f,--file", wav_path, "Path to WAV file");

    CLI11_PARSE(app, argc, argv);

    std::vector<std::complex<double>> input;
    if (wav_path.empty()) {
        std::cout << "Missed WAV file path\n";
        input = make_signal(n);
    } else {
        std::cout << "WAV path: " << wav_path << "\n";
        // std::vector<std::complex<double>> signal_wav = import_wav(wav_path);
        input = import_wav(wav_path);
    }

    std::cout << "N = " << n
              << " | " << (inverse ? "IFFT" : "FFT")
              << " | repeats = " << repeats
              << " | modes: " << (run_cpu ? "CPU " : "") << (run_gpu ? "GPU" : "")
              << "\n";

    // auto input = make_signal(n);
    // input = import_wav(wav_path);

    TimedResult cpu, gpu;

    if ( run_cpu )
    {
        std::vector<std::complex<double>> out;
        double best_ms = std::numeric_limits<double>::infinity();
        for ( int i = 0; i < repeats; ++i )
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            cpu_fft( input, out, inverse );
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>( t1 - t0 ).count();
            best_ms = std::min( best_ms, ms );
        }
        cpu.data = std::move( out );
        cpu.ms = best_ms;
        std::cout << "[CPU] best time = " << cpu.ms << " ms\n";
    }

    if ( run_gpu )
    {
        std::vector<std::complex<double>> out;
        double best_ms = std::numeric_limits<double>::infinity();
        std::string err;
        for ( int i = 0; i < repeats; ++i )
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            bool ok = gpu_fft( input, out, inverse, err );
            auto t1 = std::chrono::high_resolution_clock::now();
            if ( !ok )
            {
                gpu.ok = false;
                gpu.err = err;
                break;
            }
            double ms = std::chrono::duration<double, std::milli>( t1 - t0 ).count();
            best_ms = std::min( best_ms, ms );
        }
        if ( gpu.ok )
        {
            gpu.data = std::move( out );
            gpu.ms = best_ms;
            std::cout << "[GPU] best time = " << gpu.ms << " ms\n";
        }
        else
        {
            std::cerr << "[GPU] error: " << gpu.err << "\n";
        }
    }

    if ( check && run_cpu && run_gpu && gpu.ok )
    {
        double err = max_abs_error( cpu.data, gpu.data );
        std::cout << "[COMPARE] max |diff| = " << err << "\n";
    }

    std::cout << "Done.\n";
    return 0;
}
