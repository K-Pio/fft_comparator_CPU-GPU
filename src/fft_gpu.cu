#include "fft_gpu.hpp"
// #pragma warning(disable:4505)
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <sstream>

static const char* cudaErrToStr( cudaError_t e )
{
    return cudaGetErrorString( e );
}

static const char* cufftErrToStr( cufftResult r )
{
    switch ( r ) 
    {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        default: return "CUFFT_UNKNOWN_ERROR";
    }
}

bool gpu_fft(const std::vector<std::complex<double>>& in,
             std::vector<std::complex<double>>& out,
             bool inverse,
             std::string& err)
{
    out.resize( in.size() );
    const int n = static_cast<int>( in.size() );

    // Host bufor w formacie cuFFT (interleaved)
    std::vector<cufftDoubleComplex> h_in( n ), h_out( n );
    for ( int i = 0; i < n; ++i )
    {
        h_in[i].x = in[i].real();
        h_in[i].y = in[i].imag();
    }

    cufftDoubleComplex *d_in = nullptr, *d_out = nullptr;
    cudaError_t cerr;
    cufftResult r;

    cerr = cudaMalloc( &d_in,  sizeof( cufftDoubleComplex ) * n );
    if ( cerr != cudaSuccess ) 
    {
        err = std::string("cudaMalloc d_in: ") + cudaErrToStr( cerr ); 
        return false;
    }
    cerr = cudaMalloc( &d_out, sizeof( cufftDoubleComplex ) * n );
    if ( cerr != cudaSuccess )
    {
        err = std::string("cudaMalloc d_out: ") + cudaErrToStr( cerr ); cudaFree( d_in ); 
        return false;
    }

    cerr = cudaMemcpy( d_in, h_in.data(), sizeof( cufftDoubleComplex ) * n, cudaMemcpyHostToDevice );
    if ( cerr != cudaSuccess )
    {
        err = std::string("cudaMemcpy H2D: ") + cudaErrToStr( cerr );
        cudaFree( d_in ); cudaFree( d_out ); 
        return false;
    }

    cufftHandle plan;
    r = cufftPlan1d( &plan, n, CUFFT_Z2Z, 1 );
    if ( r != CUFFT_SUCCESS )
    {
        err = std::string("cufftPlan1d: ") + cufftErrToStr( r );
        cudaFree( d_in ); cudaFree( d_out );
        return false;
    }

    r = cufftExecZ2Z( plan, d_in, d_out, inverse ? CUFFT_INVERSE : CUFFT_FORWARD );
    if ( r != CUFFT_SUCCESS )
    {
        err = std::string("cufftExecZ2Z: ") + cufftErrToStr( r );
        cufftDestroy( plan ); cudaFree( d_in ); cudaFree( d_out );
        return false;
    }

    cerr = cudaMemcpy( h_out.data(), d_out, sizeof( cufftDoubleComplex ) * n, cudaMemcpyDeviceToHost );
    if ( cerr != cudaSuccess )
    {
        err = std::string("cudaMemcpy D2H: ") + cudaErrToStr( cerr );
        cufftDestroy( plan ); cudaFree( d_in ); cudaFree( d_out );
        return false;
    }

    cufftDestroy( plan );
    cudaFree( d_in );
    cudaFree( d_out );

    for ( int i = 0; i < n; ++i )
    {
        out[i] = { h_out[i].x, h_out[i].y };
    }
    return true;
}
