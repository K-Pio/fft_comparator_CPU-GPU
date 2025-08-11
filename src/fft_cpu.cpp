#include "fft_cpu.hpp"
#include <fftw3.h>
#include <stdexcept>

void cpu_fft( const std::vector<std::complex<double>>& in,
             std::vector<std::complex<double>>& out,
             bool inverse ) 
{
    const int n = static_cast<int>( in.size() );
    out.resize( n );

    fftw_complex* in_buf  = reinterpret_cast<fftw_complex*>( fftw_malloc(sizeof( fftw_complex ) * n ));
    fftw_complex* out_buf = reinterpret_cast<fftw_complex*>( fftw_malloc(sizeof( fftw_complex ) * n ));
    if ( !in_buf || !out_buf )
    {
        if ( in_buf ) fftw_free( in_buf );
        if ( out_buf ) fftw_free( out_buf );
        throw std::runtime_error( "FFTW: fftw_malloc failed" );
    }

    for ( int i = 0; i < n; ++i ) 
    {
        in_buf[i][0] = in[i].real();
        in_buf[i][1] = in[i].imag();
    }

    fftw_plan plan = fftw_plan_dft_1d(
        n, in_buf, out_buf,
        inverse ? FFTW_BACKWARD : FFTW_FORWARD,
        FFTW_ESTIMATE
    );
    if ( !plan )
    {
        fftw_free( in_buf );
        fftw_free( out_buf );
        throw std::runtime_error( "FFTW: fftw_plan_dft_1d failed" );
    }

    fftw_execute( plan );
    fftw_destroy_plan( plan );

    for (int i = 0; i < n; ++i)
    {
        out[i] = { out_buf[i][0], out_buf[i][1] };
    }

    fftw_free( in_buf );
    fftw_free( out_buf );
}
