#include "wav_signal.hpp"

std::vector<std::complex<double>> import_wav(const std::string& filename) 
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Nie można otworzyć pliku.");

    // Pomijanie nagłówka (zakładamy stały rozmiar dla prostego WAV)
    file.seekg(44);

    std::vector<std::complex<double>> signal;
    while (!file.eof()) {
        int16_t left = 0, right = 0;
        file.read(reinterpret_cast<char*>(&left), sizeof(int16_t));
        file.read(reinterpret_cast<char*>(&right), sizeof(int16_t));

        // Normalizacja do [-1.0, 1.0]
        double l = static_cast<double>(left) / 32768.0;
        double r = static_cast<double>(right) / 32768.0;

        signal.emplace_back(l, r); // traktujemy stereo jako liczby zespolone
    }

    return signal;
}