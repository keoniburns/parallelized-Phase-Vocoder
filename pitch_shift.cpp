// #include <algorithm>
// #include <complex>
// #include <numeric>
#include <cmath>
#include <iostream>
#include <vector>
#include <math

#include "AudioFile.h"
#include "smb_starter.cpp"

using namespace std;

const double PI = 3.141592653589793238462643383279502884;

int main() {
    // TODO: Provide necessary parameters and call phaseVocoder function
    AudioFile<double> audio;
    double* output;
    audio.load("./sin_1000hz.wav");
    // int sampleRate = audio.getSampleRate();
    // int bD = audio.getBitDepth();
    audio.printSummary();

    // for (int i = 0; i < 10; i++) {
    //     double curSample = audio.samples[0][i];
    //     cout << "sample " << i << ": " << curSample << endl;
    // }
    // for (int i = 0; i < audio.samples[0]; i++) {
    //     smbPitchShift(1.5, audio.getNumSamplesPerChannel(), 1024, 32, audio.getSampleRate(), audio.samples[0][i],
    //                   output);
    //     cout << output << endl;
    // }

    return 0;
}

// void phaseVocoder(const vector<vector<complex<double>>>& anls_stft, int n_anls_freqs, int n_anls_frames, int
// channels,
//                   int n_synth_frames, int hop_len, int win_len, int n_fft, double scaling, int sr) {
//     vector<double> freqs(n_anls_freqs);
//     // iota(freqs.begin(), freqs.end(), 0);
//     for (int i = 0; i < freqs.end(); i++) {
//         freqs[i] = i;
//     }

//     vector<double> anls_frames(n_anls_frames);
//     // iota(anls_frames.begin(), anls_frames.end(), 0);
//     for (int i = 0; i < anls_frames.end(); i++) {
//     }

//     vector<int> og_idxs(n_synth_frames);
//     for (int t = 0; t < n_synth_frames; ++t) {
//         og_idxs[t] = min(static_cast<int>(t / scaling), n_anls_frames - 1);
//     }

//     vector<vector<double>> mags(channels, vector<double>(n_anls_freqs));
//     vector<vector<double>> phases(channels, vector<double>(n_anls_freqs));

//     for (int i = 0; i < channels; ++i) {
//         for (int j = 0; j < n_anls_freqs; ++j) {
//             mags[i][j] = abs(anls_stft[i][j]);
//             phases[i][j] = arg(anls_stft[i][j]);
//         }
//     }

//     vector<vector<vector<double>>> phase_diffs(channels,
//                                                vector<vector<double>>(n_anls_freqs, vector<double>(n_anls_frames)));
//     for (int i = 0; i < channels; ++i) {
//         for (int j = 0; j < n_anls_freqs; ++j) {
//             for (int t = 0; t < n_anls_frames; ++t) {
//                 phase_diffs[i][j][t] = phases[i][j] - ((t > 0) ? phases[i][j] : 0);
//                 phase_diffs[i][j][t] = fmod(phase_diffs[i][j][t], 2 * PI);
//             }
//         }
//     }

//     // TODO: Implement interpolation functions for shifted_mags, shifted_phase_diffs, and unshifted_phases

//     vector<vector<vector<double>>> shifted_phases(channels,
//                                                   vector<vector<double>>(n_anls_freqs,
//                                                   vector<double>(n_synth_frames)));
//     for (int i = 0; i < channels; ++i) {
//         for (int j = 0; j < n_anls_freqs; ++j) {
//             shifted_phases[i][j][0] = phase_diffs[i][j][0];
//             for (int t = 1; t < n_synth_frames; ++t) {
//                 // TODO: Implement the loop logic
//             }
//         }
//     }

//     // TODO: Implement the remaining logic for synthesis and display
// }

// vector<vector<vector<double>>> interpolateFreq(const vector<int>& idxs, const vector<vector<vector<double>>>& arr) {
//     vector<int> start(idxs.begin(), idxs.end());

//     vector<vector<vector<double>>> frac(arr.size(),
//                                         vector<vector<double>>(arr[0].size(), vector<double>(arr[0][0].size(),
//                                         0.0)));
//     for (size_t i = 0; i < arr.size(); ++i) {
//         for (size_t j = 0; j < arr[0].size(); ++j) {
//             for (size_t k = 0; k < arr[0][0].size(); ++k) {
//                 frac[i][j][k] = idxs[i] - start[i];
//             }
//         }
//     }

//     vector<vector<vector<double>>> shiftedArr(
//         arr.size(), vector<vector<double>>(arr[0].size(), vector<double>(arr[0][0].size(), 0.0)));
//     for (size_t i = 0; i < arr.size(); ++i) {
//         for (size_t j = 0; j < arr[0].size(); ++j) {
//             for (size_t k = 0; k < arr[0][0].size(); ++k) {
//                 shiftedArr[i][j][k] = (j + 1 < arr[0].size()) ? arr[i][j + 1][k] : 0.0;
//             }
//         }
//     }

//     vector<vector<vector<double>>> result(arr.size(),
//                                           vector<vector<double>>(arr[0].size(), vector<double>(arr[0][0].size(),
//                                           0.0)));
//     for (size_t i = 0; i < arr.size(); ++i) {
//         for (size_t j = 0; j < arr[0].size(); ++j) {
//             for (size_t k = 0; k < arr[0][0].size(); ++k) {
//                 result[i][j][k] =
//                     arr[i][start[i]][k] * (1 - frac[i][j][k]) + shiftedArr[i][start[i]][k] * frac[i][j][k];
//             }
//         }
//     }

//     return result;
// }