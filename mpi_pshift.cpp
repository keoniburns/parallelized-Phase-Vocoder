/**
 * @file mpi_pshift.cpp
 * @author Keoni Burns
 * @brief
 * @version 0.1
 * @date 2023-12-05
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include "AudioFile.h"

#define M_PI 3.14159265358979323846
#define MAX_FRAME_LENGTH 8192
#define NUM_THREADS (8)

using std::cout;
using std::endl;
using std::setprecision;
using std::string;
using std::vector;

void smbFft(double *fftBuffer, long fftFrameSize, long sign);
double smbAtan2(double x, double y);
void smbPitchShift(double pitchShift, long numSampsToProcess, long fftFrameSize, long osamp, double sampleRate,
                   double *indata, double *outdata);

int main(int argc, char *argv[]) {
    int my_rank, comm_sz;
    long local_n, n;
    double loc_a, loc_b;
    double start, end;
    int threads = NUM_THREADS;

    // command line args else is for quick testing
    string infile, outfile;

    if (argc > 2) {
        infile = argv[1];
        outfile = argv[2];
    } else {
        infile = "./sounds/5min.wav";
        outfile = "./sounds/5minMPI.wav";
    }
    if (argc > 1) {
        threads = atoi(argv[1]);
    }
    AudioFile<double> audio;
    audio.load(infile);
    MPI_Init(NULL, NULL);

    /* Get my process rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* Let the system do what it needs to start up MPI */

    n = audio.getNumSamplesPerChannel();
    local_n = (n / comm_sz);  // tot_samples/tot_workers = num samples per worker

    /* why the fuck am i getting residuals */
    if ((n % comm_sz) != 0) {  // i think this will add residuals to the last worker
        if (my_rank == comm_sz - 1) {
            // cout << (local_n % comm_sz) << " residuals added to worker " << my_rank << endl;
            local_n += (local_n % comm_sz);
        }
    }

    loc_a = my_rank * local_n;
    loc_b = loc_a + local_n;
    MPI_Barrier(MPI_COMM_WORLD);  // Ensure all processes start timing at the same moment
    start = MPI_Wtime();

    // printf("my_rank=%d, start a=%lf, end b=%lf, and step_size=%ld\n", my_rank, loc_a, loc_b, local_n);
    double loc_indata[local_n];
    // #pragma omp parallel for num_threads(threads)
    for (long i = 0; i < local_n; i++) {
        loc_indata[i] = audio.samples[0][i + loc_a];
    }
    // step size is defined below as frame/osamp
    /* this needs to fit regardless of our number of workers */
    /** must be a multiple of 2
     * 64
     * 128
     * 256
     * 512
     * 1024
     * 2048
     * 4096
     * 8192
     */
    const long fftFrameSize = 2048;  // FFT frame size
    const long osamp = 32;           // STFT oversampling factor

    // we need these to determine the frequency i believe
    const double sampleRate = audio.getSampleRate();  // Sample rate in Hz
    const double pitchShift = 2.0;                    // Pitch shift factor (e.g., 1.5 for an upward shift)

    double local_outdata[local_n];
    long outSize = sizeof(local_outdata) / sizeof(double);
    double global_outdata[comm_sz * local_n];

    // Call the pitch shifting function
    smbPitchShift(pitchShift, outSize, fftFrameSize, osamp, sampleRate, loc_indata, local_outdata);

    // this should aggregate all the arrays from every worker
    MPI_Gather(local_outdata, local_n, MPI_DOUBLE, global_outdata, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // cout << "rank " << my_rank << " has finished the doings" << endl;

    if (my_rank == 0) {
        vector<vector<double>> out(1, vector<double>(audio.getNumSamplesPerChannel(), 0));

        long i;
        // #pragma omp parallel for
        for (i = 0; i < audio.getNumSamplesPerChannel(); i++) {
            // cout << setprecision(15) << global_outdata[i] << endl;

            out[0][i] = global_outdata[i];
        }
        audio.setAudioBufferSize(audio.getNumChannels(), audio.getNumSamplesPerChannel());
        audio.setAudioBuffer(out);
        audio.setSampleRate(sampleRate);
        audio.save(outfile);
    }
    end = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("%f", end - start);
    }
    MPI_Finalize();

    return 0;
}

/**
 * @brief we need to break the number of samples up per worker and pass them into this function
 *
 * @param pitchShift determines how high or low the shift is
 * @param numSampsToProcess total samples
 * @param fftFrameSize
 * @param osamp
 * @param sampleRate how we determine discrete frequency values
 * @param indata
 * @param outdata
 */
void smbPitchShift(double pitchShift, long numSampsToProcess, long fftFrameSize, long osamp, double sampleRate,
                   double *indata, double *outdata) {
    /* whole lotta variables for whatever reason */
    static double gInFIFO[MAX_FRAME_LENGTH];
    static double gOutFIFO[MAX_FRAME_LENGTH];
    static double gFFTworksp[2 * MAX_FRAME_LENGTH];
    static double gLastPhase[MAX_FRAME_LENGTH / 2 + 1];
    static double gSumPhase[MAX_FRAME_LENGTH / 2 + 1];
    static double gOutputAccum[2 * MAX_FRAME_LENGTH];
    static double gAnaFreq[MAX_FRAME_LENGTH];
    static double gAnaMagn[MAX_FRAME_LENGTH];
    static double gSynFreq[MAX_FRAME_LENGTH];
    static double gSynMagn[MAX_FRAME_LENGTH];
    static long gRover = false, gInit = false;

    /* these actually matter for the calculation of the sample */
    double magn, phase, tmp, window, real, imag;
    double freqPerBin, expct;
    long i, k, qpd, index, inFifoLatency, stepSize, fftFrameSize2;

    /* set up some handy variables */
    fftFrameSize2 = fftFrameSize / 2;
    stepSize = fftFrameSize / osamp;

    freqPerBin = sampleRate / (double)fftFrameSize;
    expct = 2. * M_PI * (double)stepSize / (double)fftFrameSize;
    inFifoLatency = fftFrameSize - stepSize;
    if (gRover == false) gRover = inFifoLatency;

    /* initialize our static arrays */
    if (gInit == false) {
        memset(gInFIFO, 0, MAX_FRAME_LENGTH * sizeof(double));
        memset(gOutFIFO, 0, MAX_FRAME_LENGTH * sizeof(double));
        memset(gFFTworksp, 0, 2 * MAX_FRAME_LENGTH * sizeof(double));
        memset(gLastPhase, 0, (MAX_FRAME_LENGTH / 2 + 1) * sizeof(double));
        memset(gSumPhase, 0, (MAX_FRAME_LENGTH / 2 + 1) * sizeof(double));
        memset(gOutputAccum, 0, 2 * MAX_FRAME_LENGTH * sizeof(double));
        memset(gAnaFreq, 0, MAX_FRAME_LENGTH * sizeof(double));
        memset(gAnaMagn, 0, MAX_FRAME_LENGTH * sizeof(double));
        gInit = true;
    }

    /* main processing loop */
    for (i = 0; i < numSampsToProcess; i++) {
        /* As long as we have not yet collected enough data just read in */
        gInFIFO[gRover] = indata[i];
        outdata[i] = gOutFIFO[gRover - inFifoLatency];
        gRover++;

        /* now we have enough data for processing */
        if (gRover >= fftFrameSize) {
            gRover = inFifoLatency;

            for (k = 0; k < fftFrameSize; k++) {
                window = -.5 * cos(2. * M_PI * (double)k / (double)fftFrameSize) + .5;
                gFFTworksp[2 * k] = gInFIFO[k] * window;
                gFFTworksp[2 * k + 1] = 0.;
            }

            /* ***************** ANALYSIS ******************* */
            /* do transform */

            smbFft(gFFTworksp, fftFrameSize, -1);

            // #pragma omp parallel for num_threads(NUM_THREADS)
            /* this is the analysis step */
            // #pragma omp parallel for num_threads(NUM_THREADS)
            for (k = 0; k <= fftFrameSize2; k++) {
                /* de-interlace FFT buffer */
                real = gFFTworksp[2 * k];
                imag = gFFTworksp[2 * k + 1];

                /* compute magnitude and phase */
                magn = 2. * sqrt(real * real + imag * imag);
                phase = atan2(imag, real);

                /* compute phase difference */
                tmp = phase - gLastPhase[k];
                gLastPhase[k] = phase;

                /* subtract expected phase difference */
                tmp -= (double)k * expct;

                /* map delta phase into +/- Pi interval */
                qpd = tmp / M_PI;
                if (qpd >= 0)
                    qpd += qpd & 1;
                else
                    qpd -= qpd & 1;
                tmp -= M_PI * (double)qpd;

                /* get deviation from bin frequency from the +/- Pi interval */
                tmp = osamp * tmp / (2. * M_PI);

                /* compute the k-th partials' true frequency */
                tmp = (double)k * freqPerBin + tmp * freqPerBin;

                /* store magnitude and true frequency in analysis arrays */
                gAnaMagn[k] = magn;
                gAnaFreq[k] = tmp;
            }

            /* ***************** PROCESSING ******************* */
            /* this does the actual pitch shifting */
            memset(gSynMagn, 0, fftFrameSize * sizeof(double));
            memset(gSynFreq, 0, fftFrameSize * sizeof(double));
            // #pragma omp parallel for num_threads(NUM_THREADS)
            for (k = 0; k <= fftFrameSize2; k++) {
                index = k * pitchShift;
                if (index <= fftFrameSize2) {
                    gSynMagn[index] += gAnaMagn[k];
                    gSynFreq[index] = gAnaFreq[k] * pitchShift;
                }
            }

            /* ***************** SYNTHESIS ******************* */
            /* this is the synthesis step */
            // #pragma omp parallel for num_threads(NUM_THREADS)
            for (k = 0; k <= fftFrameSize2; k++) {
                /* get magnitude and true frequency from synthesis arrays */
                magn = gSynMagn[k];
                tmp = gSynFreq[k];

                /* subtract bin mid frequency */
                tmp -= (double)k * freqPerBin;

                /* get bin deviation from freq deviation */
                tmp /= freqPerBin;

                /* take osamp into account */
                tmp = 2. * M_PI * tmp / osamp;

                /* add the overlap phase advance back in */
                tmp += (double)k * expct;

                /* accumulate delta phase to get bin phase */
                gSumPhase[k] += tmp;
                phase = gSumPhase[k];

                /* get real and imag part and re-interleave */
                gFFTworksp[2 * k] = magn * cos(phase);
                gFFTworksp[2 * k + 1] = magn * sin(phase);
            }

            /* zero negative frequencies */
            // #pragma omp parallel for num_threads(NUM_THREADS)
            for (k = fftFrameSize + 2; k < 2 * fftFrameSize; k++) gFFTworksp[k] = 0.;

            /* do inverse transform */
            smbFft(gFFTworksp, fftFrameSize, 1);

            /* do windowing and add to output accumulator NUM_THREADS*/
            // #pragma omp parallel for num_threads(NUM_THREADS)
            for (k = 0; k < fftFrameSize; k++) {
                window = -.5 * cos(2. * M_PI * (double)k / (double)fftFrameSize) + .5;
                gOutputAccum[k] += 2. * window * gFFTworksp[2 * k] / (fftFrameSize2 * osamp);
            }

            // #pragma omp parallel for num_threads(NUM_THREADS)
            for (k = 0; k < stepSize; k++) {
                gOutFIFO[k] = gOutputAccum[k];
            }

            /* shift accumulator */
            memmove(gOutputAccum, gOutputAccum + stepSize, fftFrameSize * sizeof(double));

            /* move input FIFO */
            // #pragma omp parallel for num_threads(NUM_THREADS)
            for (k = 0; k < inFifoLatency; k++) {
                gInFIFO[k] = gInFIFO[k + stepSize];
            }
        }
    }
}

/**
 * @brief
 *
 * @param fftBuffer
 * @param fftFrameSize
 * @param sign
 */
void smbFft(double *fftBuffer, long fftFrameSize, long sign) {
    double wr, wi, arg, *p1, *p2, temp;
    double tr, ti, ur, ui, *p1r, *p1i, *p2r, *p2i;
    long i, bitm, j, le, le2, k;

    // #pragma omp parallel for num_threads(NUM_THREADS) shared(fftBuffer) private(i, bitm, j, p1, p2, temp)
    for (i = 2; i < 2 * fftFrameSize - 2; i += 2) {
        for (bitm = 2, j = 0; bitm < 2 * fftFrameSize; bitm <<= 1) {
            if (i & bitm) j++;
            j <<= 1;
        }
        if (i < j) {
            p1 = fftBuffer + i;
            p2 = fftBuffer + j;
            temp = *p1;
            *(p1++) = *p2;
            *(p2++) = temp;
            temp = *p1;
            *p1 = *p2;
            *p2 = temp;
        }
    }
    for (k = 0, le = 2; k < (long)(log(fftFrameSize) / log(2.) + .5); k++) {
        le <<= 1;
        le2 = le >> 1;
        ur = 1.0;
        ui = 0.0;
        arg = M_PI / (le2 >> 1);
        wr = cos(arg);
        wi = sign * sin(arg);
        for (j = 0; j < le2; j += 2) {
            p1r = fftBuffer + j;
            p1i = p1r + 1;
            p2r = p1r + le2;
            p2i = p2r + 1;
            for (i = j; i < 2 * fftFrameSize; i += le) {
                tr = *p2r * ur - *p2i * ui;
                ti = *p2r * ui + *p2i * ur;
                *p2r = *p1r - tr;
                *p2i = *p1i - ti;
                *p1r += tr;
                *p1i += ti;
                p1r += le;
                p1i += le;
                p2r += le;
                p2i += le;
            }
            tr = ur * wr - ui * wi;
            ui = ur * wi + ui * wr;
            ur = tr;
        }
    }
}

/* MPI functions */
