#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include <complex.h>
#include <fftw3.h>
#include <sndfile.h>

#include <math.h>

#include "gnuplot_i.h"

#define FRAME_SIZE 511
#define FFT_SIZE 1024
#define HOP_SIZE 511

static gnuplot_ctrl* h;
static fftw_plan plan;

// STDDEV
double stddev(double* val, int size) {
  double sum = 0.0;
  double mean = 0.0;
  double var = 0.0;

  int i = 0;
  for (i = 0; i < size; i++)
    sum += val[i];

  mean = sum / (double)size;

  for (i = 0; i < size; i++)
    var += pow(val[i] - mean, 2);

  return sqrt(var);
}

// ZERO PHASE
void zero_phase(complex s[FRAME_SIZE]) {
  int i;

  assert(s);

  for (i = 0; i < (FRAME_SIZE / 2); i++) {
    double tmp = s[i];
    s[i] = s[FRAME_SIZE / 2 + i];
    s[FRAME_SIZE / 2 + i] = s[FRAME_SIZE + i];
    s[FFT_SIZE - FRAME_SIZE / 2 + i] = tmp;
  }
}

static void print_usage(char* progname) {
  printf("\nUsage : %s <input file> \n", progname);
  puts("\n");
}
static void fill_buffer(double* buffer, double* new_buffer) {
  int i;
  double tmp[FRAME_SIZE - HOP_SIZE];

  /* save */
  for (i = 0; i < FRAME_SIZE - HOP_SIZE; i++)
    tmp[i] = buffer[i + HOP_SIZE];

  /* save offset */
  for (i = 0; i < (FRAME_SIZE - HOP_SIZE); i++) {
    buffer[i] = tmp[i];
  }

  for (i = 0; i < HOP_SIZE; i++) {
    buffer[FRAME_SIZE - HOP_SIZE + i] = new_buffer[i];
  }
}

static int read_n_samples(SNDFILE* infile,
                          double* buffer,
                          int channels,
                          int n) {
  if (channels == 1) {
    /* MONO */
    int readcount;

    readcount = sf_readf_double(infile, buffer, n);

    return readcount == n;
  } else if (channels == 2) {
    /* STEREO */
    double buf[2 * n];
    int readcount, k;
    readcount = sf_readf_double(infile, buf, n);
    for (k = 0; k < readcount; k++)
      buffer[k] = (buf[k * 2] + buf[k * 2 + 1]) / 2.0;

    return readcount == n;
  } else {
    /* FORMAT ERROR */
    printf("Channel format error.\n");
  }

  return 0;
}

static int read_samples(SNDFILE* infile, double* buffer, int channels) {
  return read_n_samples(infile, buffer, channels, HOP_SIZE);
}

void fft_init(complex in[FFT_SIZE], complex spec[FRAME_SIZE]) {
  plan = fftw_plan_dft_1d(FFT_SIZE, in, spec, FFTW_FORWARD, FFTW_ESTIMATE);
}

void fft_exit(void) {
  fftw_destroy_plan(plan);
}

void fft_process(void) {
  fftw_execute(plan);
}

int main(int argc, char* argv[]) {
  char *progname, *infilename;
  SNDFILE* infile = NULL;
  SF_INFO sfinfo;

  progname = strrchr(argv[0], '/');
  progname = progname ? progname + 1 : argv[0];

  if (argc != 2) {
    print_usage(progname);
    return 1;
  };

  infilename = argv[1];

  if ((infile = sf_open(infilename, SFM_READ, &sfinfo)) == NULL) {
    printf("Not able to open input file %s.\n", infilename);
    puts(sf_strerror(NULL));
    return 1;
  };

  /* Read WAV */
  int nb_frames = 0;
  double new_buffer[HOP_SIZE];
  double buffer[FRAME_SIZE];

  /* Plot Init */
  h = gnuplot_init();
  gnuplot_setstyle(h, "lines");

  int i;
  for (i = 0; i < (FRAME_SIZE / HOP_SIZE - 1); i++) {
    if (read_samples(infile, new_buffer, sfinfo.channels) == 1)
      fill_buffer(buffer, new_buffer);
    else {
      printf("not enough samples !!\n");
      return 1;
    }
  }

  complex samples[FFT_SIZE];
  double amplitude[FFT_SIZE];
  double phase[FFT_SIZE];
  double phase_prev[FFT_SIZE];
  complex spectrum[FFT_SIZE];

  /* FFT init */
  fft_init(samples, spectrum);

  for (i = 0; i < FFT_SIZE; i++)
    phase_prev[i] = 0;

  while (read_samples(infile, new_buffer, sfinfo.channels) == 1) {
    /* Process Samples */
    printf("Processing frame %d\n", nb_frames);

    /* hop size */
    fill_buffer(buffer, new_buffer);

    // fft process
    for (i = 0; i < FRAME_SIZE; i++) {
      // Fenetre Hann
      samples[i] =
          buffer[i] * (0.5 - 0.5 * cos(2.0 * M_PI * (double)i / FRAME_SIZE));
    }

    for (i = FRAME_SIZE; i < FFT_SIZE; i++) {
      samples[i] = 0.0;
    }

    zero_phase(samples);

    fft_process();

    // spectrum contient les complexes résultats de la fft
    for (i = 0; i < FFT_SIZE; i++) {
      amplitude[i] = 20.0 * log10(cabs(spectrum[i]) / 0.000001);

      /* Phase de la trame précédente */
      phase_prev[i] = phase[i];

      /* Phase de la trame actuelle */
      phase[i] = carg(spectrum[i]);
    }

    int imax = 0;
    double max = 0.0;

    for (i = 0; i < FFT_SIZE / 2; i++) {
      if (amplitude[i] > max) {
        max = amplitude[i];
        imax = i;
      }
    }

    /* Fixe le 10ème pont du spectre discret */
    imax = 10;

    printf("max %d %f\n", imax, (double)imax * sfinfo.samplerate / FFT_SIZE);
    printf("phase %f\n", phase[imax]);

    /* plot amplitude */
    gnuplot_resetplot(h);

    // pic sinus
    gnuplot_plot_x(h, phase, 20, "phase");

    // pic bruit
    // gnuplot_plot_x(h,phase+100,20,"phase");
    sleep(1);

    nb_frames++;
  }

  sf_close(infile);

  /* FFT exit */
  fft_exit();

  return 0;
} /* main */
