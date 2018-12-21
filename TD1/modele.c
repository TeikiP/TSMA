#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include <sndfile.h>

#include <math.h>
#include <complex.h>
#include <fftw3.h>

/* Taille (cad duree) des buffers */
#define FRAME_SIZE 1024

/* Avancement */
#define N1 512
#define N2 512

typedef complex spectrum[FRAME_SIZE];

static fftw_plan plan = NULL;
static fftw_plan iplan = NULL;

static void print_usage(char *progname) {
  printf("\nUsage : %s <input file> <output file>\n", progname);
  puts("\n");
}

// HANN Window
double hann_window(double s[FRAME_SIZE]) {
  double sum = 0;

  int i;

  assert(s);

  for (i = 0; i < FRAME_SIZE; i++) {
    double w_i = 0.5 * (1 - cos(2 * M_PI * i / FRAME_SIZE));

    s[i] *= w_i;

    sum += w_i;
  }

  return sum;
}

// IFFT
void ifft_init(spectrum in, spectrum out) {
  iplan = fftw_plan_dft_1d(FRAME_SIZE, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
}

void ifft_exit(void) { fftw_destroy_plan(iplan); }

void ifft_process(void) { fftw_execute(iplan); }

// FFT
void fft_init(spectrum in, spectrum out) {
  plan = fftw_plan_dft_1d(FRAME_SIZE, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
}

void fft_exit(void) { fftw_destroy_plan(plan); }

void fft_process(void) { fftw_execute(plan); }

int main(int argc, char *argv[]) {
  char *progname, *infilename, *outfilename;
  SNDFILE *infile = NULL;
  SNDFILE *outfile = NULL;
  SF_INFO sfinfo;
  SF_INFO sfinfo_out;

  progname = strrchr(argv[0], '/');
  progname = progname ? progname + 1 : argv[0];

  if (argc != 3) {
    print_usage(progname);
    return 1;
  };

  infilename = argv[1];
  outfilename = argv[2];

  if (strcmp(infilename, outfilename) == 0) {
    printf("Error : Input and output filenames are the same.\n\n");
    print_usage(progname);
    return 1;
  };

  if ((infile = sf_open(infilename, SFM_READ, &sfinfo)) == NULL) {
    printf("Not able to open input file %s.\n", infilename);
    puts(sf_strerror(NULL));
    return 1;
  };

  printf("Processing : %s\n", infilename);

  /* verify sampling rate */
  printf("Sampling rate : %d\n", sfinfo.samplerate);
  printf("Channels : %d\n", sfinfo.channels);

  if (sfinfo.samplerate != 44100) {
    printf("Error : processing only 44100 Hz files \n");
    exit(EXIT_FAILURE);
  }

  if (sfinfo.channels != 1) {
    printf("Error : processing only MONO files \n");
    exit(EXIT_FAILURE);
  }

  if ((sfinfo.format | SF_FORMAT_PCM_16) == 0) {
    printf("Error : processing only 16bits files \n");
    exit(EXIT_FAILURE);
  }

  sfinfo_out.samplerate = sfinfo.samplerate;
  sfinfo_out.channels = sfinfo.channels;
  sfinfo_out.format = sfinfo.format;

  if ((outfile = sf_open(outfilename, SFM_WRITE, &sfinfo_out)) == NULL) {
    printf("Not able to open output file %s.\n", outfilename);
    puts(sf_strerror(NULL));
    return 1;
  };

  // variables
  complex in[FRAME_SIZE];
  complex out[FRAME_SIZE];

  fft_init(in, out);
  ifft_init(out, in);

  // Samples
  int l = (int)sfinfo.frames;
  int length_in = (int)sfinfo.frames + FRAME_SIZE + FRAME_SIZE - (l % N1);
  double *samples_in = malloc(sizeof(double) * length_in);
  assert(samples_in);

  int length_out = FRAME_SIZE + (int)ceil(length_in);
  double *samples_out = malloc(sizeof(double) * length_out);
  assert(samples_out);

  int i;
  for (i = 0; i < length_in; i++) samples_in[i] = 0.0;
  for (i = 0; i < length_out; i++) samples_out[i] = 0.0;

  // Wav read
  int readcount = 0;
  readcount = sf_readf_double(infile, samples_in + FRAME_SIZE, sfinfo.frames);
  if (readcount < sfinfo.frames)
    printf("problème lecture %d %d\n", readcount, (int)sfinfo.frames);

  //
  int pin = 0;
  int pout = 0;
  int pend = length_in - FRAME_SIZE;
  double s[FRAME_SIZE];
  int j = 0;
  double norm = (1.2 * FRAME_SIZE / 2.0);
  double amplitude[FRAME_SIZE];
  double phase[FRAME_SIZE];

  /* spectre amplitude de la partie sinus */
  double sinus_amplitude[FRAME_SIZE];
  double amplitude_prev[FRAME_SIZE];

  for (i = 0; i < FRAME_SIZE; i++) {
    amplitude_prev[i] = 0;
  }

  while (pin < pend) {
    for (j = 0; j < FRAME_SIZE; j++) s[j] = samples_in[pin + j];

    printf("Processing Frame\n");

    // FFT
    for (j = 0; j < FRAME_SIZE; j++) in[j] = s[j];

    fft_process();

    for (j = 0; j < FRAME_SIZE / 2; j++) {
      /* Amplitude de la trame précédente */
      amplitude_prev[j] = amplitude[j];
      amplitude[j] = cabs(out[j]);

      phase[j] = carg(out[j]);

      // Processing

      /* TODO */
      /* Ici tous les casiers composent la partie sinus */
      if (j == j) {
        sinus_amplitude[j] = amplitude[j];
        // printf("bin sinus : %d %f\n",  (int)j*sfinfo.samplerate/FRAME_SIZE,
        // amplitude[j]);
      } else {
        sinus_amplitude[j] = 0.0;
      }
    }

    /* Reconstruction */
    sinus_amplitude[0] = amplitude[0];
    sinus_amplitude[FRAME_SIZE / 2] = amplitude[0];
    for (j = 1; j < FRAME_SIZE / 2; j++) {
      sinus_amplitude[FRAME_SIZE - j] = sinus_amplitude[j];
      phase[FRAME_SIZE - j] = -phase[j];
    }

    // IFFT
    for (j = 0; j < FRAME_SIZE; j++) {
      out[j] = sinus_amplitude[j] * cexp(I * phase[j]);
    }

    ifft_process();

    for (j = 0; j < FRAME_SIZE; j++) s[j] = creal(in[j]) / norm;

    hann_window(s);

    for (j = 0; j < FRAME_SIZE; j++) samples_out[pout + j] += s[j];

    pin = pin + N1;
    pout = pout + N2;
  }

  // Save and normalize
  double max_samples = 0.0;

  for (i = 0; i < length_out; i++) {
    if (fabs(samples_out[i]) > max_samples) max_samples = fabs(samples_out[i]);
  }

  /* normalisation si besoin */
  /*
  printf("normalize : %f\n", max_samples);

    for (i=0; i< length_out;i++)
    samples_out[i] = samples_out[i]/max_samples;
  */

  int writecount = 0;
  writecount = sf_writef_double(outfile, samples_out + FRAME_SIZE,
                                length_out - FRAME_SIZE);
  if (writecount < length_out - FRAME_SIZE)
    printf("problème ecriture %d %d\n", writecount, length_out - FRAME_SIZE);

  /* exit */
  fft_exit();
  ifft_exit();
  free(samples_in);
  free(samples_out);
  sf_close(infile);
  sf_close(outfile);

  return EXIT_SUCCESS;
} /* main */
