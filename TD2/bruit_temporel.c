#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <assert.h>

#include <sndfile.h>

#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include "gnuplot_i.h"

#define FRAME_SIZE 1024
#define HOP_SIZE 512

#define TOTAL_SAMPLES 441000
/* 10 sec */

static gnuplot_ctrl* h;
static fftw_plan plan = NULL;
static fftw_plan iplan = NULL;

static void print_usage(char* progname) {
  printf("\nUsage : %s [output file]\n", progname);
  puts("\n");
}

static int write_n_samples(SNDFILE* outfile,
                           double* buffer,
                           int channels,
                           int n) {
  if (channels == 1) {
    /* MONO */
    int writecount;

    writecount = sf_writef_double(outfile, buffer, n);

    return writecount == n;
  } else {
    /* FORMAT ERROR */
    printf("Channel format output error.\n");
  }

  return 0;
}

static int write_samples(SNDFILE* outfile, double* buffer, int channels) {
  return write_n_samples(outfile, buffer, channels, HOP_SIZE);
}

// IFFT
void ifft_init(complex in[FRAME_SIZE], complex spec[FRAME_SIZE]) {
  iplan = fftw_plan_dft_1d(FRAME_SIZE, in, spec, FFTW_BACKWARD, FFTW_ESTIMATE);
}

void ifft_exit(void) { 
  fftw_destroy_plan(iplan); 
}

void ifft_process(void) {
  fftw_execute(iplan);
}

// FFT
void fft_init(complex in[FRAME_SIZE], complex spec[FRAME_SIZE]) {
  plan = fftw_plan_dft_1d(FRAME_SIZE, in, spec, FFTW_FORWARD, FFTW_ESTIMATE);
}

void fft_exit(void) {
  fftw_destroy_plan(plan);
}

void fft_process(void) {
  fftw_execute(plan);
}

int main(int argc, char* argv[]) {
  char *progname, *outfilename;
  SNDFILE* outfile = NULL;
  SF_INFO sfinfo = {0};
  srand(time(NULL));

  sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
  sfinfo.channels = 1;
  sfinfo.samplerate = 44100;

  progname = strrchr(argv[0], '/');
  progname = progname ? progname + 1 : argv[0];

  if (argc > 2) {
    print_usage(progname);
    return 1;
  };

  if (argc == 2) {
    outfilename = argv[1];

    /* Open the output file. */
    if ((outfile = sf_open(outfilename, SFM_WRITE, &sfinfo)) == NULL) {
      printf("Not able to open output file %s.\n", outfilename);
      puts(sf_strerror(NULL));
      return 1;
    };
  }

  double buffer[FRAME_SIZE];
  complex samples[FRAME_SIZE];
  double amplitude[FRAME_SIZE];
  double phase[FRAME_SIZE];
  double new_amplitude[FRAME_SIZE];
  complex spec[FRAME_SIZE];
  double x_axis[FRAME_SIZE];

  int nb_frames = 0;
  int i = 0;
  int j = 0;
  double oldBuffer = 0;
  double norm = (1.2 * FRAME_SIZE / 2.0);
  
  int l = (int)sfinfo.frames;
  int length_in = (int)sfinfo.frames + FRAME_SIZE + FRAME_SIZE - (l % HOP_SIZE);
  int length_out = FRAME_SIZE + (int)ceil(length_in);
  double *samples_out = malloc(sizeof(double) * length_out);
  assert(samples_out);

  /* Plot Init */
  h = gnuplot_init();
  gnuplot_setstyle(h, "lines");

  /* FFT init */
  fft_init(samples, spec);
  ifft_init(spec, samples);

  while (nb_frames * HOP_SIZE < TOTAL_SAMPLES) {
    /* Process Samples */
    printf("Processing frame %d\n", nb_frames);


    /* SYNTHESE BRUIT*/
    /* Bruit blanc */
    /*for (i = 0; i < FRAME_SIZE; i++) {
      buffer[i] += (float) rand() / (float) (RAND_MAX / 2) - 1.f;      
      
      if (buffer[i] < -1.0 || buffer[i] > 1.0)
        printf("Bruit de %f hors bornes.\n", buffer[i]);  
    }*/
    
    /* Bruit gaussien */
    /*for (i = 0; i < FRAME_SIZE; i++) {
      buffer[i] = 0;
      
      for (int j=0; j<12; j++)
        buffer[i] += (float) rand() / (float) (RAND_MAX / 2) - 1.f;
        
      buffer[i] /= 12;      
      
      if (buffer[i] < -1.0 || buffer[i] > 1.0)
        printf("Bruit de %f hors bornes.\n", buffer[i]);       
    }*/
    
    /* Bruit brownien */
    /*if (nb_frames == 0)
      buffer[0] = (float) rand() / (float) (RAND_MAX / 2) - 1.f;
      
    else
      buffer[0] = oldBuffer;
    
    for (i = 1; i < FRAME_SIZE; i++) {
      do {
        buffer[i] = buffer[i-1] + ((float) rand() / (float) (RAND_MAX / 2) - 1.f) / 10.f;
      } while (buffer[i] > 1.0 || buffer[i] < -1.0);      
      
      if (buffer[i] < -1.0 || buffer[i] > 1.0)
        printf("Bruit de %f hors bornes.\n", buffer[i]);          
    }
    
    oldBuffer = buffer[FRAME_SIZE-1];*/


    // fft input
    for (i = 0; i < FRAME_SIZE; i++)
      samples[i] = buffer[i];

    fft_process();

    // calcul du spectre amplitude
    for (i = 0; i < FRAME_SIZE; i++)
      amplitude[i] = cabs(spec[i]) / FRAME_SIZE;
    
    
    /* Reconstruction */
    new_amplitude[0] = amplitude[0];
    new_amplitude[FRAME_SIZE / 2] = amplitude[0];
    for (j = 1; j < FRAME_SIZE / 2; j++) {
      new_amplitude[j] = 1.0;
      new_amplitude[FRAME_SIZE - j] = new_amplitude[j];
      
      phase[j-1] = 2.0 * M_PI * rand() / RAND_MAX;
      phase[FRAME_SIZE - j] = phase[j-1];
    }
    
    /* PLOT */
    for (i = 0; i < FRAME_SIZE; i++)
      x_axis[i] = (double)i * 44100.0 / FRAME_SIZE;
    
    if (argc == 1) {
      gnuplot_resetplot(h);
      //gnuplot_plot_xy(h, x_axis, amplitude, FRAME_SIZE / 2, "Rep spectrale");
      gnuplot_plot_xy(h, x_axis, new_amplitude, FRAME_SIZE / 2, "Rep spectrale");  
      sleep(1);
    }
    

    // IFFT
    for (j = 0; j < FRAME_SIZE; j++) {
      spec[j] = new_amplitude[j] * cexp(I * phase[j]);
    }

    ifft_process();

    for (j = 0; j < FRAME_SIZE; j++) 
      buffer[j] = creal(samples[j]) / norm;

    for (j = 0; j < FRAME_SIZE; j++)
      samples_out[HOP_SIZE * nb_frames + j] += buffer[j];


    /* SAVE */
    if (argc == 2)
      if (write_samples(outfile, buffer, sfinfo.channels) != 1)
        printf("saving problem !! \n");

    nb_frames++;
  }

  sf_close(outfile);

  /* FFT exit */
  fft_exit();

  return 0;
} /* main */

