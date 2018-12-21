#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>

#define FE 44100
#define DUREE 3
#define N DUREE*FE
#define FRAME_SIZE 1024

/* output */
static char *RAW_OUT = "tmp-out.raw";
static char *FILE_OUT = "out.wav";

static fftw_plan iplan = NULL;

/******************************************************************************/

void ifft_init(complex in[N], complex out[N]) {
  iplan = fftw_plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
}

void ifft_exit(void) {
  fftw_destroy_plan(iplan);
}

void ifft_process(void) {
  fftw_execute(iplan);
}

/******************************************************************************/

FILE *
sound_file_open_write (void)
{
  return fopen (RAW_OUT, "wb");
}

void
sound_file_close_write (FILE *fp)
{
  char cmd[256];
  fclose (fp);
  snprintf(cmd, 256, "sox -c 1 -r %d -e signed-integer -b 16 %s %s", (int)FE, RAW_OUT, FILE_OUT);
  system (cmd);
}

void
sound_file_write (double *s, FILE *fp)
{
  int i;
  short tmp[N];
  for (i=0; i<N; i++)
    {
      tmp[i] = (short)(s[i]*32767);
    }
  fwrite (tmp, sizeof(short), N, fp);
}

/******************************************************************************/

void
normalize (double *s) {
	double max = 0;

	for (int i = 0; i < N; i++) {
		if (s[i] > max)
			max = s[i];
	}

	for (int i = 0; i < N; i++) {
		s[i] /= max;
	}
}

void
synthese_bruit (double *s)
{
  for (int i=0; i<N; i++)
    s[i] = 0.10 * (rand() % 100) / 100.0;
}

void
synthese_sinus (double *s)
{
  for (int i=0; i<N; i++)
    s[i] = 0.10 * sin(2.0 * M_PI * 440.0 * i / (1.0 * FE));
}

void
synthese_sinus_add (double *s)
{
  for (int i=0; i<N; i++) {
    s[i] = .0;

    for (int j=0; j<500; j++)
      s[i] += 1.0/500.0 * 5.0 * sin(2.0 * M_PI * 440.0 * i / (1.0 * FE) + j);
  }
}

void
synthese_forme_couplee (double *s)
{
    double y[N];

    double delta = 2.0 * M_PI * 440.0 / (1.0 * FE);
    double phase = .0;

    double Cx = cos(delta);
    double Cy = sin(delta);

    s[0] = cos(phase);
    y[0] = sin(phase);

    for (int i=1; i<N; i++) {
        s[i] = s[i-1] * Cx - y[i-1] * Cy;
        y[i] = s[i-1] * Cy + y[i-1] * Cx;
    }
}

void
synthese_ifft (double *s)
{
    complex spec[N];
    complex out[N];

    ifft_init(spec, out);

		double phase = .0;
		double amp = 1;
		double freq = 440.0;

    for (int j = 0; j < N; j++)
      	spec[j] = 0;

		int ix = (freq * N) / FE;

    spec[ix] = amp / 2 * cexp(I * phase);
    spec[N - ix - 1] = amp / 2 * cexp(I * -phase);

    ifft_process();

    for (int j = 0; j < N; j++)
      	s[j] = creal(out[j]);

    ifft_exit();
}

void
synthese_ifft_frames (double *s)
{/*
    complex spec[FRAME_SIZE];
    complex out[FRAME_SIZE];

    ifft_init(spec, out);

		double phase = .0;
		double amp = 1;
		double freq = 440.0;

		for (int i = 0; i * FRAME_SIZE < N / 2; i++) {
		    for (int j = 0; j < FRAME_SIZE; j++)
		      	spec[j] = 0;

				int ix = (freq * N) / FE;

				if (ix < (i+1) * FRAME_SIZE && ix > i * FRAME_SIZE) {
				    spec[ix - i * FRAME_SIZE] = amp / 2 * cexp(I * phase);
				    spec[FRAME_SIZE - ix - 1] = amp / 2 * cexp(I * -phase);
				}

		    ifft_process();

		    for (int j = 0; j < FRAME_SIZE; j++)
		      	s[i * FRAME_SIZE + j] = creal(out[j]);
		}

    ifft_exit();*/
}

void
synthese_AM (double *s)
{
	double freq1 = 1000, freq2 = 5000, freq3 = 500;

  for (int i=0; i<N; i++) {
		s[i] = 0.1 * sin(2.0 * M_PI * freq1 * i / FE) + 0.5 * sin (2.0 * M_PI * freq2 * i / FE);

		s[i] *= 0.5 + sin(2.0 * M_PI * freq3 * i / FE);
	}

	normalize(s);
}

void
synthese_FM (double *s)
{
	double freq1 = 1320, freq2 = 440, ind_mod = 2;

  for (int i=0; i<N; i++) {
		s[i] = 0.5 * sin(2.0 * M_PI * freq1 * i / FE + ind_mod * sin(2.0 * M_PI * freq2 * i / FE));
	}

	normalize(s);
}

void
synthese_Karplus_Strong (double *s)
{
	int frame = N / 10;

  for (int i=0; i < frame; i++)
    s[i] = 1.0 * rand() / RAND_MAX * 2 - 1;

	for (int i = frame; i < N; i++) {
		s[i] = (s[i - frame] + s[i - (frame + 1)]) / 2;
	}

	normalize(s);
}

/******************************************************************************/

int
main (int argc, char *argv[])
{
  FILE *output;
  double s[N];

  if (argc != 1)
    {
      printf ("usage: %s\n", argv[0]);
      exit (EXIT_FAILURE);
    }

  output = sound_file_open_write ();

  //synthese_bruit(s);
  //synthese_sinus(s);
  //synthese_sinus_add(s);
  //synthese_forme_couplee(s);
  //synthese_ifft(s);
  //synthese_ifft_frames(s); //TODO
  //synthese_AM(s);
  //synthese_FM(s);
  synthese_Karplus_Strong(s);

  sound_file_write (s, output);
  sound_file_close_write (output);

  exit (EXIT_SUCCESS);
}
