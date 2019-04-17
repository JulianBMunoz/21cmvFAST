#include "../Parameter_files/INIT_PARAMS.H"
#include "../Parameter_files/ANAL_PARAMS.H"
#include "../Parameter_files/Variables.h"
#include "filter.c"

/*
  Generates the initial conditions:
  gaussian random density field (DIM^3)
  as well as the equal or lower resolution
  velocity fields, and smoothed density field (HII_DIM^3).
  See INIT_PARAMS.H and ANAL_PARAMS.H to set the appropriate parameters.
  Output is written to ../Boxes

  Author: Andrei Mesinger
  Date: 9/29/06
*/

/*
Modified by Julian B Munoz to include DM-b relative velocities
@Harvard 08/2018
*/



#define TOTAL_COSMOLOGY_FILEPARAMS (int)7

#define NUM_HIGH_LEVEL_RNG (int) 5


/*****  Adjust the complex conjugate relations for a real array  *****/
void adj_complex_conj(fftwf_complex *box){
  int i, j, k;

  // corners
  box[C_INDEX(0,0,0)] = 0;
  box[C_INDEX(0,0,MIDDLE)] = crealf(box[C_INDEX(0,0,MIDDLE)]);
  box[C_INDEX(0,MIDDLE,0)] = crealf(box[C_INDEX(0,MIDDLE,0)]);
  box[C_INDEX(0,MIDDLE,MIDDLE)] = crealf(box[C_INDEX(0,MIDDLE,MIDDLE)]);
  box[C_INDEX(MIDDLE,0,0)] = crealf(box[C_INDEX(MIDDLE,0,0)]);
  box[C_INDEX(MIDDLE,0,MIDDLE)] = crealf(box[C_INDEX(MIDDLE,0,MIDDLE)]);
  box[C_INDEX(MIDDLE,MIDDLE,0)] = crealf(box[C_INDEX(MIDDLE,MIDDLE,0)]);
  box[C_INDEX(MIDDLE,MIDDLE,MIDDLE)] = crealf(box[C_INDEX(MIDDLE,MIDDLE,MIDDLE)]);


  // do entire i except corners
  for (i=1; i<MIDDLE; i++){
    // just j corners
    for (j=0; j<=MIDDLE; j+=MIDDLE){
      for (k=0; k<=MIDDLE; k+=MIDDLE){
	box[C_INDEX(i,j,k)] = conjf(box[C_INDEX(DIM-i,j,k)]);
      }
    }

    // all of j
    for (j=1; j<MIDDLE; j++){
      for (k=0; k<=MIDDLE; k+=MIDDLE){
	box[C_INDEX(i,j,k)] = conjf(box[C_INDEX(DIM-i,DIM-j,k)]);
	box[C_INDEX(i,DIM-j,k)] = conjf(box[C_INDEX(DIM-i,j,k)]);
      }
    }
  } // end loop over i


  // now the i corners
  for (i=0; i<=MIDDLE; i+=MIDDLE){
    for (j=1; j<MIDDLE; j++){
      for (k=0; k<=MIDDLE; k+=MIDDLE){
	box[C_INDEX(i,j,k)] = conjf(box[C_INDEX(i,DIM-j,k)]);
      }
    }
  } // end loop over remaining j
}

void gsl_rng_free_threaded (gsl_rng **r, int num_th){
  int i;
  for (i=0; i<num_th; i++)
    gsl_rng_free (r[i]);
}

/* MAIN PROGRAM */
int main(int argc, char ** argv){
  fftwf_complex *box;
  fftwf_plan plan;
  unsigned long long ct;
  int n_x, n_y, n_z, i, j, k, thread_num;
  float k_x, k_y, k_z, k_mag, p, a, b, k_sq, *smoothed_box;
  float p_vcb; //JBM:have to declare it here b/c C99
  double pixel_deltax;
  FILE *OUT, *IN, *F;
  float f_pixel_factor;
  char filename[80];
  gsl_rng * r[NUMCORES];
  time_t start_time, curr_time;
  int NUM_RNG_THREADS;

 //JBM:
   fftwf_complex *box_vcb_x;
   fftwf_complex *box_vcb_y;
   fftwf_complex *box_vcb_z;

   fftwf_complex *box_vcb;
   float *box_vcb_real; //we have to make an extra component to define it in real space

   float *smoothed_box_vcb;
   float *smoothed_box_vcb_x;
   float *smoothed_box_vcb_y;
   float *smoothed_box_vcb_z;

 //we need each of the components of v_i(k) to obtain v(x) (keeping correlations with \delta_m).
 //we will only need to smooth vcb, since that is the one we care about

 //JBM:we also warn the user if kmin>0.01 Mpc-1, so that not all the vcb power is included

  if(BOX_LEN < 400 && USE_RELATIVE_VELOCITIES){
    fprintf(stderr, "init: kmin > 0.01Mpc-1, so not all the power in v_cb will be included, change BOX_LEN to be larger \n.");
  }
  if(DELTA_K * DIM < 1.0 && USE_RELATIVE_VELOCITIES){
    fprintf(stderr, "init: kmax < 1 Mpc-1, so not all the power in v_cb will be included, change DIM to be larger \n.");
  }



    /****** Modifications to allow the cosmology and random seed to be obtained from file *****/

    char dummy_string[500];

    INDIVIDUAL_ID = atof(argv[1]);
    INDIVIDUAL_ID_2 = atof(argv[2]);

    double *PARAM_COSMOLOGY_VALS = calloc(TOTAL_COSMOLOGY_FILEPARAMS,sizeof(double));

    sprintf(filename,"WalkerCosmology_%1.6lf_%1.6lf.txt",INDIVIDUAL_ID,INDIVIDUAL_ID_2);
    F = fopen(filename,"rt");

    for(i=0;i<TOTAL_COSMOLOGY_FILEPARAMS;i++) {
        fscanf(F,"%s\t%lf\n",&dummy_string,&PARAM_COSMOLOGY_VALS[i]);
    }
    fclose(F);

    RANDOM_SEED = (int)PARAM_COSMOLOGY_VALS[0];
    SIGMA8 = (float)PARAM_COSMOLOGY_VALS[1];
    hlittle = (float)PARAM_COSMOLOGY_VALS[2];
    OMm = (float)PARAM_COSMOLOGY_VALS[3];
    OMl = (float)PARAM_COSMOLOGY_VALS[4];
    OMb = (float)PARAM_COSMOLOGY_VALS[5];
    POWER_INDEX = (float)PARAM_COSMOLOGY_VALS[6];


  /************  INITIALIZATION **********************/



  time(&start_time);
  // initialize power spectrum functions
  init_ps();
  //JBM:We have added an initializer to the CLASS interpolator inside init_ps().
  system("mkdir ../Boxes");

  // initialize and allocate thread info
  if (fftwf_init_threads()==0){
    fprintf(stderr, "init: ERROR: problem initializing fftwf threads\nAborting\n.");
    return -1;
  }
  fftwf_plan_with_nthreads(NUMCORES); // use all processors for init
  if (NUMCORES < NUM_HIGH_LEVEL_RNG)
    NUM_RNG_THREADS = NUMCORES;
  else
    NUM_RNG_THREADS = NUM_HIGH_LEVEL_RNG;
  omp_set_num_threads(NUM_RNG_THREADS);

  // seed the random number generators
  fprintf(stderr, "Creating Gaussian random field.\n");
  for (thread_num = 0; thread_num < NUM_RNG_THREADS; thread_num++){
    switch (thread_num){
    case 0:
      r[thread_num] = gsl_rng_alloc(gsl_rng_mt19937);
      gsl_rng_set(r[thread_num], RANDOM_SEED+thread_num);
      fprintf(stderr, "Thread #%i will use a Mersenne Twister RNG seeded with %u\n", thread_num, RANDOM_SEED+thread_num);
      break;
    case 1:
      r[thread_num] = gsl_rng_alloc(gsl_rng_ranlux389);
      gsl_rng_set(r[thread_num], RANDOM_SEED+thread_num);
      fprintf(stderr, "Thread #%i will use a ranlux389 RNG seeded with %u\n", thread_num, RANDOM_SEED+thread_num);
      break;
    case 2:
      r[thread_num] = gsl_rng_alloc(gsl_rng_cmrg);
      gsl_rng_set(r[thread_num], RANDOM_SEED+thread_num);
      fprintf(stderr, "Thread #%i will use a CMRG by L'Ecuyer seeded with %u\n", thread_num, RANDOM_SEED+thread_num);
      break;
    case 3:
      r[thread_num] = gsl_rng_alloc(gsl_rng_mrg);
      gsl_rng_set(r[thread_num], RANDOM_SEED+thread_num);
      fprintf(stderr, "Thread #%i will use a 5th order MRG seeded with %u\n", thread_num, RANDOM_SEED+thread_num);
      break;
    case 4:
      r[thread_num] = gsl_rng_alloc(gsl_rng_taus);
      gsl_rng_set(r[thread_num], RANDOM_SEED+thread_num);
      fprintf(stderr, "Thread #%i will use a Tausworthe RNG seeded with %u\n", thread_num, RANDOM_SEED+thread_num);
      break;
    } // end switch
  }

  // allocate array for the k-space and real-space boxes
  box = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
  if (!box){
    gsl_rng_free_threaded (r, NUM_RNG_THREADS); fprintf(stderr, "Init.c: Error allocating memory for box.\nAborting...\n");
    fftwf_cleanup_threads();
    free_ps(); return -1;
    //JBM:We always free the interpolator for CLASS in free_ps().
  }

  printf("Nkpix=%.0ld, DIM^3=%.0ld, Vol=%.1le \n", KSPACE_NUM_PIXELS,  DIM*DIM*DIM, VOLUME);
  printf("Npix=%.0ld, NFFT=%.0ld, 2*Nkpix=%.0ld \n\n\n\n",TOT_NUM_PIXELS, TOT_FFT_NUM_PIXELS, 2*KSPACE_NUM_PIXELS);
//JBM:
	box_vcb_x = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
	box_vcb_y = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
	box_vcb_z = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

	box_vcb = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

	box_vcb_real = (float *) malloc(sizeof(float)* (TOT_NUM_PIXELS));
    if (!box_vcb_real){
    fprintf(stderr, "Init.c: Error allocating memory for box_vcb_real.\nAborting...\n");
    gsl_rng_free_threaded (r, NUM_RNG_THREADS); fftwf_free(box);    fftwf_cleanup_threads();
    free_ps(); return -1;
  }


  // now allocate memory for the lower-resolution box
  // use HII_DIM from ANAL_PARAMS
  smoothed_box = (float *) malloc(sizeof(float)*HII_TOT_NUM_PIXELS);
  if (!smoothed_box){
    fprintf(stderr, "Init.c: Error allocating memory for low-res vcb box.\nAborting...\n");
    gsl_rng_free_threaded (r, NUM_RNG_THREADS); fftwf_free(box);    fftwf_cleanup_threads();
    free_ps(); return -1;
  }

//JBM:
  smoothed_box_vcb = (float *) malloc(sizeof(float)*HII_TOT_NUM_PIXELS);
  if (!smoothed_box_vcb){
    fprintf(stderr, "Init.c: Error allocating memory for low-res box_vcb.\nAborting...\n");
    gsl_rng_free_threaded (r, NUM_RNG_THREADS); fftwf_free(box);    fftwf_cleanup_threads();
    free_ps(); return -1;
  }
  smoothed_box_vcb_x = (float *) malloc(sizeof(float)*HII_TOT_NUM_PIXELS);
  if (!smoothed_box_vcb_x){
    fprintf(stderr, "Init.c: Error allocating memory for low-res box_vcb_x.\nAborting...\n");
    gsl_rng_free_threaded (r, NUM_RNG_THREADS); fftwf_free(box);    fftwf_cleanup_threads();
    free_ps(); return -1;
  }
  smoothed_box_vcb_y = (float *) malloc(sizeof(float)*HII_TOT_NUM_PIXELS);
  if (!smoothed_box_vcb_y){
    fprintf(stderr, "Init.c: Error allocating memory for low-res box_vcb_y.\nAborting...\n");
    gsl_rng_free_threaded (r, NUM_RNG_THREADS); fftwf_free(box);    fftwf_cleanup_threads();
    free_ps(); return -1;
  }
  smoothed_box_vcb_z = (float *) malloc(sizeof(float)*HII_TOT_NUM_PIXELS);
  if (!smoothed_box_vcb_z){
    fprintf(stderr, "Init.c: Error allocating memory for low-res box_vcb_z.\nAborting...\n");
    gsl_rng_free_threaded (r, NUM_RNG_THREADS); fftwf_free(box);    fftwf_cleanup_threads();
    free_ps(); return -1;
  }



  // find factor of HII pixel size / deltax pixel size
  f_pixel_factor = DIM/(float)HII_DIM;
  /************  END INITIALIZATION ******************/






  /************ CREATE K-SPACE GAUSSIAN RANDOM FIELD ***********/
#pragma omp parallel shared(box, box_vcb_x, box_vcb_y, box_vcb_z, r) private(n_x, k_x, n_y, k_y, n_z, k_z, k_mag, p, p_vcb , a,b)
  { // need to find a parallel random number generator
    //    fprintf(stderr, "Hello from thread #%i\n", omp_get_thread_num());
#pragma omp for
  for (n_x=0; n_x<DIM; n_x++){
    // convert index to numerical value for this component of the k-mode: k = (2*pi/L) * n
    if (n_x>MIDDLE)
      k_x =(n_x-DIM) * DELTA_K;  // wrap around for FFT convention
    else
      k_x = n_x * DELTA_K;

    for (n_y=0; n_y<DIM; n_y++){
      // convert index to numerical value for this component of the k-mode: k = (2*pi/L) * n
      if (n_y>MIDDLE)
	k_y =(n_y-DIM) * DELTA_K;
      else
	k_y = n_y * DELTA_K;

      // since physical space field is real, only half contains independent modes
      for (n_z=0; n_z<=MIDDLE; n_z++){
	// convert index to numerical value for this component of the k-mode: k = (2*pi/L) * n
	k_z = n_z * DELTA_K;

	// now get the power spectrum; remember, only the magnitude of k counts (due to isotropy)
	// this could be used to speed-up later maybe
	k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);
	p = power_in_k(k_mag);


//JBM: we have defined a new function power_in_vcb, for the v_cb power spectrum
	if(USE_RELATIVE_VELOCITIES){
		p_vcb =  power_in_vcb(k_mag);
	}


	// ok, now we can draw the values of the real and imaginary part
	// of our k entry from a Gaussian distribution
	a = gsl_ran_ugaussian(r[omp_get_thread_num()]);
	b = gsl_ran_ugaussian(r[omp_get_thread_num()]);
	box[C_INDEX(n_x, n_y, n_z)] = sqrt(VOLUME*p/2.0) * (a + b*I);

//JBM: we also create a rel. velocity box
//define it this way so it keeps the correlation with delta_m
	box_vcb_x[C_INDEX(n_x, n_y, n_z)] = I * k_x/k_mag * C_KMS * sqrt(VOLUME*p_vcb/2.0) * (a + b*I);
	box_vcb_y[C_INDEX(n_x, n_y, n_z)] = I * k_y/k_mag * C_KMS * sqrt(VOLUME*p_vcb/2.0) * (a + b*I);
	box_vcb_z[C_INDEX(n_x, n_y, n_z)] = I * k_z/k_mag * C_KMS * sqrt(VOLUME*p_vcb/2.0) * (a + b*I);
//JBM:we keep the i even though it does not matter as long as we're coherent.



//	printf("k=%.1le and Tv/Tm = %.1le \n", k_mag, C_KMS * sqrt(p_vcb/p));


      }
    }
    //    fprintf(stderr, "%i ", n_x);
  }
} // end omp parallel





  // no need to worry about RNG correlated streams, so use all processors now
  omp_set_num_threads(NUMCORES);

  /*****  Adjust the complex conjugate relations for a real array  *****/
  adj_complex_conj(box);

//JBM: we do it for v_cb_i as well:
  adj_complex_conj(box_vcb_x);
  adj_complex_conj(box_vcb_y);
  adj_complex_conj(box_vcb_z);


  /***** Write out the k-box *****/
  fprintf(stderr, "\nWriting k-space box...\n");
  sprintf(filename, "../Boxes/deltak_z0.00_%i_%.0fMpc", DIM, BOX_LEN);
  if (!(OUT=fopen(filename, "wb"))){
    fprintf(stderr, "init.c: Error opening %s to write to\n", filename);
  }
  else if (mod_fwrite(box, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS, 1, OUT)!=1){
    fprintf(stderr, "init.c: Write error occured writing deltak box!\n");
  }
  fclose(OUT);




  /*****JBM: Also write out the vcb k-boxes *****/
  //we do not need them, but it doesn't take long to save them

  fprintf(stderr, "\nWriting vcb k-space boxes...\n");
  sprintf(filename, "../Boxes/delta_vcb_x_k_z0.00_%i_%.0fMpc", DIM, BOX_LEN);
  if (!(OUT=fopen(filename, "wb"))){
    fprintf(stderr, "init.c: Error opening %s to write to\n", filename);
  }
  else if (mod_fwrite(box_vcb_x, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS, 1, OUT)!=1){
    fprintf(stderr, "init.c: Write error occured writing delta_vcb_k_z0 box!\n");
  }
  fclose(OUT);

  sprintf(filename, "../Boxes/delta_vcb_y_k_z0.00_%i_%.0fMpc", DIM, BOX_LEN);
  if (!(OUT=fopen(filename, "wb"))){
    fprintf(stderr, "init.c: Error opening %s to write to\n", filename);
  }
  else if (mod_fwrite(box_vcb_y, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS, 1, OUT)!=1){
    fprintf(stderr, "init.c: Write error occured writing delta_vcb_k_z0 box!\n");
  }
  fclose(OUT);

  sprintf(filename, "../Boxes/delta_vcb_z_k_z0.00_%i_%.0fMpc", DIM, BOX_LEN);
  if (!(OUT=fopen(filename, "wb"))){
    fprintf(stderr, "init.c: Error opening %s to write to\n", filename);
  }
  else if (mod_fwrite(box_vcb_z, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS, 1, OUT)!=1){
    fprintf(stderr, "init.c: Write error occured writing delta_vcb_k_z0 box!\n");
  }
  fclose(OUT);






//////////////////////////////////////////
////// JBM: beginning of vcb part	/////
////////////////////////////////////////

  /*** Let's also create a lower-resolution version of each vcb_i field  ***/
  time(&start_time);
  fprintf(stderr, "Filtering and sampling each vcb_i box to get low-res version...\n");

  if (DIM != HII_DIM){
    filter(box_vcb_x, 0, L_FACTOR*BOX_LEN/(HII_DIM+0.0));
    filter(box_vcb_y, 0, L_FACTOR*BOX_LEN/(HII_DIM+0.0));
    filter(box_vcb_z, 0, L_FACTOR*BOX_LEN/(HII_DIM+0.0));
    }

  // FFT back to real space each component
  plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)box_vcb_x, (float *)box_vcb_x, FFTW_ESTIMATE);
  fftwf_execute(plan);
  plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)box_vcb_y, (float *)box_vcb_y, FFTW_ESTIMATE);
  fftwf_execute(plan);
  plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)box_vcb_z, (float *)box_vcb_z, FFTW_ESTIMATE);
  fftwf_execute(plan);
  // now sample the filtered box
  for (i=0; i<HII_DIM; i++){
    for (j=0; j<HII_DIM; j++){
      for (k=0; k<HII_DIM; k++){
	smoothed_box_vcb_x[HII_R_INDEX(i,j,k)] =
	*((float *)box_vcb_x + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
				       (unsigned long long)(j*f_pixel_factor+0.5),
				       (unsigned long long)(k*f_pixel_factor+0.5)))/VOLUME;
	smoothed_box_vcb_y[HII_R_INDEX(i,j,k)] =
	*((float *)box_vcb_y + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
				       (unsigned long long)(j*f_pixel_factor+0.5),
				       (unsigned long long)(k*f_pixel_factor+0.5)))/VOLUME;
	smoothed_box_vcb_z[HII_R_INDEX(i,j,k)] =
	*((float *)box_vcb_z + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
				       (unsigned long long)(j*f_pixel_factor+0.5),
				       (unsigned long long)(k*f_pixel_factor+0.5)))/VOLUME;
	smoothed_box_vcb[HII_R_INDEX(i,j,k)] = sqrtf(smoothed_box_vcb_x[HII_R_INDEX(i,j,k)] * smoothed_box_vcb_x[HII_R_INDEX(i,j,k)]
	+ smoothed_box_vcb_y[HII_R_INDEX(i,j,k)]*smoothed_box_vcb_y[HII_R_INDEX(i,j,k)]
	+smoothed_box_vcb_z[HII_R_INDEX(i,j,k)]*smoothed_box_vcb_z[HII_R_INDEX(i,j,k)]);

      }
    }
  }

  // now write the box
  sprintf(filename, "../Boxes/smoothed_vcb_x_z0.00_%i_%.0fMpc", HII_DIM, BOX_LEN);
  OUT=fopen(filename, "wb");
  if (mod_fwrite(smoothed_box_vcb, sizeof(float)*HII_TOT_NUM_PIXELS, 1, OUT)!=1){
    fprintf(stderr, "init.c: Write error occured writing smoothed deltax box!\n");
  }
  fclose(OUT);

  //JBM:we also save as a regular text file instead of binary:
  sprintf(filename, "../Boxes/readable_smoothed_vcb_x_z0.00_%i_%.0fMpc.dat", HII_DIM, BOX_LEN);
  OUT=fopen(filename, "w");
  for (i=0; i<HII_DIM; i++){
    for (j=0; j<HII_DIM; j++){
      for (k=0; k<HII_DIM; k++){
		fprintf(OUT,"%e %e %e %e \n", i * BOX_LEN/(HII_DIM-1.0),
				j* BOX_LEN/(HII_DIM-1.0), k* BOX_LEN/(HII_DIM-1.0), (float) smoothed_box_vcb[HII_R_INDEX(i,j,k)]);
      }
    }
  }
  fclose(OUT);

//JBM: and print vrms and <v> just in case:
  float vrms_test=0.0;
  float vavg_test=0.0;

    for (i=0; i<HII_DIM; i++){
      for (j=0; j<HII_DIM; j++){
        for (k=0; k<HII_DIM; k++){
          vrms_test += ((float) smoothed_box_vcb[HII_R_INDEX(i,j,k)]) *((float) smoothed_box_vcb[HII_R_INDEX(i,j,k)]);
          vavg_test += (float) smoothed_box_vcb[HII_R_INDEX(i,j,k)];
        }
      }
    }

    vrms_test=sqrt(vrms_test/(HII_DIM*HII_DIM*HII_DIM));
    vavg_test/=(HII_DIM*HII_DIM*HII_DIM);

    printf("Vrms (test) = %.2le km/s \n", vrms_test);
    printf("Vavg (test) = %.2le km/s \n", vavg_test);


//////////////////////////////////
////// JBM: end of vcb part	/////
////////////////////////////////









  /*** Let's also create a lower-resolution version of the density field  ***/
  time(&start_time);
  fprintf(stderr, "Filtering and sampling the density box to get low-res version...\n");

  if (DIM != HII_DIM)
    filter(box, 0, L_FACTOR*BOX_LEN/(HII_DIM+0.0));
  // FFT back to real space
  time(&curr_time);
  fprintf(stderr, "End filtering which took %g min. Preparing to FFT\n", difftime(curr_time, start_time)/60.0);
  time(&start_time);
  plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)box, (float *)box, FFTW_ESTIMATE);
  fftwf_execute(plan);
  time(&curr_time);
  fprintf(stderr, "End FFT  which took %g min.\n", difftime(curr_time, start_time)/60.0);
  time(&start_time);
  // now sample the filtered box
  for (i=0; i<HII_DIM; i++){
    for (j=0; j<HII_DIM; j++){
      for (k=0; k<HII_DIM; k++){
	smoothed_box[HII_R_INDEX(i,j,k)] =
	*((float *)box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
				       (unsigned long long)(j*f_pixel_factor+0.5),
				       (unsigned long long)(k*f_pixel_factor+0.5)))/VOLUME;
      }
    }
  }
  // now write the box
  sprintf(filename, "../Boxes/smoothed_deltax_z0.00_%i_%.0fMpc", HII_DIM, BOX_LEN);
  OUT=fopen(filename, "wb");
  if (mod_fwrite(smoothed_box, sizeof(float)*HII_TOT_NUM_PIXELS, 1, OUT)!=1){
    fprintf(stderr, "init.c: Write error occured writing smoothed deltax box!\n");
  }
  fclose(OUT);

// 
//   //JBM:we also save as a regular text file instead of binary:
//   sprintf(filename, "../Boxes/readable_smoothed_deltax_z0.00_%i_%.0fMpc.dat", HII_DIM, BOX_LEN);
//   OUT=fopen(filename, "w");
//   for (i=0; i<HII_DIM; i++){
//     for (j=0; j<HII_DIM; j++){
//       for (k=0; k<HII_DIM; k++){
// 		fprintf(OUT,"%le %le %le %le \n", i * BOX_LEN/(HII_DIM-1.0),
// 				j* BOX_LEN/(HII_DIM-1.0), k* BOX_LEN/(HII_DIM-1.0), smoothed_box[HII_R_INDEX(i,j,k)]);
//       }
//     }
//   }
//   fclose(OUT);




  /******* PERFORM INVERSE FOURIER TRANSFORM *****************/
  fprintf(stderr, "Getting and writing real-space box...\n");
  sprintf(filename, "../Boxes/deltak_z0.00_%i_%.0fMpc", DIM, BOX_LEN);
  IN = fopen(filename, "rb");
  if (!IN){
    fprintf(stderr, "Couldn't open file %s for reading\nAborting...\n", filename);
    gsl_rng_free_threaded (r, NUM_RNG_THREADS); free(smoothed_box);  fftwf_free(box);  fclose(IN); fftwf_cleanup_threads();
    free_ps(); return -1;
  }
  if (mod_fread(box, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS, 1, IN)!=1){
    fprintf(stderr, "init.c: Read error occured!\n");
    gsl_rng_free_threaded (r, NUM_RNG_THREADS); free(smoothed_box);  fftwf_free(box);  fclose(IN); fftwf_cleanup_threads();
    free_ps(); return -1;
  }
  // add the 1/VOLUME factor when converting from k space to real space
  for (ct=0; ct<KSPACE_NUM_PIXELS; ct++){
     box[ct] /= VOLUME;
  }
  plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)box, (float *)box, FFTW_ESTIMATE);
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
  fftwf_cleanup();

  /***** Write the real space field *****/
  sprintf(filename, "../Boxes/deltax_z0.00_%i_%.0fMpc", DIM, BOX_LEN);
  if (!(OUT=fopen(filename, "wb"))){
    fprintf(stderr, "init.c: Error openning %s to write to\n", filename);
  }
  else if (mod_fwrite(box, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS, 1, OUT)!=1){
    fprintf(stderr, "init.c: Write error occured writing deltax box!\n");
  }
  fclose(OUT);

  /*** Now let's set the velocity field/dD/dt (in comoving Mpc) ***/
  /**** first x component ****/
  fprintf(stderr, "Setting x velocity field...\n");
  // read in the box
  rewind(IN);
  if (mod_fread(box, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS, 1, IN)!=1){
    fprintf(stderr, "init.c: Read error occured!\n");
    gsl_rng_free_threaded (r, NUM_RNG_THREADS); free(smoothed_box);  fftwf_free(box);  fclose(IN); fftwf_cleanup_threads();
    free_ps(); return -1;
  }
  // set velocities/dD/dt
#pragma omp parallel shared(box, r) private(n_x, k_x, n_y, k_y, n_z, k_z, k_sq)
  {
#pragma omp for
  for (n_x=0; n_x<DIM; n_x++){
    if (n_x>MIDDLE)
      k_x =(n_x-DIM) * DELTA_K;  // wrap around for FFT convention
    else
      k_x = n_x * DELTA_K;

    for (n_y=0; n_y<DIM; n_y++){
      if (n_y>MIDDLE)
	k_y =(n_y-DIM) * DELTA_K;
      else
	k_y = n_y * DELTA_K;

      for (n_z=0; n_z<=MIDDLE; n_z++){
	k_z = n_z * DELTA_K;

	k_sq = k_x*k_x + k_y*k_y + k_z*k_z;

	// now set the velocities
	if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
	  box[0] = 0;
	}
	else{
	  box[C_INDEX(n_x,n_y,n_z)] *= k_x*I/k_sq/VOLUME;
	  // note the last factor of 1/VOLUME accounts for the scaling in real-space, following the FFT
	}
      }
    }
    //    fprintf(stderr, "%i ", n_x);
      //printf("%i, (%f+%f*I)\n", n_x, creal(v_y[C_INDEX(n_x,0,0)]), cimag(v_y[C_INDEX(n_x,0,0)]));
  }
  }
  fprintf(stderr, "Filtering the high res box\n");
  if (DIM != HII_DIM)
    filter(box, 0, L_FACTOR*BOX_LEN/(HII_DIM+0.0));
  fprintf(stderr, "Now doing the FFT to get real-space field\n");
  plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)box, (float *)box, FFTW_ESTIMATE);
  fftwf_execute(plan);
  fprintf(stderr, "Sampling...\n");
  // now sample to lower res
  // now sample the filtered box
  fprintf(stderr, "Sampling...\n");
  for (i=0; i<HII_DIM; i++){
    for (j=0; j<HII_DIM; j++){
      for (k=0; k<HII_DIM; k++){
	smoothed_box[HII_R_INDEX(i,j,k)] =
	  *((float *)box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
				       (unsigned long long)(j*f_pixel_factor+0.5),
				       (unsigned long long)(k*f_pixel_factor+0.5)));
      }
    }
  }
  // write out file
  fprintf(stderr, "Done\n\nNow write out files\n");
  sprintf(filename, "../Boxes/vxoverddot_%i_%.0fMpc", HII_DIM, BOX_LEN);
  OUT=fopen(filename, "wb");
  if (mod_fwrite(smoothed_box, sizeof(float)*HII_TOT_NUM_PIXELS, 1, OUT)!=1){
    fprintf(stderr, "init.c: Write error occured writing v_x box!\n");
  }
  fclose(OUT);


  /**** y component ****/
  fprintf(stderr, "Setting y velocity field...\n");
  // read in the box
  rewind(IN);
  if (mod_fread(box, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS, 1, IN)!=1){
    fprintf(stderr, "init.c: Read error occured!\n");
    gsl_rng_free_threaded (r, NUM_RNG_THREADS); free(smoothed_box);  fftwf_free(box);  fclose(IN); fftwf_cleanup_threads();
    free_ps(); return -1;
  }
  // set velocities/dD/dt
#pragma omp parallel shared(box, r) private(n_x, k_x, n_y, k_y, n_z, k_z, k_sq)
  {
#pragma omp for
  for (n_x=0; n_x<DIM; n_x++){
    if (n_x>MIDDLE)
      k_x =(n_x-DIM) * DELTA_K;  // wrap around for FFT convention
    else
      k_x = n_x * DELTA_K;

    for (n_y=0; n_y<DIM; n_y++){
      if (n_y>MIDDLE)
	k_y =(n_y-DIM) * DELTA_K;
      else
	k_y = n_y * DELTA_K;

      for (n_z=0; n_z<=MIDDLE; n_z++){
	k_z = n_z * DELTA_K;

	k_sq = k_x*k_x + k_y*k_y + k_z*k_z;

	// now set the velocities
	if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
	  box[0] = 0;
	}
	else{
	  box[C_INDEX(n_x,n_y,n_z)] *= k_y*I/k_sq/VOLUME;
	  // note the last factor of 1/VOLUME accounts for the scaling in real-space, following the FFT
	}
      }
    }
  }
  //    fprintf(stderr, "%i ", n_x);
      //printf("%i, (%f+%f*I)\n", n_x, creal(v_y[HII_C_INDEX(n_x,0,0)]), cimag(v_y[HII_C_INDEX(n_x,0,0)]));
  }
  fprintf(stderr, "Filtering the high res box\n");
  if (DIM != HII_DIM)
    filter(box, 0, L_FACTOR*BOX_LEN/(HII_DIM+0.0));
  fprintf(stderr, "Now doing the FFT to get real-space field\n");
  plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)box, (float *)box, FFTW_ESTIMATE);
  fftwf_execute(plan);
  fprintf(stderr, "Sampling...\n");
  // now sample to lower res
  // now sample the filtered box
  for (i=0; i<HII_DIM; i++){
    for (j=0; j<HII_DIM; j++){
      for (k=0; k<HII_DIM; k++){
	smoothed_box[HII_R_INDEX(i,j,k)] =
	  *((float *)box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
				       (unsigned long long)(j*f_pixel_factor+0.5),
				       (unsigned long long)(k*f_pixel_factor+0.5)));
      }
    }
  }
  // write out file
  fprintf(stderr, "Done\n\nNow write out files\n");
  sprintf(filename, "../Boxes/vyoverddot_%i_%.0fMpc", HII_DIM, BOX_LEN);
  OUT=fopen(filename, "wb");
  if (mod_fwrite(smoothed_box, sizeof(float)*HII_TOT_NUM_PIXELS, 1, OUT)!=1){
    fprintf(stderr, "init.c: Write error occured writing v_y box!\n");
  }
  fclose(OUT);


  /**** z component ****/
  fprintf(stderr, "Setting z velocity field...\n");
  // read in the box
  rewind(IN);
  if (mod_fread(box, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS, 1, IN)!=1){
    fprintf(stderr, "init.c: Read error occured!\n");
    gsl_rng_free_threaded (r, NUM_RNG_THREADS); free(smoothed_box);  fftwf_free(box);  fclose(IN); fftwf_cleanup_threads();
    free_ps(); return -1;
  }
  // set velocities/dD/dt
#pragma omp parallel shared(box, r) private(n_x, k_x, n_y, k_y, n_z, k_z, k_sq)
  {
#pragma omp for
  for (n_x=0; n_x<DIM; n_x++){
    if (n_x>MIDDLE)
      k_x =(n_x-DIM) * DELTA_K;  // wrap around for FFT convention
    else
      k_x = n_x * DELTA_K;

    for (n_y=0; n_y<DIM; n_y++){
      if (n_y>MIDDLE)
	k_y =(n_y-DIM) * DELTA_K;
      else
	k_y = n_y * DELTA_K;

      for (n_z=0; n_z<=MIDDLE; n_z++){
	k_z = n_z * DELTA_K;

	k_sq = k_x*k_x + k_y*k_y + k_z*k_z;

	// now set the velocities
	if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
	  box[0] = 0;
	}
	else{
	  box[C_INDEX(n_x,n_y,n_z)] *= k_z*I/k_sq/VOLUME;
	  // note the last factor of 1/VOLUME accounts for the scaling in real-space, following the FFT
	}
      }
    }
    //fprintf(stderr, "%i ", n_x);
  }
  }
  fprintf(stderr, "Filtering the high res box\n");
  if (DIM != HII_DIM)
    filter(box, 0, L_FACTOR*BOX_LEN/(HII_DIM+0.0));
  fprintf(stderr, "Now doing the FFT to get real-space field\n");
  plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)box, (float *)box, FFTW_ESTIMATE);
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
  fftwf_cleanup();
  fprintf(stderr, "Sampling...\n");
  // now sample to lower res
  // now sample the filtered box
  for (i=0; i<HII_DIM; i++){
    for (j=0; j<HII_DIM; j++){
      for (k=0; k<HII_DIM; k++){
	smoothed_box[HII_R_INDEX(i,j,k)] =
	  *((float *)box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
				       (unsigned long long)(j*f_pixel_factor+0.5),
				       (unsigned long long)(k*f_pixel_factor+0.5)));
      }
    }
  }
  // write out file
  fprintf(stderr, "Done\n\nNow write out files\n");
  sprintf(filename, "../Boxes/vzoverddot_%i_%.0fMpc", HII_DIM, BOX_LEN);
  OUT=fopen(filename, "wb");
  if (mod_fwrite(smoothed_box, sizeof(float)*HII_TOT_NUM_PIXELS, 1, OUT)!=1){
    fprintf(stderr, "init.c: Write error occured writing v_z box!\n");
  }
  fclose(OUT);

/* *************************************************** *
 *              BEGIN 2LPT PART                        *
 * *************************************************** */

  // Generation of the second order Lagrangian perturbation theory (2LPT) corrections to
  // the ZA
  // reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D

  // Parameter set in ANAL_PARAMS.H
  if(SECOND_ORDER_LPT_CORRECTIONS){
    fprintf(stderr, "Begin 2LPT part\n");
    // use six supplementary boxes to store the gradients of phi_1 (eq. D13b)
    // Allocating the boxes
#define PHI_INDEX(i, j) ((int) ((i) - (j)) + 3*((j)) - ((int)(j))/2  )
    // ij -> INDEX
    // 00 -> 0
    // 11 -> 3
    // 22 -> 5
    // 10 -> 1
    // 20 -> 2
    // 21 -> 4

    fftwf_complex *phi_1[6];

    for(i = 0; i < 3; ++i){
      for(j = 0; j <= i; ++j){
        fprintf(stderr, "Initialization phi_1[%d, %d] = phi_1[%d]\n", i, j, PHI_INDEX(i, j));
        phi_1[PHI_INDEX(i, j)] = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

        if (!phi_1[PHI_INDEX(i, j)]){
          gsl_rng_free_threaded (r, NUM_RNG_THREADS); fprintf(stderr, "Init.c: Error allocating memory for phi_1[%d, %d].\nAborting...\n", i, j);
          fftwf_cleanup_threads();
          free_ps(); return -1;
        }
      }
    }


    for(i = 0; i < 3; ++i){
      for(j = 0; j <= i; ++j){

        fprintf(stderr, "Computing phi_1[%d, %d]...\n", i, j);
        // read in the box
        rewind(IN);
        if (mod_fread(box, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS, 1, IN)!=1){
          fprintf(stderr, "init.c: Read error occured!\n");
          gsl_rng_free_threaded (r, NUM_RNG_THREADS); free(smoothed_box);  fftwf_free(box);  fclose(IN); fftwf_cleanup_threads();

          for(i = 0; i < 3; ++i){
            for(j = 0; j <= i; ++j){
              fftwf_free(phi_1[PHI_INDEX(i,j)]);
            }
          }
          free_ps(); return -1;
        }


        // generate the phi_1 boxes in Fourier transform
#pragma omp parallel shared(phi_1, box, r) private(n_x, k_x, n_y, k_y, n_z, k_z, k_sq)
        {
#pragma omp for
        for (n_x=0; n_x<DIM; n_x++){
          if (n_x>MIDDLE)
            k_x =(n_x-DIM) * DELTA_K;  // wrap around for FFT convention
          else
            k_x = n_x * DELTA_K;

          for (n_y=0; n_y<DIM; n_y++){
            if (n_y>MIDDLE)
	            k_y =(n_y-DIM) * DELTA_K;
            else
	            k_y = n_y * DELTA_K;

            for (n_z=0; n_z<=MIDDLE; n_z++){
	            k_z = n_z * DELTA_K;

	            k_sq = k_x*k_x + k_y*k_y + k_z*k_z;

              	    float k[] = {k_x, k_y, k_z};
              //fprintf(stderr, "(k = %.2e %.2e %.2e) ", k[0], k[1], k[2]);
	            // now set the velocities
	            if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
	              phi_1[PHI_INDEX(i, j)][0] = 0;
	            }
	            else{
                //fprintf(stderr, "%.2e ", phi_1[PHI_INDEX(i, j)][C_INDEX(n_x, n_y, n_z)] );
	              phi_1[PHI_INDEX(i, j)][C_INDEX(n_x,n_y,n_z)] = -k[i]*k[j]*box[C_INDEX(n_x, n_y, n_z)]/k_sq/VOLUME;
                //fprintf(stderr, "%.2e ", phi_1[PHI_INDEX(i, j)][C_INDEX(n_x, n_y, n_z)] );

	            // note the last factor of 1/VOLUME accounts for the scaling in real-space, following the FFT
	            }
            }
          }
          fprintf(stderr, "%i ", n_x);
      //printf("%i, (%f+%f*I)\n", n_x, creal(v_y[C_INDEX(n_x,0,0)]), cimag(v_y[C_INDEX(n_x,0,0)]));
        }
        }
        fprintf(stderr, "\n");
       // Now we can generate the real phi_1[i,j]

        fprintf(stderr, "fftwf c2r phi_1[%d, %d]\n", i, j);
        plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)phi_1[PHI_INDEX(i, j)], (float *)phi_1[PHI_INDEX(i, j)], FFTW_ESTIMATE);
        fftwf_execute(plan);


      }
    }



   // Then we will have the laplacian of phi_2 (eq. D13b)
   // After that we have to return in Fourier space and generate the Fourier transform of phi_2


    fprintf(stderr, "Generating RHS eq. D13b\n");
    int m, l;
    for (i=0; i<DIM; i++){
      fprintf(stderr, "%d ", i);
      for (j=0; j<DIM; j++){
        for (k=0; k<DIM; k++){
          //fprintf(stderr, "%d %d %d\n", i, j, k);
          *( (float *)box + R_FFT_INDEX((unsigned long long)(i), (unsigned long long)(j), (unsigned long long)(k) )) = 0.0;
          for(m = 0; m < 3; ++m){
            for(l = m+1; l < 3; ++l){
              //fprintf(stderr, "(%d %d) %d   ", l, m, PHI_INDEX(l, m));
    	        *((float *)box + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)) ) += ( *((float *)(phi_1[PHI_INDEX(l, l)]) + R_FFT_INDEX((unsigned long long) (i),(unsigned long long) (j),(unsigned long long) (k)))  ) * (  *((float *)(phi_1[PHI_INDEX(m, m)]) + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)))  );
              //fprintf(stderr, "%.2f ", *((float *)box + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)) ) );
    	        *((float *)box + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)) ) -= ( *((float *)(phi_1[PHI_INDEX(l, m)]) + R_FFT_INDEX((unsigned long long)(i),(unsigned long long) (j),(unsigned long long)(k) ) )  ) * (  *((float *)(phi_1[PHI_INDEX(l, m)]) + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k) ))  );
    	        //box[R_FFT_INDEX(i,j,k)] -= phi_1[PHI_INDEX(l, m)][R_FFT_INDEX(i,j,k)] *  phi_1[PHI_INDEX(l, m)][R_FFT_INDEX(i,j,k)];
              //fprintf(stderr, "%.2e ", *((float *)box + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)) ) );
    	        *((float *)box + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)) ) /= TOT_NUM_PIXELS;

//	        smoothed_box[HII_R_INDEX(i,j,k)] =
//	          *((float *)box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
//				       (unsigned long long)(j*f_pixel_factor+0.5),
//				       (unsigned long long)(k*f_pixel_factor+0.5)));
            }
          }
          //fprintf(stderr, "\n");
        }
      }
    }

    fprintf(stderr, "Done\nNow fft r2c\n");
    plan = fftwf_plan_dft_r2c_3d(DIM, DIM, DIM, (float *)box, (fftwf_complex *)box, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fprintf(stderr, "Done\n");

    // Now we can store the content of box in a back-up file
    // Then we can generate the gradients of phi_2 (eq. D13b and D9)


    /***** Write out back-up k-box RHS eq. D13b *****/
    fprintf(stderr, "\nWriting back-up k-space box...\n");
    sprintf(filename, "../Boxes/backup_eqD13b_z0.00_%i_%.0fMpc", DIM, BOX_LEN);
    if (!(OUT=fopen(filename, "wb"))){
      fprintf(stderr, "init.c: Error openning %s to write to\n", filename);
    }
    else if (mod_fwrite(box, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS, 1, OUT)!=1){
      fprintf(stderr, "init.c: Write error occured writing deltak box!\n");
    }
    fclose(OUT);

    // For each component, we generate the velocity field (same as the ZA part)

    sprintf(filename, "../Boxes/backup_eqD13b_z0.00_%i_%.0fMpc", DIM, BOX_LEN);
    IN = fopen(filename, "rb");
    if (!IN){
      fprintf(stderr, "Couldn't open file %s for reading\nAborting...\n", filename);
      gsl_rng_free_threaded (r, NUM_RNG_THREADS); free(smoothed_box);  fftwf_free(box);  fclose(IN); fftwf_cleanup_threads();
      free_ps(); return -1;
    }
    /*** Now let's set the velocity field/dD/dt (in comoving Mpc) ***/
    /**** first x component ****/
    fprintf(stderr, "Setting x velocity field 2LPT...\n");
    // read in the box
    // TODO correct free of phi_1
    rewind(IN);
    if (mod_fread(box, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS, 1, IN)!=1){
      fprintf(stderr, "init.c: Read error occured!\n");
      gsl_rng_free_threaded (r, NUM_RNG_THREADS); free(smoothed_box);  fftwf_free(box);  fclose(IN); fftwf_cleanup_threads();
      free_ps(); return -1;
    }
    // set velocities/dD/dt
#pragma omp parallel shared(box, r) private(n_x, k_x, n_y, k_y, n_z, k_z, k_sq)
    {
#pragma omp for
    for (n_x=0; n_x<DIM; n_x++){
      if (n_x>MIDDLE)
        k_x =(n_x-DIM) * DELTA_K;  // wrap around for FFT convention
      else
        k_x = n_x * DELTA_K;

      for (n_y=0; n_y<DIM; n_y++){
        if (n_y>MIDDLE)
	        k_y =(n_y-DIM) * DELTA_K;
        else
	        k_y = n_y * DELTA_K;

        for (n_z=0; n_z<=MIDDLE; n_z++){
	        k_z = n_z * DELTA_K;

	        k_sq = k_x*k_x + k_y*k_y + k_z*k_z;

	        // now set the velocities
	        if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
	          box[0] = 0;
	        }
	        else{
	          box[C_INDEX(n_x,n_y,n_z)] *= k_x*I/k_sq;
	          // note the last factor of 1/VOLUME accounts for the scaling in real-space, following the FFT
	        }
        }
      }
        fprintf(stderr, "%i ", n_x);
      //printf("%i, (%f+%f*I)\n", n_x, creal(v_y[C_INDEX(n_x,0,0)]), cimag(v_y[C_INDEX(n_x,0,0)]));
    }
    }
    fprintf(stderr, "Filtering the high res box\n");

    if (DIM != HII_DIM)
      filter(box, 0, L_FACTOR*BOX_LEN/(HII_DIM+0.0));
    fprintf(stderr, "Now doing the FFT to get real-space field\n");
    plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)box, (float *)box, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fprintf(stderr, "Sampling...\n");
    // now sample to lower res
    // now sample the filtered box
    fprintf(stderr, "Sampling...\n");
    for (i=0; i<HII_DIM; i++){
      for (j=0; j<HII_DIM; j++){
        for (k=0; k<HII_DIM; k++){
	        smoothed_box[HII_R_INDEX(i,j,k)] =
	          *((float *)box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
				       (unsigned long long)(j*f_pixel_factor+0.5),
				       (unsigned long long)(k*f_pixel_factor+0.5)));
        }
      }
    }
    // write out file
    fprintf(stderr, "Done\n\nNow write out files\n");
    sprintf(filename, "../Boxes/vxoverddot_2LPT_%i_%.0fMpc", HII_DIM, BOX_LEN);
    OUT=fopen(filename, "wb");
    if (mod_fwrite(smoothed_box, sizeof(float)*HII_TOT_NUM_PIXELS, 1, OUT)!=1){
      fprintf(stderr, "init.c: Write error occured writting v_x box!\n");
    }
    fclose(OUT);


    /**** y component ****/
    fprintf(stderr, "Setting y velocity field 2LPT...\n");
    // read in the box
    rewind(IN);
    // TODO set free properly
    if (mod_fread(box, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS, 1, IN)!=1){
      fprintf(stderr, "init.c: Read error occured!\n");
      gsl_rng_free_threaded (r, NUM_RNG_THREADS); free(smoothed_box);  fftwf_free(box);  fclose(IN); fftwf_cleanup_threads();
      free_ps(); return -1;
    }
    // set velocities/dD/dt
#pragma omp parallel shared(box, r) private(n_x, k_x, n_y, k_y, n_z, k_z, k_sq)
    {
#pragma omp for
    for (n_x=0; n_x<DIM; n_x++){
      if (n_x>MIDDLE)
        k_x =(n_x-DIM) * DELTA_K;  // wrap around for FFT convention
      else
        k_x = n_x * DELTA_K;

      for (n_y=0; n_y<DIM; n_y++){
        if (n_y>MIDDLE)
	        k_y =(n_y-DIM) * DELTA_K;
        else
	        k_y = n_y * DELTA_K;

        for (n_z=0; n_z<=MIDDLE; n_z++){
	        k_z = n_z * DELTA_K;

	        k_sq = k_x*k_x + k_y*k_y + k_z*k_z;

	        // now set the velocities
	        if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
	          box[0] = 0;
	        }
	        else{
	        box[C_INDEX(n_x,n_y,n_z)] *= k_y*I/k_sq;
          //fprintf(stderr, "%.2e ", box[C_INDEX(n_x, n_y, n_z)]);
	        // note the last factor of 1/VOLUME accounts for the scaling in real-space, following the FFT
	        }
        }
      }
      fprintf(stderr, "%i ", n_x);
      //printf("%i, (%f+%f*I)\n", n_x, creal(v_y[HII_C_INDEX(n_x,0,0)]), cimag(v_y[HII_C_INDEX(n_x,0,0)]));
    }
    }
    fprintf(stderr, "Filtering the high res box\n");
    if (DIM != HII_DIM)
      filter(box, 0, L_FACTOR*BOX_LEN/(HII_DIM+0.0));
    fprintf(stderr, "Now doing the FFT to get real-space field\n");
    plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)box, (float *)box, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fprintf(stderr, "Sampling...\n");
    // now sample to lower res
    // now sample the filtered box
    for (i=0; i<HII_DIM; i++){
      for (j=0; j<HII_DIM; j++){
        for (k=0; k<HII_DIM; k++){
	        smoothed_box[HII_R_INDEX(i,j,k)] =
	          *((float *)box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
				       (unsigned long long)(j*f_pixel_factor+0.5),
				       (unsigned long long)(k*f_pixel_factor+0.5)));
        }
      }
    }
    // write out file
    fprintf(stderr, "Done\n\nNow write out files\n");
    sprintf(filename, "../Boxes/vyoverddot_2LPT_%i_%.0fMpc", HII_DIM, BOX_LEN);
    OUT=fopen(filename, "wb");
    if (mod_fwrite(smoothed_box, sizeof(float)*HII_TOT_NUM_PIXELS, 1, OUT)!=1){
      fprintf(stderr, "init.c: Write error occured writting v_y box!\n");
    }
    fclose(OUT);


    /**** z component ****/
    fprintf(stderr, "Setting z velocity field 2LPT...\n");
    // read in the box
    rewind(IN);
    // TODO set free properly
    if (mod_fread(box, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS, 1, IN)!=1){
      fprintf(stderr, "init.c: Read error occured!\n");
      gsl_rng_free_threaded (r, NUM_RNG_THREADS); free(smoothed_box);  fftwf_free(box);  fclose(IN); fftwf_cleanup_threads();
      free_ps(); return -1;
    }
    // set velocities/dD/dt
#pragma omp parallel shared(box, r) private(n_x, k_x, n_y, k_y, n_z, k_z, k_sq)
    {
#pragma omp for
    for (n_x=0; n_x<DIM; n_x++){
      if (n_x>MIDDLE)
        k_x =(n_x-DIM) * DELTA_K;  // wrap around for FFT convention
      else
        k_x = n_x * DELTA_K;

      for (n_y=0; n_y<DIM; n_y++){
        if (n_y>MIDDLE)
	        k_y =(n_y-DIM) * DELTA_K;
        else
	        k_y = n_y * DELTA_K;

        for (n_z=0; n_z<=MIDDLE; n_z++){
	        k_z = n_z * DELTA_K;

	        k_sq = k_x*k_x + k_y*k_y + k_z*k_z;

	        // now set the velocities
	        if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
	          box[0] = 0;
	        }
	        else{
	          box[C_INDEX(n_x,n_y,n_z)] *= k_z*I/k_sq;
	        // note the last factor of 1/VOLUME accounts for the scaling in real-space, following the FFT
	        }
        }
      }
        fprintf(stderr, "%i ", n_x);
    }
    }
      fprintf(stderr, "Filtering the high res box\n");
    if (DIM != HII_DIM)
      filter(box, 0, L_FACTOR*BOX_LEN/(HII_DIM+0.0));
    fprintf(stderr, "Now doing the FFT to get real-space field\n");
    plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)box, (float *)box, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fprintf(stderr, "Sampling...\n");
    // now sample to lower res
    // now sample the filtered box
    for (i=0; i<HII_DIM; i++){
      for (j=0; j<HII_DIM; j++){
        for (k=0; k<HII_DIM; k++){
	        smoothed_box[HII_R_INDEX(i,j,k)] =
	          *((float *)box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
				       (unsigned long long)(j*f_pixel_factor+0.5),
				       (unsigned long long)(k*f_pixel_factor+0.5)));
          //fprintf(stderr, "%.2e ", smoothed_box[HII_R_INDEX(i, j, k)]);
        }
      }
    }
    // write out file
    fprintf(stderr, "Done\n\nNow write out files\n");
    sprintf(filename, "../Boxes/vzoverddot_2LPT_%i_%.0fMpc", HII_DIM, BOX_LEN);
    OUT=fopen(filename, "wb");
    if (mod_fwrite(smoothed_box, sizeof(float)*HII_TOT_NUM_PIXELS, 1, OUT)!=1){
      fprintf(stderr, "init.c: Write error occured writting v_z box!\n");
    }
    fclose(OUT);

    // deallocate the supplementary boxes
    for(i = 0; i < 3; ++i){
      for(j = 0; j <= i; ++j){
        fftwf_free(phi_1[PHI_INDEX(i,j)]);
      }
    }
  }
/* *********************************************** *
 *               END 2LPT PART                     *
 * *********************************************** */


  // deallocate
  gsl_rng_free_threaded (r, NUM_RNG_THREADS);
  free(smoothed_box);  fftwf_free(box);  fclose(IN); fftwf_cleanup_threads();

  free(box_vcb_x); free(box_vcb_y); free(box_vcb_z); free(box_vcb_real); free(box_vcb);
  free(smoothed_box_vcb_x);free(smoothed_box_vcb_y);free(smoothed_box_vcb_z);free(smoothed_box_vcb);

  free_ps(); return 0;
}
