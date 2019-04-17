#include "../Parameter_files/INIT_PARAMS.H"
#include "../Parameter_files/ANAL_PARAMS.H"
#include "../Parameter_files/Variables.h"
#include "bubble_helper_progs.c"
#include "heating_helper_progs.c"
#include "gsl/gsl_sf_erf.h"
#include "filter.c"

/*
This program smooths an output box to make it look prettier
Julian B Munoz 2018
*/

// For storing the 21cm PS light cone filenames to be able to write them to file to be read by the MCMC sampler
char lightcone_box_names[1000][500];

float REDSHIFT;

void init_21cmMC_Ts_arrays();
void init_21cmMC_HII_arrays();
void init_21cmMC_TsSaveBoxes_arrays();

void ComputeBoxesForFile();
void ComputeTsBoxes();
void ComputeIonisationBoxes(int sample_index, float REDSHIFT_SAMPLE, float PREV_REDSHIFT);

void adj_complex_conj();
void ComputeInitialConditions();
void ComputePerturbField(float REDSHIFT_SAMPLE);
void GeneratePS(int CO_EVAL, double AverageTb);

void ReadFcollTable();

void destroy_21cmMC_Ts_arrays();
void destroy_21cmMC_HII_arrays();
void destroy_21cmMC_TsSaveBoxes_arrays();

#define HII_DIM_smooth (int) HII_DIM/1
#define f_smooth_factor (double) (1.0*HII_DIM/HII_DIM_smooth)


#define HII_D_smooth (unsigned long long) (HII_DIM_smooth)
#define HII_MIDDLE_smooth (HII_DIM_smooth/2)
#define HII_MID_smooth ((unsigned long long)HII_MIDDLE)

#define HII_TOT_NUM_PIXELS_smooth (unsigned long long)(HII_D_smooth*HII_D_smooth*HII_D_smooth)
#define HII_TOT_FFT_NUM_PIXELS_smooth ((unsigned long long)(HII_D_smooth*HII_D_smooth*2llu*(HII_MID_smooth+1llu)))
#define HII_KSPACE_NUM_PIXELS_smooth ((unsigned long long)(HII_D_smooth*HII_D_smooth*(HII_MID_smooth+1llu)))


#define HII_C_INDEX_smooth(x,y,z)((unsigned long long)((z)+(HII_MID_smooth+1llu)*((y)+HII_D_smooth*(x))))// for 3D complex array
#define HII_R_FFT_INDEX_smooth(x,y,z)((unsigned long long)((z)+2llu*(HII_MID_smooth+1llu)*((y)+HII_D_smooth*(x)))) // for 3D real array with the FFT padding
#define HII_R_INDEX_smooth(x,y,z)((unsigned long long)((z)+HII_D_smooth*((y)+HII_D_smooth*(x)))) // for 3D real array with no padding


//instead of defining a new array with lower number of pixels, you can use a SMOOTHING_FACTOR between 1 and 2:

//#define SMOOTHING_FACTOR 90.
#define SMOOTHING_FACTOR 1.2
//JBM: if 1.0 exactly does not smooth at all.
#define MAX_VALUE_T 200.0
//maximum value for T21 in mK before it caps it, to avoid super large numbers screwing with the smoothing.

#define SAVE_ENTIRE_BOX 0
//whether to save the entire box in addition to just the intersting slice.


// /*****  Adjust the complex conjugate relations for a real array  *****/
// //JBM:modified for a HII-sized array (UNUSED!)
// void adj_complex_conj_HII(fftwf_complex *box){
//   int i, j, k;
//
//   // corners
//   box[HII_C_INDEX(0,0,0)] = 0;
//   box[HII_C_INDEX(0,0,HII_MIDDLE)] = crealf(box[HII_C_INDEX(0,0,HII_MIDDLE)]);
//   box[HII_C_INDEX(0,HII_MIDDLE,0)] = crealf(box[HII_C_INDEX(0,HII_MIDDLE,0)]);
//   box[HII_C_INDEX(0,HII_MIDDLE,HII_MIDDLE)] = crealf(box[HII_C_INDEX(0,HII_MIDDLE,HII_MIDDLE)]);
//   box[HII_C_INDEX(HII_MIDDLE,0,0)] = crealf(box[HII_C_INDEX(HII_MIDDLE,0,0)]);
//   box[HII_C_INDEX(HII_MIDDLE,0,HII_MIDDLE)] = crealf(box[HII_C_INDEX(HII_MIDDLE,0,HII_MIDDLE)]);
//   box[HII_C_INDEX(HII_MIDDLE,HII_MIDDLE,0)] = crealf(box[HII_C_INDEX(HII_MIDDLE,HII_MIDDLE,0)]);
//   box[HII_C_INDEX(HII_MIDDLE,HII_MIDDLE,HII_MIDDLE)] = crealf(box[HII_C_INDEX(HII_MIDDLE,HII_MIDDLE,HII_MIDDLE)]);
//
//
//   // do entire i except corners
//   for (i=1; i<HII_MIDDLE; i++){
//     // just j corners
//     for (j=0; j<=HII_MIDDLE; j+=HII_MIDDLE){
//       for (k=0; k<=HII_MIDDLE; k+=HII_MIDDLE){
// 	box[HII_C_INDEX(i,j,k)] = conjf(box[HII_C_INDEX(HII_DIM-i,j,k)]);
//       }
//     }
//
//     // all of j
//     for (j=1; j<HII_MIDDLE; j++){
//       for (k=0; k<=HII_MIDDLE; k+=HII_MIDDLE){
// 	box[HII_C_INDEX(i,j,k)] = conjf(box[HII_C_INDEX(HII_DIM-i,HII_DIM-j,k)]);
// 	box[HII_C_INDEX(i,HII_DIM-j,k)] = conjf(box[HII_C_INDEX(HII_DIM-i,j,k)]);
//       }
//     }
//   } // end loop over i
//
//
//   // now the i corners
//   for (i=0; i<=HII_MIDDLE; i+=HII_MIDDLE){
//     for (j=1; j<HII_MIDDLE; j++){
//       for (k=0; k<=HII_MIDDLE; k+=HII_MIDDLE){
// 	box[HII_C_INDEX(i,j,k)] = conjf(box[HII_C_INDEX(i,HII_DIM-j,k)]);
//       }
//     }
//   } // end loop over remaining j
// }



int main(int argc, char ** filenameinput){



//we start by reading the ini file:
	if(filenameinput[1]==NULL){
		printf("Error, no input file specified. Use ./smooth_output_box FILENAME \n");
		return 0;
	}



  char filename[500];
  char dummy_string[500];
  FILE *F;

  int i, j, k;
  long ct;

  double trash;

  float *box;
  float *box2;
  float *smoothed_box;

  fftwf_plan plan;


  box = (float *) calloc(HII_TOT_FFT_NUM_PIXELS,sizeof(float));
  if (!box){
    fprintf(stderr, "Init.c: Error allocating memory for low-res box.\nAborting...\n");
     free(box);    fftwf_cleanup_threads();
    free_ps(); return -1;
  }


  box2 = (float *) calloc(HII_TOT_FFT_NUM_PIXELS,sizeof(float));
  if (!box2){
    fprintf(stderr, "Init.c: Error allocating memory for low-res box.\nAborting...\n");
     free(box2);    fftwf_cleanup_threads();
    free_ps(); return -1;
  }
  smoothed_box = (float *) malloc(sizeof(float)*HII_TOT_FFT_NUM_PIXELS_smooth);
  if (!smoothed_box){
    fprintf(stderr, "Init.c: Error allocating memory for lower-res smoothed_box.\nAborting...\n");
    free(smoothed_box);    fftwf_cleanup_threads();
    free_ps(); return -1;
  }

  fprintf(stderr, "Getting and writing box...\n");
  sprintf(filename, filenameinput[1]);
  F = fopen(filename, "rb");
  if (!F){
    fprintf(stderr, "Couldn't open file %s for reading\nAborting...\n", filename);
    free(box);  fclose(F); fftwf_cleanup_threads();
    return -1;
  }

  //F=fopen(filenameinput[1], "r");
  // for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
  //   fscanf(F,"%e %e %e %e \n", &trash,&trash,&trash, &box[ct]);
  // }
  // fclose(F);
  if (mod_fread(box, sizeof(float)*HII_TOT_NUM_PIXELS, 1, F)!=1){
    fprintf(stderr, "smooth_output_box.c: Read error occured!\n");
    free(box);  fclose(F); fftwf_cleanup_threads();
    return -1;
  }

//We have to add padding so that FFTW can deal with it
for (i=0; i<HII_DIM; i++){
  for (j=0; j<HII_DIM; j++){
    for (k=0; k<HII_DIM; k++){
      box2[HII_R_FFT_INDEX(i,j,k)] =  box[HII_R_INDEX(i,j,k)];
    }
  }
}

for (ct=0; ct<HII_TOT_FFT_NUM_PIXELS; ct++){
   box[ct] = box2[ct];
	 if(box[ct]>=MAX_VALUE_T){
		 box[ct]=MAX_VALUE_T;
	 }
   // printf("%e \n",box[ct]);
}
//we have to save it in box2 so that box is not erased...
  free(box2);





  plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)box, (fftwf_complex *)box, FFTW_ESTIMATE);
  fftwf_execute(plan);




  fprintf(stderr, "Filtering and sampling \n");

  if (SMOOTHING_FACTOR>1 || HII_DIM_smooth != HII_DIM){
		filter_smooth( (fftwf_complex *) box, 2, SMOOTHING_FACTOR*L_FACTOR*BOX_LEN/(HII_DIM_smooth+0.0));
  }
  //we have to use filter_smooth() since filter() assumes DIM->HII_DIM smoothing.

  fprintf(stderr, "Filtered! FFT'ing back to real space \n");


  // FFT back to real space
  plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)box, (float *)box, FFTW_ESTIMATE);
  fftwf_execute(plan);

  fprintf(stderr, "FFT'd \n");



  // for (ct=0; ct<HII_TOT_FFT_NUM_PIXELS; ct++){
  //    smoothed_box[ct] = box[ct]/HII_TOT_NUM_PIXELS;
  // }

  // now sample the filtered box
  for (i=0; i<HII_DIM_smooth; i++){
    for (j=0; j<HII_DIM_smooth; j++){
      for (k=0; k<HII_DIM_smooth; k++){
  //smoothed_box[HII_R_INDEX_smooth(i,j,k)] =
  // smoothed_box[HII_R_FFT_INDEX(i,j,k)] =
  //   box[HII_R_FFT_INDEX((unsigned long long)(i*f_smooth_factor+0.5),
  //              (unsigned long long)(j*f_smooth_factor+0.5),
  //              (unsigned long long)(k*f_smooth_factor+0.5))];
 smoothed_box[HII_R_FFT_INDEX_smooth(i,j,k)] =
 *((float *)box + HII_R_FFT_INDEX((unsigned long long)(i*f_smooth_factor+0.5),
              (unsigned long long)(j*f_smooth_factor+0.5),
              (unsigned long long)(k*f_smooth_factor+0.5)))/HII_TOT_NUM_PIXELS;
      }
    }
  }
//last factor accounts for the FFT conversion normalization.

  printf("And save to file \n");


  //JBM:we also save as a regular text file instead of binary:
  //sprintf(filename, "smoothed_readable_box.dat");
	if(SAVE_ENTIRE_BOX){
	  sprintf(filename, "%s_smoothed_box_%d.dat",filenameinput[1],(int) SMOOTHING_FACTOR);
	  F=fopen(filename, "w");
	  for (i=0; i<HII_DIM_smooth; i++){
	    for (j=0; j<HII_DIM_smooth; j++){
	      for (k=0; k<HII_DIM_smooth; k++){
			fprintf(F,"%e %e %e %e \n", i * BOX_LEN/(HII_DIM_smooth-1.0),
					j * BOX_LEN/(HII_DIM_smooth-1.0), k * BOX_LEN/(HII_DIM_smooth-1.0), (float) smoothed_box[HII_R_FFT_INDEX_smooth(i,j,k)]);
	      }
	    }
	  }
	  fclose(F);
	}


  //JBM:we also save the first slice because its easy:
//  sprintf(filename, "smoothed_readable_slice.dat");
  sprintf(filename, "%s_smoothed_slice_%d.dat",filenameinput[1],(int) SMOOTHING_FACTOR);
  F=fopen(filename, "w");
  for (i=0; i<1; i++){
    for (j=0; j<HII_DIM_smooth; j++){
      for (k=0; k<HII_DIM_smooth; k++){
    fprintf(F,"%e %e %e \n",
        j * BOX_LEN/(HII_DIM_smooth-1.0), k * BOX_LEN/(HII_DIM_smooth-1.0), (float) smoothed_box[HII_R_FFT_INDEX_smooth(i,j,k)]);
      }
    }
  }
  fclose(F);




  free(box);
  free(smoothed_box);





 return 1;



}
