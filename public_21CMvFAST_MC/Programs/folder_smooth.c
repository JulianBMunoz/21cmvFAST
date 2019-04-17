#include "../Parameter_files/INIT_PARAMS.H"
#include "../Parameter_files/ANAL_PARAMS.H"
#include "../Parameter_files/Variables.h"
#include "bubble_helper_progs.c"
#include "heating_helper_progs.c"
#include "gsl/gsl_sf_erf.h"
#include "filter.c"

/*
This program smooths a folder of outputs to make it look prettier
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


//smooths over all files in the file filenameinput in the folder folder_boxes


#define folder_boxes "/Users/julian/Dropbox/Research/18/21cmBAO/Code/21CMMC-\
master/21CMMC_SourceCode/Programs/output_v_3_boxes_1Mpc/"



int main(int argc, char ** filenameinput){



//we start by reading the ini file:
	if(filenameinput[1]==NULL){
		printf("Error, no input file specified. Use ./folder_smooth FILENAME \n");
		return 0;
	}



  char filename[500];
  char dummy_string[500];
  FILE *F;



  sprintf(filename, filenameinput[1]);
  F = fopen(filename, "r");


  char file21name[500];

  char command[500];


	while (fgets (file21name , 500 , F) != NULL )
	{
		sprintf(filename,"%s%s",folder_boxes,file21name);

		sprintf(command, "./smooth_output_box %s", filename);

        system(command);

	}








  return 1;


}
