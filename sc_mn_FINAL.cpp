/* 	Run Multinest for Late-type stellar density profile analysis

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <string>
#include <float.h>
#include "multinest.h"
#include "gauss_legendre.h"
#include "ez_thread.hpp"
#include <iostream>
#include <iomanip>
#include <boost/config/user.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/numeric/quadrature/adaptive.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
using std::cin;
using std::cout;
using std::vector;
using std::ofstream;
using std::ifstream;
using std::ios;
using std::string;
using std::stoi;
using std::endl;
using namespace boost::numeric;

//define constant values
const double PI = 3.14159265358979;
const double mass = 3.960e6; //From my 14_06_18 align, with less RV
const double dist = 7828.0;
const double G = 6.6726e-8;
const double msun = 1.99e33;
const double sec_in_yr = 3.1557e7;
const double cm_in_au = 1.496e13;
const double cm_in_pc = 3.086e18;
const double km_in_pc = 3.086e13;
const double au_in_pc = 206265.0;
const double asy_to_kms = dist * cm_in_au / (1e5 * sec_in_yr);
const double as_to_km = dist * cm_in_au / (1e5);
const double GM = G * mass* msun;

std::mutex mutex2;
//number of threads running in parallel for likelihood calculations
int NThreads = 10;

//define limits of priors
double min_g = -3.0; //gamma, inner slope
double max_g = 2.0;
double min_d = 2.0; //delta, sharpness of break
double max_d = 10.0;
double min_a = 0.0; //alpha, outer slope
double max_a = 10.0;
double min_b = 5.0*dist*cm_in_au; //break radius, in cm
double max_b = 2.0*cm_in_pc;

double plate_scale = 0.00995;
double dxy = plate_scale * dist * cm_in_au; //dx or dy of a pixel

//define ints and vectors
vector<double> r2dv, r2dvm, arv,arve, amodv, like_returnv, starlikev, starlikevm, like_returnvm, Xgcows,Ygcows,Rgcows,Zmax_gcows,rho_gcows, pOldv, pOldvm, Zmax_star, Zmax_starm,norm_posv, lnl_v, lnl_vm;
vector<int> gcows_vrows, gcows_vcols;

double density_normvgc, density_normvm, max_r, Rcut, innerCut, outerCut, gmodv, almodv, demodv, brmodv;
int num_stars, num_gcows, num_maser, situation,nonRadial;



//Using external Gauss Legendre integrator


//function to integrate broken power law density profile in cylindrical coordinates
double density_intZcyl(double Zprime, void* data)
{
  double Rcyl = *(double *)data;
  double tmpvalue = pow((sqrt(Rcyl*Rcyl + Zprime*Zprime)/brmodv),demodv);
  return fabs(2.0*PI*Rcyl*pow((Rcyl*Rcyl + Zprime*Zprime)/(brmodv*brmodv),(gmodv/-2.0))*
	      pow((1.0+tmpvalue),((gmodv-almodv)/demodv)));
}


//function to integrate broken power law, calls density_intZcyl
//integrates along z axis for given R (x,y), as integrating over
//sphere altogether, limits of integration for z are set by R
//max_z = sqrt(max_r**2 - R**2)
//symmetry over z-axis so only integrate from 0 to max z
double density_intR(double Rprime, void* data)
{
  double max_Z = sqrt(max_r*max_r - Rprime * Rprime);
  double Rcyl = Rprime * 1.0; 
  return gauss_legendre(100,density_intZcyl,&Rcyl,0.0,max_Z);
}


//used to integrate over broken power law over z
//X and Y values need to be given (through data value)
//this funciton is used to integrate over observational footprint for normalization
//iii is used, also needed input, to select desired X and Y values
double density_intZ(double Zprime, void* data)
{
  int iii = *(int *)data;
  return fabs(dxy*dxy*pow((Xgcows[iii]*Xgcows[iii] + Ygcows[iii]*Ygcows[iii] + Zprime*Zprime)/(brmodv*brmodv),(gmodv/-2.0))*
	      pow((1.0+pow((sqrt(Xgcows[iii]*Xgcows[iii] + Ygcows[iii]*Ygcows[iii] + Zprime*Zprime)/brmodv),demodv)),
		  ((gmodv-almodv)/demodv)));
}


//calculates the likelihood for a given star, selected with iii index
//used to integrate over z for given star
//likelihood at given z = p(a_R | R,z,I) p(R,z | I)
//probability of radial acceleration given R and z and other information (such as mass and distance of supermassive black hole)
//times the probability of R and z given other information (like density profile, broken power law)
double star_likeZ(double z0modv, void* data)
{
  int iii = *(int *)data;
//as long as star has not been flagged that acceleration fit should not be trusted, acceleration component
//of likelihood is calulated (log L in this case)
  if (arve[iii] > 0.0)
    {
        amodv[iii] = -1.0*GM*r2dv[iii] / pow((sqrt(r2dv[iii]*r2dv[iii] + z0modv*z0modv)),3.0);
        lnl_v[iii] = -1.0*(arv[iii]-amodv[iii])*(arv[iii]-amodv[iii])/(2.0*arve[iii]*arve[iii]);
    }
  else
    {
        amodv[iii] = 0.0;
        lnl_v[iii] = 0.0;
    }

    //density component of likelihood for given star
  lnl_v[iii] -= gmodv * log(sqrt(r2dv[iii]*r2dv[iii]+z0modv*z0modv)/brmodv);
  lnl_v[iii] += ((gmodv-almodv)/demodv) * log(1.0+pow((sqrt(r2dv[iii]*r2dv[iii]+z0modv*z0modv)/brmodv),demodv));

  like_returnv[iii] = exp(lnl_v[iii]);
    
  if(isinf(like_returnv[iii])==1)
    {
        cout << "Got oo in summation for gcows stars" << endl;
        cout << "Star " << iii << " and R " << r2dv[iii] << " and Z " << z0modv << endl;
    }

  return like_returnv[iii];
}


//calculates likelihood for given star which just has position information, no acceleration
//this is the Schodel sample
//likelihood at given z = p(R,z | I)
//probability of R and z given other information (like density profile, broken power law)
double star_likeZmaser(double z0modv, void* data)
{
  int iii = *(int *)data;
  lnl_vm[iii] = -1.0 * gmodv * log(sqrt(r2dvm[iii]*r2dvm[iii]+z0modv*z0modv)/brmodv);
  lnl_vm[iii] += ((gmodv-almodv)/demodv) * log(1.0+pow((sqrt(r2dvm[iii]*r2dvm[iii]+z0modv*z0modv)/brmodv),demodv));
  like_returnvm[iii] = exp(lnl_vm[iii]);

  if(isinf(like_returnvm[iii])==1)
    {
      cout << "Got oo in summation for schodel stars";
      cout << endl;
    }
  return like_returnvm[iii];
}  





//functiont which puts all likelihood and normalization calculations together
//calculates the ln L which gets returned to Multinest
//Cube is vector of values from 0 to 1, to be used for prior, here all priors are flat
void LogLike(double *Cube, int &ndim, int &npars, double &lnew, void *context)
{
  lnew = 0.0;
  gmodv = Cube[0] * (max_g - min_g) + min_g;
    //if alpha, delta, and break radius are left as free parameters, situation param controls this
    //odd values, 1 or 3, leave all 4 parameters (including gamma) free
  if (fabs(remainder(situation,2)) > 0.0)
    {
      almodv = Cube[1] * (max_a - min_a) + min_a;
      demodv = Cube[2] * (max_d - min_d) + min_d;
      brmodv = Cube[3] * (max_b - min_b) + min_b; //flat prior on r_break
    }
  density_normvgc= 0.0;
  density_normvm = 0.0;
  double total_lnLv = 0.0;

    //nonRadial param controls whether stars in GCOWS sample are required to be in observational footprint
    //if set to 1 then this requirement is used
    //this set calculates the density normalization for this footprint
  if (nonRadial > 0)
    {
      ez_thread(threadnum, NThreads)
      {
          //cycle through all pixels which are in footprint and intregrate over z to get total density
          // over this observational area
          for(int i=threadnum; i<num_gcows; i+=NThreads)
          {
              int iii = i;

              if ((Rgcows[iii] < Rcut) & (Rgcows[iii] >= 0.0))
              {
                  rho_gcows[iii] = gauss_legendre(100,density_intZ,&iii,0.0,Zmax_gcows[iii]);
                  
                  if (rho_gcows[iii] <= 0.0)
                  {
                      cout << "Integration problem with GCOWS density norm" << endl;
                  }

              }
          }
      };
        for (auto &&elem : rho_gcows)
    density_normvgc += elem;
    }

  else
    {
        //if only using radial cut to define spatial extent of GCOWS sample, integrate using cylindrical coords
      density_normvgc += gauss_legendre(100,density_intR,NULL,0.0,Rcut);
      if (density_normvgc <= 0.0){cout << "GCOWS part of density, radial integration, is 0"<<endl;}
    }

    //situation also controls whether other sample, schodel is used
    //if situation param is set to 3 or 4 then Schodel (et al 2010) is used and density over that observational volume is calculated
  if (situation > 2)
    {
        //integrate in cylindrical coords
      double tmp = gauss_legendre(100,density_intR,NULL,innerCut,outerCut);
      if (tmp <= 0.0)
      {
          cout << "Schodel density norm integration is 0, something is wrong" << endl;
      }
        //add this to normalization calculated for GCOWS sample
      density_normvm += tmp;
    }

    //ez_thread is function written and provided by G. Martinez for the purpose of running calculations in parallel
  ez_thread(threadnum,NThreads)
    {
        //cycle through stars in GCOWS sample
      for(int i=threadnum; i<num_stars; i+=NThreads)
      {
          int iii = i;

          //marginalize over z or line of sight for each star
          starlikev[iii] = gauss_legendre(100,star_likeZ,&iii,0.0,Zmax_star[iii]);

          //divide by density normalization
          starlikev[iii] /= (density_normvgc);
          //weigh each star by it's probability of being late-type
          starlikev[iii] = pOldv[iii] * (log(starlikev[iii]));
          if(starlikev[iii] < -1e20){cout << "Gcow star, got lnL less than -1e20" << endl;}
          if(starlikev[iii] != starlikev[iii]){cout << "Gcow star, got nan value" << endl;}
          if(isinf(starlikev[iii])==1){cout << "Gcow star, got inf value" << endl;}
      }
    };
        for (auto &&elem : starlikev)
	total_lnLv += elem;
    
    //if situation is greater than 2 (either 3 or 4) Schodel sample is included in likelihood calculation
  if (situation > 2)
    {
      ez_thread(threadnum,NThreads)
        {
            //cycle through Schodel stars
            for(int i=threadnum; i<num_maser; i+=NThreads)
            {
                int iii = i;

                //marginalize over z, line of sight, for each star
                starlikevm[iii] = gauss_legendre(100,star_likeZmaser,&iii,0.0,Zmax_starm[iii]);

                //divide by normalization
                starlikevm[iii] /= density_normvm;
                //weighted by probability of being late-type
                starlikevm[iii] = pOldvm[iii] * (log(starlikevm[iii]));
                if(starlikevm[iii] < -1e20){cout << "Schodel star, got lnL less than -1e20" << endl;}
                if(starlikevm[iii] != starlikevm[iii]){cout << "Schodel star, got nan value" << endl;}
                if(isinf(starlikevm[iii])==1){cout << "Schodel star, got inf value" << endl;}
            }
        };
        for (auto &&elem : starlikevm)
    total_lnLv += elem;
    }

  lnew = total_lnLv * 1.0;

}



//dumper function is directly from example code provided for MultiNest, none of this function is written by S. Chappell
void dumper(int &nSamples, int &nlive, int &nPar, double **physLive, double **posterior, double **paramConstr, double &maxLogLike, double &logZ, double &INSlogZ, double &logZerr, void *context)
{
    // convert the 2D Fortran arrays to C++ arrays
    
    
    // the posterior distribution
    // postdist will have nPar parameters in the first nPar columns & loglike value & the posterior probability in the last two columns
    
    int i, j;
    
    double postdist[nSamples][nPar + 2];
    for( i = 0; i < nPar + 2; i++ )
    for( j = 0; j < nSamples; j++ )
    postdist[j][i] = posterior[0][i * nSamples + j];
    
    
    
    // last set of live points
    // pLivePts will have nPar parameters in the first nPar columns & loglike value in the last column
    
    double pLivePts[nlive][nPar + 1];
    for( i = 0; i < nPar + 1; i++ )
    for( j = 0; j < nlive; j++ )
    pLivePts[j][i] = physLive[0][i * nlive + j];
}


//Main function
int main(int argc, char *argv[])
{
  //Model info
  string tmp = argv[1];
  char root[100];
    //label info for output files
  strcpy(root,tmp.c_str());
  tmp = argv[11];
  char ending[100];
  strcpy(ending,tmp.c_str());
    //maximum r, or largest value integrated out to
  max_r = atof(argv[5]);
  max_r *= cm_in_pc;
    //projected radius cut, needed for density normalization
  Rcut = atof(argv[6]);
  Rcut *= cm_in_pc;
  situation = stoi(argv[7]);
  nonRadial = stoi(argv[10]);
    //if only gamma is left as free parameter (controlled by situation param)
    //fix other density model parameters to values inputted
  if (fabs(remainder(situation,2)) < 1.0)
    {
      cout << "Fixing alpha, delta, and r_break"<<endl;
      almodv = atof(argv[2]);
      demodv = atof(argv[3]);
      brmodv = atof(argv[4]);
      brmodv *= cm_in_pc;
    }
    //cut off radii for Schodel sample
  innerCut = atof(argv[8]) * cm_in_pc;
  outerCut = atof(argv[9]) * cm_in_pc;


  //Read in values for stars in GCOWS sample for dat file
  std::string gcowsfile;
  gcowsfile.append("stars_mn");
  gcowsfile.append(ending);
  gcowsfile.append(".dat");
  ifstream in(gcowsfile,ios::in);
  double tmp1,tmp2,tmp3,tmp4,tmpmin,tmpmax,tmpval;
  if (in.is_open())
    {
      while (!in.eof())
	{
	  in >> tmp1 >> tmp2 >> tmp3 >> tmp4;
	  r2dv.push_back(tmp1);
	  arv.push_back(tmp2);
	  arve.push_back(tmp3);
	  pOldv.push_back(tmp4);
      tmpval = sqrt(max_r*max_r - tmp1*tmp1);
	  Zmax_star.push_back(tmpval);
	  starlikev.push_back(0.0);
	  amodv.push_back(0.0);
	  like_returnv.push_back(0.0);
      lnl_v.push_back(0.0);
	}
    }
  in.close();
  num_stars = r2dv.size();

  if (situation > 2)
    {
      //Read in values for stars in maser field
      cout <<"Reading in stars in maser fields"<< endl;
      std::string maserfile;
      maserfile.append("maser_mn");
      maserfile.append(ending);
      maserfile.append(".dat");
      ifstream in(maserfile,ios::in);
      double tmp1,tmp2,tmp3,tmp4,tmpvalue;
      if (in.is_open())
	{
	  while (!in.eof())
	    {
	      in >> tmp1 >> tmp2 >> tmp3 >> tmp4;
	      r2dvm.push_back(tmp1);
	      pOldvm.push_back(tmp2);
	      tmpvalue = sqrt(max_r*max_r - tmp1*tmp1);
	      Zmax_starm.push_back(tmpvalue);
	      starlikevm.push_back(0.0);
	      like_returnvm.push_back(0.0);
          lnl_vm.push_back(0.0);
	    }
	}
      in.close();
      num_maser = r2dvm.size();
    }



  if (nonRadial > 0)
    {
      //Read in gcows field
      cout << "Reading in GCOWS field info"<< endl;
      ifstream in("gcows_field.dat",ios::in);
      double tmp1,tmp2, tmpx, tmpy, tmpR, tmpz;
      if (in.is_open())
	{
	  while(!in.eof())
	    {
	      in >> tmp1 >> tmp2;
	      gcows_vrows.push_back((int) tmp1);
	      gcows_vcols.push_back((int) tmp2);
	      tmpx = sqrt((tmp2-1500.0)*(tmp2-1500.0))*dxy;
	      tmpy = sqrt((tmp1-1500.0)*(tmp1-1500.0))*dxy;
	      tmpR = sqrt(tmpx*tmpx + tmpy*tmpy);
	      Xgcows.push_back(tmpx);
	      Ygcows.push_back(tmpy);
	      Rgcows.push_back(tmpR);
	      tmpz = sqrt(max_r*max_r - tmpR*tmpR);
	      Zmax_gcows.push_back(tmpz);
	      rho_gcows.push_back(0.0);
	    }
	}
      in.close();
      num_gcows = gcows_vrows.size();
    }
  
  std::string pfile;
  pfile.append(root);
  pfile.append("priors.txt");

  //Writing priors to file
  ofstream priors;
  priors.open(pfile);
  priors << "#Gamma priors:\n" << min_g << " " << max_g << "\n";
  if (fabs(remainder(situation,2)) > 0.0)
    {
      priors << "#Alpha priors:\n" << min_a << " " << max_a << "\n";
      priors << "#Delta priors:\n" << min_d << " " << max_d << "\n";
      priors << "#Break r priors (pc):\n" << min_b/cm_in_pc << " " << max_b/cm_in_pc << "\n";
    }
  priors.close();

	
  // set the MultiNest sampling parameters, comments from MultiNest example code
	
  int mmodal = 0;// do mode separation?
	
  int ceff = 0;// run in constant efficiency mode?
	
  int nlive = 1000;// number of live points
	
  double efr = 0.8;// set the required efficiency
	
  double tol = 0.5;// tol, defines the stopping criteria
	
  int ndims = 4;// dimensionality (no. of free parameters)
	
  int nPar = 4;// total no. of parameters including free & derived parameters
	
  int nClsPar = 4;// no. of parameters to do mode separation on

  if (fabs(remainder(situation,2)) < 1.0)
    {
      cout <<"Fixing alpha, delta, and r_break"<< "\n";
      ndims = 1;
      nPar = 1;
      nClsPar = 1;
    }
	
  int updInt = 100;// after how many iterations feedback is required & the output files should be updated
                    // note: posterior files are updated & dumper routine is called after every updInt*10 iterations
	
  double Ztol = -1E90;// all the modes with logZ < Ztol are ignored
	
  int maxModes = 100;// expected max no. of modes (used only for memory allocation)
	
  int pWrap[ndims];// which parameters to have periodic boundary conditions?
  for(int i = 0; i < ndims; i++) pWrap[i] = 0;
	
  int seed = -1;// random no. generator seed, if < 0 then take the seed from system clock
	
  int fb = 1;// need feedback on standard output?
	
  int resume = 0;// resume from a previous job?
	
  int outfile = 1;// write output files?
	
  int initMPI = 1;// initialize MPI routines?, relevant only if compiling with MPI
		  // set it to F if you want your main program to handle MPI initialization
	
  double logZero = -1E100;// points with loglike < logZero will be ignored by MultiNest
	
  int maxiter = 0;// max no. of iterations, a non-positive value means infinity. MultiNest will terminate if either it 
		  // has done max no. of iterations or convergence criterion (defined through tol) has been satisfied
	
  void *context = 0;// not required by MultiNest, any additional information user wants to pass
    
    int IS = 1;
    
	
  // calling MultiNest
	
  nested::run(IS, mmodal, ceff, nlive, tol, efr, ndims, nPar, nClsPar, maxModes, updInt, Ztol, root, seed, pWrap, fb, resume, outfile, initMPI,logZero, maxiter, LogLike, dumper, context);
}
