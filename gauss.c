/*---------------------------------------------------------------------------*/
/*! 
 * \file   gauss_seidel.c
 * \author Wayne Joubert
 * \date   Mon Jul 27 11:38:40 EDT 2015
 * \brief  Code for block Gauss-Seidel solve with block wavefront method.
 * \note   Copyright (C) 2015 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */
/*---------------------------------------------------------------------------*/

/*=============================================================================

This code performs block Gauss-Seidel iteration to solve a specific linear
system of equations.  It is known that the Gauss-Seidel calculation has
recursive dependencies which are amenable to a wavefront method for parallel
solves.

The problem solved here is a multiple-dof 5-point finite difference stencil
problem on a regular 2D grid.  The Gauss-Seidel ordering used is induced by
aStandard lexicographical ordering applied to the grid unknowns.

=============================================================================*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

typedef double Float_t;

/*===========================================================================*/
/*---Timer function---*/

double get_time()
{
    struct timeval tv;
    int i = gettimeofday (&tv, NULL);
    return ( (double) tv.tv_sec +
             (double) tv.tv_usec * 1.e-6 );
}

/*===========================================================================*/
/*---Utility functions---*/

int min(int i, int j)
{
    return i < j ? i : j;
}

/*---------------------------------------------------------------------------*/

int max(int i, int j)
{
    return i > j ? i : j;
}

/*---------------------------------------------------------------------------*/

int ceil_(int i, int j)
{
    return (i + j - 1 ) / j;
}

/*===========================================================================*/
/*---Accessor functions to access elements of array with multidim index---*/
/*---For matrix entries---*/

Float_t* refm(Float_t* __restrict__ a, int iu, int ju, int ix, int iy,
              int nu, int ncellx, int ncelly)
{
    assert(iu >= 0 && iu < nu);
    assert(ju >= 0 && ju < nu);
    assert(ix >= 0 && ix < ncellx);
    assert(iy >= 0 && iy < ncelly);

    return &( a[ju+nu*(iu+nu*(ix+ncellx*iy))] );
}

/*---------------------------------------------------------------------------*/

const Float_t* const_refm(const Float_t* __restrict__ a,
                          int iu, int ju, int ix, int iy,
                          int nu, int ncellx, int ncelly)
{
    assert(iu >= 0 && iu < nu);
    assert(ju >= 0 && ju < nu);
    assert(ix >= 0 && ix < ncellx);
    assert(iy >= 0 && iy < ncelly);

    return &( a[ju+nu*(iu+nu*(ix+ncellx*iy))] );
}

/*===========================================================================*/
/*---Accessor functions to access elements of array with multidim index---*/
/*---For vector entries---*/

Float_t* refv(Float_t* __restrict__ a, int iu, int ix, int iy,
              int nu, int ncellx, int ncelly)
{
    assert(iu >= 0 && iu < nu);
    assert(ix >= 0 && ix < ncellx);
    assert(iy >= 0 && iy < ncelly);

    return &( a[iu+nu*(ix+ncellx*iy)] );
}

/*---------------------------------------------------------------------------*/

const Float_t* const_refv(const Float_t* __restrict__ a, int iu, int ix, int iy,
                          int nu, int ncellx, int ncelly)
{
    assert(iu >= 0 && iu < nu);
    assert(ix >= 0 && ix < ncellx);
    assert(iy >= 0 && iy < ncelly);

    return &( a[iu+nu*(ix+ncellx*iy)] );
}

/*===========================================================================*/
/*---5-point stencil operation applied at a gridcell---*/

void process_cell(int ix, int iy, int nu, int ncellx, int ncelly,
                        Float_t* __restrict__ vo,
                  const Float_t* __restrict__ vi,
                  const Float_t* __restrict__ an,
                  const Float_t* __restrict__ as,
                  const Float_t* __restrict__ ae,
                  const Float_t* __restrict__ aw)
{
    int iu = 0;
    int ju = 0;

    /*---South, east connections use "old" values, north, west use "new"---*/
    /*---The connection operation is matrix-vector product---*/

    for (iu=0; iu<nu; ++iu)
    {
        *refv( vo, iu, ix, iy, nu, ncellx, ncelly) = 0.;
        for (ju=0; ju<nu; ++ju)
        {
            *refv(      vo, iu,     ix,   iy,   nu, ncellx, ncelly) +=

            *const_refm(aw, iu, ju, ix,   iy,   nu, ncellx, ncelly) *
            *const_refv(vo,     ju, ix-1, iy,   nu, ncellx, ncelly) +

            *const_refm(an, iu, ju, ix,   iy,   nu, ncellx, ncelly) *
            *const_refv(vo,     ju, ix,   iy-1, nu, ncellx, ncelly) +

            *const_refm(ae, iu, ju, ix,   iy,   nu, ncellx, ncelly) *
            *const_refv(vi,     ju, ix+1, iy,   nu, ncellx, ncelly) +

            *const_refm(as, iu, ju, ix,   iy,   nu, ncellx, ncelly) *
            *const_refv(vi,     ju, ix,   iy+1, nu, ncellx, ncelly);
        }
    }
}

/*===========================================================================*/
/*---Perform Gauss-Seidel sweep with standard order of operations---*/

void solve_standard(int nu, int ncellx, int ncelly,
                          Float_t* __restrict__ vo,
                    const Float_t* __restrict__ vi,
                    const Float_t* __restrict__ an,
                    const Float_t* __restrict__ as,
                    const Float_t* __restrict__ ae,
                    const Float_t* __restrict__ aw)
{
    /*---Loop over interior gridcells---*/
    int iy = 0;
    for (iy=1; iy<ncelly-1; ++iy)
    {
        int ix = 0;
        for (ix=1; ix<ncellx-1; ++ix)
        {
            process_cell(ix, iy, nu, ncellx, ncelly, vo, vi, an, as, ae, aw);
        }
    }
}

/*===========================================================================*/
/*---Perform Gauss-Seidel sweep with wavefront order of operations---*/
/*---Note all required recursion dependencies are satisfied---*/

void solve_wavefronts(int nu, int ncellx, int ncelly,
                            Float_t* __restrict__ vo,
                      const Float_t* __restrict__ vi,
                      const Float_t* __restrict__ an,
                      const Float_t* __restrict__ as,
                      const Float_t* __restrict__ ae,
                      const Float_t* __restrict__ aw)
{
    const int nwavefronts = ncellx + ncelly - 2;
    int wavefront = 0;

    /*---Loop over wavefronts---*/
    for (wavefront=0; wavefront<nwavefronts; ++wavefront)
    {
        printf("wavefront %d: ", wavefront);
    
        const int ixmin = max(1, wavefront-ncelly+2);
        const int ixmax = min(ncellx-1, wavefront);
        int ix = 0;
        /*---Loop over gridcells in wavefront---*/
        for (ix=ixmin; ix<ixmax; ++ix)
        {
            const int iy = wavefront - ix;
            printf("(%d,%d) ", ix, iy);
            process_cell(ix, iy, nu, ncellx, ncelly, vo, vi, an, as, ae, aw);
        }
        printf("\n");
    }
}

/*===========================================================================*/
/*---Perform Gauss-Seidel sweep with block-wavefront order of operations---*/
/*---Note all required recursion dependencies are satisfied---*/

void solve_block_wavefronts(int nu, int ncellx, int ncelly,
                                  Float_t* __restrict__ vo,
                            const Float_t* __restrict__ vi,
                            const Float_t* __restrict__ an,
                            const Float_t* __restrict__ as,
                            const Float_t* __restrict__ ae,
                            const Float_t* __restrict__ aw)
{
    /*---Set block sizes---*/
    const int nbsizex = 20;
    const int nbsizey = 20;

    const int nbx = ceil_(ncellx, nbsizex);
    const int nby = ceil_(ncelly, nbsizey);

    const int nbwavefronts = nbx + nby - 1;
    int bwavefront = 0;

    /*---Loop over block wavefronts---*/
    for (bwavefront=0; bwavefront<nbwavefronts; ++bwavefront)
    {
        const int ibxmin = max(0, bwavefront-nby);
        const int ibxmax = min(nbx, bwavefront+1);
        int ibx = 0;
        /*---Loop over blocks in wavefront---*/
        for (ibx=ibxmin; ibx<ibxmax; ++ibx)
        {
            const int iby = bwavefront - ibx;

            const int iymin = max(1, iby*nbsizey);
            const int iymax = min(ncelly-1, (iby+1)*nbsizey);
            int iy = 0;
            /*---Loop over gridcells in block---*/
            for (iy=iymin; iy<iymax; ++iy)
            {
                const int ixmin = max(1, ibx*nbsizex);
                const int ixmax = min(ncellx-1, (ibx+1)*nbsizex);
                int ix = 0;
                for (ix=ixmin; ix<ixmax; ++ix)
                {
                    process_cell(ix, iy, nu, ncellx, ncelly, vo, vi,
                                 an, as, ae, aw);
                }
            }
        }
    }
}

#if 0

/*===========================================================================*/
/*---Experimental code to use OpenMP tasks---*/

void solve_block_wavefronts_tasks(int nu, int ncellx, int ncelly,
                                  Float_t* __restrict__ vo,
                            const Float_t* __restrict__ vi,
                            const Float_t* __restrict__ an,
                            const Float_t* __restrict__ as,
                            const Float_t* __restrict__ ae,
                            const Float_t* __restrict__ aw)
{
    const int nbsizex = 20;
    const int nbsizey = 20;

    const int nbx = ceil_(ncellx, nbsizex);
    const int nby = ceil_(ncelly, nbsizey);

    const int nbwavefronts = nbx + nby - 1;
    int bwavefront = 0;

#pragma omp parallel
#pragma omp single
    {

    for (ibx=0; ibx<nbx; ++ibx)
    {
    for (iby=0; iby<nby; ++iby)
    {
#pragma omp task \
  depend(in: *ref(vo, nbsizex*(ibx-1), nbsizey* iby,    ncellx, ncelly)) \
  depend(in: *ref(vo, nbsizex* ibx,    nbsizey*(iby-1), ncellx, ncelly)) \
  depend(out:*ref(vo, nbsizex* ibx,    nbsizey* iby,    ncellx, ncelly))
    {
#pragma omp taskgroup if(MULTILEVEL_TASKING)
    {
            const int iymin = max(1, iby*nbsizey);
            const int iymax = min(ncelly-1, (iby+1)*nbsizey);
            int iy = 0;
            for (iy=iymin; iy<iymax; ++iy)
            {
                const int ixmin = max(1, ibx*nbsizex);
                const int ixmax = min(ncellx-1, (ibx+1)*nbsizex);
                int ix = 0;
                for (ix=ixmin; ix<ixmax; ++ix)
                {
/*
May want to experiment with different action based on device used at runtime:
  - tasks vs. parallel for - via "if" clause
  - block sizes
*/
#pragma omp task \
  depend(in: *ref(vi, ix-1, iy,   ncellx, ncelly)) \
  depend(in: *ref(vi, ix,   iy-1, ncellx, ncelly)) \
  depend(out:*ref(vi, ix,   iy,   ncellx, ncelly)) \
  if(MULTILEVEL_TASKING)
    {
                    process_cell(ix, iy, nu, ncellx, ncelly, vo, vi,
                                 an, as, ae, aw);
    } /*---task---*/
                }
            }
    } /*---taskgroup---*/

    } /*---task---*/

    } /*---for---*/
    } /*---for---*/

    } /*---single---*/
}


#endif

enum{ solve_method_standard = 1,
      solve_method_wavefronts = 2,
      solve_method_block_wavefronts = 3 };

/*===========================================================================*/
/*---Driver---*/

int main( int argc, char** argv )
{
    /*---Settings---*/

    const int niterations = 2;
    const int ncellx = 5;
    const int ncelly = 4;
    const int nu = 10;

    const int solve_method = solve_method_wavefronts;

    /*
    const int solve_method = solve_method_standard;
    const int solve_method = solve_method_wavefronts;
    const int solve_method = solve_method_block_wavefronts;
    */

    /*---Allocations---*/

    Float_t* __restrict__ v1 = malloc(nu   *ncellx*ncelly*sizeof(Float_t));
    Float_t* __restrict__ v2 = malloc(nu   *ncellx*ncelly*sizeof(Float_t));
    Float_t* __restrict__ an = malloc(nu*nu*ncellx*ncelly*sizeof(Float_t));
    Float_t* __restrict__ as = malloc(nu*nu*ncellx*ncelly*sizeof(Float_t));
    Float_t* __restrict__ ae = malloc(nu*nu*ncellx*ncelly*sizeof(Float_t));
    Float_t* __restrict__ aw = malloc(nu*nu*ncellx*ncelly*sizeof(Float_t));

    /*---Initializations: interior---*/

    int iy = 0;
    int ix = 0;
    int iu = 0;
    int ju = 0;
    for (iy=1; iy<ncelly-1; ++iy)
    {
        for (ix=1; ix<ncellx-1; ++ix)
        {
            for (iu=0; iu<nu; ++iu)
            {
                for (ju=0; ju<nu; ++ju)
                {
                    *refm(an, iu, ju, ix, iy, nu, ncellx, ncelly) = 0.;
                    *refm(as, iu, ju, ix, iy, nu, ncellx, ncelly) = 0.;
                    *refm(ae, iu, ju, ix, iy, nu, ncellx, ncelly) = 0.;
                    *refm(aw, iu, ju, ix, iy, nu, ncellx, ncelly) = 0.;
                }
                /*---Initial guess is 1 on interior, 0 on bdry---*/
                /*---Assume right hand side is 0---*/
                *refv(v1, iu, ix, iy, nu, ncellx, ncelly) = 1.;
                *refv(v2, iu, ix, iy, nu, ncellx, ncelly) = 1.;
                /*---Constant coefficient Laplacian operator---*/
                *refm(an, iu, iu, ix, iy, nu, ncellx, ncelly) = .25;
                *refm(as, iu, iu, ix, iy, nu, ncellx, ncelly) = .25;
                *refm(ae, iu, iu, ix, iy, nu, ncellx, ncelly) = .25;
                *refm(aw, iu, iu, ix, iy, nu, ncellx, ncelly) = .25;
            }
        }
    }

    /*---Initializations: boundary---*/
    /*---Assume Dirichlet zero boundaries---*/

    for (iy=0; iy<ncelly; ++iy)
    {
        for (iu=0; iu<nu; ++iu)
        {
            *refv(v1, iu, 0,        iy, nu, ncellx, ncelly) = 0.;
            *refv(v2, iu, 0,        iy, nu, ncellx, ncelly) = 0.;
            *refv(v1, iu, ncellx-1, iy, nu, ncellx, ncelly) = 0.;
            *refv(v2, iu, ncellx-1, iy, nu, ncellx, ncelly) = 0.;
        }
    }

    for (ix=0; ix<ncellx; ++ix)
    {
        for (iu=0; iu<nu; ++iu)
        {
            *refv(v1, iu, ix, 0,        nu, ncellx, ncelly) = 0.;
            *refv(v2, iu, ix, 0,        nu, ncellx, ncelly) = 0.;
            *refv(v1, iu, ix, ncelly-1, nu, ncellx, ncelly) = 0.;
            *refv(v2, iu, ix, ncelly-1, nu, ncellx, ncelly) = 0.;
        }
    }

    /*---Iteration loop---*/

    double tbeg = get_time();

    int iteration = 0;
    for (iteration=0; iteration<niterations; ++iteration)
    {
        const Float_t* const __restrict__ vi = iteration%2 ? v1 : v2;
              Float_t* const __restrict__ vo = iteration%2 ? v2 : v1;

        if (solve_method==solve_method_standard)
            solve_standard(nu, ncellx, ncelly, vo, vi, an, as, ae, aw);

        if (solve_method==solve_method_wavefronts)
            solve_wavefronts(nu, ncellx, ncelly, vo, vi, an, as, ae, aw);

        if (solve_method==solve_method_block_wavefronts)
            solve_block_wavefronts(nu, ncellx, ncelly, vo, vi, an, as, ae, aw);
    }

    /*---Finish---*/

    double time = get_time() - tbeg;

    const Float_t* __restrict__ vfinal = niterations%2 ? v1 : v2;

    Float_t sumsq = 0.;

    for (iy=0; iy<ncelly; ++iy)
    {
        for (ix=0; ix<ncellx; ++ix)
        {
            for (iu=0; iu<nu; ++iu)
            {
                Float_t value = *const_refv(vfinal, iu, ix, iy,
                                            nu, ncellx, ncelly);
                sumsq += value * value;
            }
        }
    }

    printf("Unknowns %e sumsq %.16e time %.6f\n",
           (double)nu*(double)ncellx*(double)ncelly, (double)sumsq, time);

    /*---Deallocations---*/

    free(v1);
    free(v2);
    free(an);
    free(as);
    free(ae);
    free(aw);
}

/*===========================================================================*/